"""
app.py — Maddy AI Complete 5-Tab Personal Quant Terminal (Mobile + Desktop).
Deploy free from your phone via GitHub + Streamlit Community Cloud!
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from alerts import format_signal_message, send_telegram_alert
from backtest import (
    run_parameter_heatmap,
    run_portfolio_backtest,
    run_walk_forward_validation,
)
from data_feed import (
    BENCHMARK_TICKER,
    DEFAULT_WATCHLIST,
    BrokerBridge,
    fetch_ohlcv,
    generate_demo_ohlcv,
    get_sector,
    load_watchlist_data,
    safe_float,
)
from risk import (
    RiskConfig,
    calculate_position_size,
    compute_portfolio_heat,
    compute_sector_exposure,
    evaluate_trade_gate,
    run_block_bootstrap_mc,
)
from scanner import (
    add_indicators,
    check_nifty_regime,
    detect_fvg_zones,
    detect_market_gaps,
    scan_watchlist,
)


# ============================================================================
# SQLite Paper Trading Storage (Tab 5)
# ============================================================================
DB_PATH = "maddy_paper_trades.db"


def init_paper_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sector TEXT NOT NULL,
                setup TEXT,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target REAL NOT NULL,
                qty INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                exit_date TEXT,
                exit_price REAL,
                realized_pnl REAL DEFAULT 0.0
            )
            """
        )
        conn.commit()


def load_paper_trades() -> pd.DataFrame:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC", conn)


def add_paper_trade(
    ticker: str,
    sector: str,
    setup: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    qty: int,
) -> None:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (ticker, sector, setup, entry_date, entry_price, stop_loss, target, qty, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (
                ticker,
                sector,
                setup,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                round(float(entry_price), 2),
                round(float(stop_loss), 2),
                round(float(target), 2),
                int(qty),
            ),
        )
        conn.commit()


def close_paper_trade(trade_id: int, exit_price: float) -> None:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT entry_price, qty FROM paper_trades WHERE id = ?", (trade_id,))
        row = cur.fetchone()
        if not row:
            return
        entry_p, qty = float(row[0]), int(row[1])
        pnl = (float(exit_price) - entry_p) * qty
        conn.execute(
            """
            UPDATE paper_trades
            SET status = 'CLOSED', exit_date = ?, exit_price = ?, realized_pnl = ?
            WHERE id = ?
            """,
            (datetime.now().strftime("%Y-%m-%d %H:%M"), round(float(exit_price), 2), round(pnl, 2), trade_id),
        )
        conn.commit()


def reset_paper_db() -> None:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.commit()


# ============================================================================
# Page Setup & Mobile-Friendly Styling
# ============================================================================
st.set_page_config(
    page_title="Maddy AI — Personal Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 10px 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

init_paper_db()

# ============================================================================
# Collapsible Sidebar Controls
# ============================================================================
with st.sidebar:
    st.header("Maddy AI Controls")
    use_demo_mode = st.toggle(
        "Offline Demo Data Mode",
        value=False,
        help="Instant synthetic NSE-style candles if live yfinance is slow or unavailable.",
    )
    capital = st.number_input(
        "Portfolio Capital (₹)",
        min_value=10_000,
        max_value=50_000_000,
        value=100_000,
        step=10_000,
    )
    sl_mult = st.slider("Stop-Loss ATR Multiplier", 1.0, 3.0, 1.5, 0.1)
    tp_mult = st.slider("Take-Profit ATR Multiplier", 1.5, 6.0, 3.0, 0.25)
    period = st.selectbox("Backtest Window", ["1y", "2y", "3y"], index=1)

    watchlist_text = st.text_area(
        "NSE Watchlist (1 ticker per line)",
        value="\n".join(DEFAULT_WATCHLIST),
        height=170,
    )
    watchlist = [line.strip().upper() for line in watchlist_text.splitlines() if line.strip()]

    st.divider()
    st.subheader("Live Broker Bridge")
    broker_choice = st.selectbox("Broker Connector", BrokerBridge.SUPPORTED_BROKERS, index=0)
    broker = BrokerBridge(broker_choice)
    b_status = broker.status_summary()
    if b_status["connected"]:
        st.success(b_status["mode"])
    else:
        st.caption(f"**{b_status['mode']}**\n\n{b_status['message']}")

# ============================================================================
# Header & Data Load
# ============================================================================
st.title("Maddy AI — Personal Quant Terminal")
st.caption("Dual-Engine White-Box Quant Terminal · NSE / Nifty 50 · SMC (FVG + Gaps) · Institutional Risk Gate")

if use_demo_mode:
    nifty_df = generate_demo_ohlcv(BENCHMARK_TICKER, days=380, seed=42)
    stock_data = {t: generate_demo_ohlcv(t, days=380, seed=100 + i) for i, t in enumerate(watchlist)}
else:
    with st.spinner("Loading NSE market data..."):
        nifty_df = fetch_ohlcv(BENCHMARK_TICKER, period=period)
        stock_data = load_watchlist_data(watchlist, period=period)

if nifty_df.empty or not stock_data:
    st.warning("Live NSE data load nahi ho paya. Sidebar khol kar **Offline Demo Data Mode** ON karo aur turant test karo.")
    st.stop()

regime = check_nifty_regime(nifty_df)
risk_cfg = RiskConfig()

# Run backtest once per parameter set so Tab 2, Tab 3, Tab 4 share grounded results
bt_results = run_portfolio_backtest(
    stock_data=stock_data,
    nifty_df=nifty_df,
    initial_capital=float(capital),
    sl_mult=sl_mult,
    tp_mult=tp_mult,
    risk_config=risk_cfg,
)

# ============================================================================
# 5-Tab Dashboard Layout
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Live Command Center",
        "2. Smart Charts",
        "3. Backtest & Analytics",
        "4. Stress Testing Lab",
        "5. Paper Trade & Risk",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1: LIVE COMMAND CENTER (DAILY SCANNER)
# ----------------------------------------------------------------------------
with tab1:
    if regime["is_bullish"]:
        st.success(
            f"**Nifty 50 Regime: {regime['label']}** | Close: {regime['close']:,.2f} | EMA20: {regime['ema20']:,.2f} | EMA50: {regime['ema50']:,.2f}"
        )
    else:
        st.error(
            f"**Nifty 50 Regime: {regime['label']}** | Close: {regime['close']:,.2f} | EMA20: {regime['ema20']:,.2f} | EMA50: {regime['ema50']:,.2f}"
        )

    scan_df = scan_watchlist(
        stock_data=stock_data,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        regime_bullish=regime["is_bullish"],
    )

    if not scan_df.empty:
        scan_df["Qty (1% Risk)"] = scan_df.apply(
            lambda r: calculate_position_size(float(capital), r["Entry"], r["StopLoss"], risk_cfg)
            if r["Status"] == "READY"
            else 0,
            axis=1,
        )

        st.subheader("Live Signal Table")
        st.dataframe(scan_df, use_container_width=True, hide_index=True)

        ready_df = scan_df[scan_df["Status"] == "READY"]
        st.subheader(f"Exact Trade Cards ({len(ready_df)} READY)")

        if ready_df.empty:
            st.info("Abhi koi stock READY criteria pass nahi kar raha (ya Nifty 50 macro bearish hai).")
        else:
            cols = st.columns(min(3, len(ready_df)))
            for idx, (_, row) in enumerate(ready_df.iterrows()):
                qty = int(row["Qty (1% Risk)"])
                with cols[idx % len(cols)]:
                    with st.container(border=True):
                        st.markdown(f"### {row['Ticker']} · {row['Score']}/100")
                        st.caption(f"{row['Sector']} · {row['Setup']}")
                        st.metric("Entry Price", f"₹{row['Entry']:,.2f}")
                        st.write(f"**ATR Stop-Loss:** ₹{row['StopLoss']:,.2f}")
                        st.write(f"**Target:** ₹{row['Target']:,.2f}")
                        st.write(f"**1% Risk Qty:** `{qty}` shares")

                        b1, b2 = st.columns(2)
                        if b1.button("Paper Buy", key=f"pb_{row['Ticker']}", use_container_width=True):
                            if qty > 0:
                                add_paper_trade(
                                    ticker=row["Ticker"],
                                    sector=row["Sector"],
                                    setup=row["Setup"],
                                    entry_price=row["Entry"],
                                    stop_loss=row["StopLoss"],
                                    target=row["Target"],
                                    qty=qty,
                                )
                                st.toast(f"Paper trade logged for {row['Ticker']} ({qty} qty)!")
                        if b2.button("Telegram", key=f"tg_{row['Ticker']}", use_container_width=True):
                            msg = format_signal_message(row.to_dict(), qty)
                            ok, info = send_telegram_alert(msg)
                            if ok:
                                st.success(info)
                            else:
                                st.info(info)
                                st.code(msg, language="text")

# ----------------------------------------------------------------------------
# TAB 2: INTERACTIVE SMART CHARTS (PLOTLY VISUALS)
# ----------------------------------------------------------------------------
with tab2:
    c_sel, c_bars = st.columns([2, 1])
    selected_ticker = c_sel.selectbox("Select Stock for Smart Chart", options=list(stock_data.keys()))
    bars_to_show = c_bars.slider("Candles to Display", 40, 180, 90, 10)

    df_chart = add_indicators(stock_data[selected_ticker]).tail(bars_to_show)
    fvg_zones = detect_fvg_zones(df_chart, lookback=bars_to_show)
    gap_zones = detect_market_gaps(df_chart, lookback=bars_to_show)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name=selected_ticker,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["EMA20"],
            mode="lines",
            name="EMA20",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["EMA50"],
            mode="lines",
            name="EMA50",
            line=dict(color="#f59e0b", width=1.4, dash="dot"),
        )
    )

    last_dt = df_chart.index[-1]

    # Plot Bullish/Bearish FVG Boxes + 50% Midpoint Line
    for z in fvg_zones[-10:]:
        is_bull = z["type"] == "Bullish FVG"
        fill_col = "rgba(16, 185, 129, 0.16)" if is_bull else "rgba(239, 68, 68, 0.16)"
        line_col = "rgba(16, 185, 129, 0.65)" if is_bull else "rgba(239, 68, 68, 0.65)"

        fig.add_shape(
            type="rect",
            x0=z["start_date"],
            x1=last_dt,
            y0=z["bottom"],
            y1=z["top"],
            fillcolor=fill_col,
            line=dict(color=line_col, width=1),
        )
        # 50% Midpoint dashed line
        fig.add_shape(
            type="line",
            x0=z["start_date"],
            x1=last_dt,
            y0=z["midpoint"],
            y1=z["midpoint"],
            line=dict(color=line_col, width=1, dash="dash"),
        )

    # Plot Market Gap Zones
    for g in gap_zones[-6:]:
        g_col = "rgba(99, 102, 241, 0.18)" if g["direction"] == "Gap Up" else "rgba(249, 115, 22, 0.18)"
        fig.add_shape(
            type="rect",
            x0=g["prev_date"],
            x1=g["date"],
            y0=g["bottom"],
            y1=g["top"],
            fillcolor=g_col,
            line=dict(color="rgba(99, 102, 241, 0.7)", width=1),
        )

    # Overlay Backtest Buy / Sell Markers for this stock
    if "trade_log" in bt_results and not bt_results["trade_log"].empty:
        tlog = bt_results["trade_log"]
        sub_t = tlog[tlog["Ticker"] == selected_ticker]
        if not sub_t.empty:
            start_dt = df_chart.index[0]
            buy_x, buy_y, sell_x, sell_y = [], [], [], []
            for _, tr in sub_t.iterrows():
                ed = pd.Timestamp(tr["EntryDate"])
                xd = pd.Timestamp(tr["ExitDate"])
                if ed >= start_dt:
                    buy_x.append(ed)
                    buy_y.append(tr["Entry"])
                if xd >= start_dt:
                    sell_x.append(xd)
                    sell_y.append(tr["Exit"])
            if buy_x:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x,
                        y=buy_y,
                        mode="markers",
                        name="Buy Entry",
                        marker=dict(symbol="triangle-up", size=12, color="#10b981"),
                    )
                )
            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x,
                        y=sell_y,
                        mode="markers",
                        name="Sell Exit",
                        marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
                    )
                )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    c_fvg, c_gap = st.columns(2)
    with c_fvg:
        st.caption(f"Detected FVG Zones ({len(fvg_zones)}) — 50% midpoint shown as dashed line")
        if fvg_zones:
            st.dataframe(
                pd.DataFrame(fvg_zones[-6:])[["type", "bottom", "midpoint", "top", "mitigated_50", "is_fresh"]],
                use_container_width=True,
                hide_index=True,
            )
    with c_gap:
        st.caption(f"Detected Market Gaps ({len(gap_zones)})")
        if gap_zones:
            st.dataframe(
                pd.DataFrame(gap_zones[-6:])[["direction", "gap_class", "gap_pct", "vol_ratio", "filled"]],
                use_container_width=True,
                hide_index=True,
            )

# ----------------------------------------------------------------------------
# TAB 3: PORTFOLIO BACKTEST & ANALYTICS
# ----------------------------------------------------------------------------
with tab3:
    if "error" in bt_results:
        st.error(bt_results["error"])
    else:
        m = bt_results["metrics"]
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Final Equity", f"₹{m['FinalEquity']:,.0f}", f"{m['NetReturn%']}%")
        k2.metric("Max Drawdown", f"{m['MaxDrawdown%']}%")
        k3.metric("Win Rate", f"{m['WinRate%']}%")
        k4.metric("Profit Factor", f"{m['ProfitFactor']}")
        k5.metric("CAPM Alpha", f"{m['Alpha%']}%")
        k6.metric("Beta", f"{m['Beta']}")

        eq_df = bt_results["equity_curve"]
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=eq_df["Equity"], name="Maddy AI (₹)", line=dict(color="#4f46e5", width=2.4)),
            row=1,
            col=1,
        )
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=eq_df["NiftyNorm"], name="Nifty 50 Benchmark (₹)", line=dict(color="#94a3b8", width=1.8, dash="dot")),
            row=1,
            col=1,
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df.index,
                y=eq_df["Drawdown%"],
                name="Drawdown (%)",
                fill="tozeroy",
                line=dict(color="#ef4444", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig_eq.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig_eq, use_container_width=True)

        c_tl, c_rj = st.columns(2)
        with c_tl:
            st.subheader(f"Complete Trade Log ({m['TotalTrades']})")
            if not bt_results["trade_log"].empty:
                st.dataframe(bt_results["trade_log"], use_container_width=True, hide_index=True)
            else:
                st.info("No closed trades in this window.")
        with c_rj:
            st.subheader(f"Rejection Ledger ({m['Rejections']})")
            if not bt_results["rejection_ledger"].empty:
                st.dataframe(bt_results["rejection_ledger"], use_container_width=True, hide_index=True)
            else:
                st.info("No signals were rejected by the Institutional Risk Gate.")

# ----------------------------------------------------------------------------
# TAB 4: STRESS TESTING LAB (QUANT VALIDATION)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("1. Block-Bootstrap Monte Carlo (1,000 Simulations)")
    if "equity_curve" in bt_results:
        mc = run_block_bootstrap_mc(
            bt_results["equity_curve"]["Equity"],
            initial_capital=float(capital),
            n_sims=1000,
            block_size=5,
            ruin_dd_pct=risk_cfg.kill_switch_dd_pct,
        )
        if "error" in mc:
            st.warning(mc["error"])
        else:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Risk of Ruin (25% DD)", f"{mc['risk_of_ruin_pct']}%")
            mc2.metric("Median Max Drawdown", f"{mc['median_max_dd_pct']}%")
            mc3.metric("5th %ile Worst Equity", f"₹{mc['p5_final_equity']:,.0f}")
            mc4.metric("Median Final Equity", f"₹{mc['median_final_equity']:,.0f}")

            x_days = list(range(1, mc["n_days"] + 1))
            fig_mc = go.Figure()
            for path in mc["sample_paths"]:
                fig_mc.add_trace(
                    go.Scatter(x=x_days, y=path, mode="lines", line=dict(color="rgba(148,163,184,0.22)", width=1), showlegend=False)
                )
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p95_path"], name="95th %ile", line=dict(color="#10b981", width=2)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p50_path"], name="Median (50th)", line=dict(color="#4f46e5", width=2.5)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p5_path"], name="5th %ile Worst-Case", line=dict(color="#ef4444", width=2)))
            fig_mc.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Trading Days", yaxis_title="Equity (₹)")
            st.plotly_chart(fig_mc, use_container_width=True)

    st.divider()
    st.subheader("2. Parameter Stability Heatmap & Walk-Forward Optimization (WFO)")
    if st.button("Run Heatmap + Walk-Forward Validation", type="primary"):
        with st.spinner("Running SL x TP stability grid & WFO splits..."):
            hm_df = run_parameter_heatmap(stock_data, nifty_df, initial_capital=float(capital))
            wfo_df = run_walk_forward_validation(stock_data, nifty_df, initial_capital=float(capital), n_splits=3)

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=hm_df.values,
                x=list(hm_df.columns),
                y=list(hm_df.index),
                colorscale="RdYlGn",
                text=hm_df.values,
                texttemplate="%{text}%",
            )
        )
        fig_hm.update_layout(height=340, margin=dict(l=10, r=10, t=25, b=10), title="Net Return % (SL vs TP ATR Multipliers)")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("**Walk-Forward Optimization (In-Sample → Unseen Out-of-Sample)**")
        if not wfo_df.empty:
            st.dataframe(wfo_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select a 2y or 3y historical window for multi-fold Walk-Forward Validation.")
    else:
        st.caption("Tap the button above to compute the SL × TP Stability Heatmap and Walk-Forward Optimization folds (kept on-demand so mobile load stays fast).")

# ----------------------------------------------------------------------------
# TAB 5: LIVE PAPER TRADING & RISK MONITOR
# ----------------------------------------------------------------------------
with tab5:
    paper_df = load_paper_trades()
    open_df = paper_df[paper_df["status"] == "OPEN"] if not paper_df.empty else pd.DataFrame()
    closed_df = paper_df[paper_df["status"] == "CLOSED"] if not paper_df.empty else pd.DataFrame()

    latest_prices: Dict[str, float] = {
        t: safe_float(df["Close"].iloc[-1]) for t, df in stock_data.items() if not df.empty
    }

    realized_pnl = safe_float(closed_df["realized_pnl"].sum()) if not closed_df.empty else 0.0
    unrealized_pnl = 0.0
    open_pos_dict: Dict[str, dict] = {}

    if not open_df.empty:
        for _, r in open_df.iterrows():
            ltp = latest_prices.get(r["ticker"], float(r["entry_price"]))
            unrealized_pnl += (ltp - float(r["entry_price"])) * int(r["qty"])
            open_pos_dict[r["ticker"]] = {
                "entry": float(r["entry_price"]),
                "stop_loss": float(r["stop_loss"]),
                "qty": int(r["qty"]),
                "sector": r["sector"],
            }

    virtual_equity = float(capital) + realized_pnl + unrealized_pnl
    peak_virtual = max(float(capital), virtual_equity)
    dd_pct = ((peak_virtual - virtual_equity) / peak_virtual * 100.0) if peak_virtual > 0 else 0.0
    kill_switch_hit = dd_pct >= (risk_cfg.kill_switch_dd_pct * 100.0)

    port_heat_pct = compute_portfolio_heat(open_pos_dict, virtual_equity) * 100.0
    sector_exp = compute_sector_exposure(open_pos_dict, virtual_equity, latest_prices)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Virtual Portfolio Equity", f"₹{virtual_equity:,.2f}", f"₹{(realized_pnl + unrealized_pnl):+,.2f}")
    p2.metric("Portfolio Heat (Max 5%)", f"{port_heat_pct:.2f}%")
    p3.metric("Open Positions", f"{len(open_df)}")
    p4.metric("25% Kill-Switch", "HALTED" if kill_switch_hit else "ACTIVE · SAFE")

    # Live Risk Gauges
    g1, g2 = st.columns(2)
    with g1:
        fig_heat = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(port_heat_pct, 2),
                number={"suffix": "%"},
                title={"text": "Portfolio Heat (Cap: 5%)"},
                gauge={
                    "axis": {"range": [0, 8]},
                    "bar": {"color": "#ef4444" if port_heat_pct > 5 else "#10b981"},
                    "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 5},
                },
            )
        )
        fig_heat.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with g2:
        st.markdown("**Sector Exposure (Max 30% per Sector)**")
        if sector_exp:
            for sec, frac in sorted(sector_exp.items(), key=lambda x: x[1], reverse=True):
                pct = frac * 100.0
                st.write(f"**{sec}:** {pct:.1f}% / 30%")
                st.progress(min(1.0, pct / 30.0))
        else:
            st.info("No open paper positions yet. Click **Paper Buy** in Tab 1 or log a trade below.")

    # Open Positions Management
    st.subheader("Open Paper Positions")
    if open_df.empty:
        st.caption("Zero open positions.")
    else:
        for _, r in open_df.iterrows():
            ltp = latest_prices.get(r["ticker"], float(r["entry_price"]))
            u_pnl = (ltp - float(r["entry_price"])) * int(r["qty"])
            c_info, c_act = st.columns([4, 1])
            c_info.write(
                f"**{r['ticker']}** ({r['sector']}) · Qty `{r['qty']}` · Entry ₹{r['entry_price']:,.2f} · LTP ₹{ltp:,.2f} · SL ₹{r['stop_loss']:,.2f} · Target ₹{r['target']:,.2f} · **P&L: ₹{u_pnl:+,.2f}**"
            )
            if c_act.button("Close @ LTP", key=f"close_{r['id']}", use_container_width=True):
                close_paper_trade(int(r["id"]), ltp)
                st.rerun()

    # Closed Trades + CSV Backup + Reset
    with st.expander(f"Closed Paper Trades ({len(closed_df)}) & Backup"):
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
        if not paper_df.empty:
            st.download_button(
                "Download Paper Trade History (CSV)",
                data=paper_df.to_csv(index=False).encode("utf-8"),
                file_name="maddy_ai_paper_trades.csv",
                mime="text/csv",
            )
        if st.button("Reset Paper Portfolio"):
            reset_paper_db()
            st.rerun()
