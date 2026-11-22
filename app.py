"""
app.py — Maddy AI Complete 5-Tab Personal Quant Terminal (Mobile + Desktop).
Deploy free from your phone via GitHub + Streamlit Community Cloud!
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from alerts import format_signal_message, send_telegram_alert
from journal_sync import is_cloud_sync_configured, pull_journal_from_github, push_journal_to_github
from news_shield import get_news_shield
from backtest import (
    run_parameter_heatmap,
    run_portfolio_backtest,
    run_walk_forward_validation,
)
from data_feed import (
    BENCHMARK_TICKER,
    CRYPTO_BENCHMARK,
    DEFAULT_COMMODITIES_GLOBAL_WATCHLIST,
    DEFAULT_CRYPTO_WATCHLIST,
    GLOBAL_BENCHMARK,
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
    compute_sector_strength,
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
                qty REAL NOT NULL,
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
    qty: float,
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
                round(float(qty), 4),
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
        entry_p, qty = float(row[0]), float(row[1])
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


def restore_paper_trades_from_df(df: pd.DataFrame) -> int:
    init_paper_db()
    if df is None or df.empty:
        return 0
    cols = [
        "ticker", "sector", "setup", "entry_date", "entry_price",
        "stop_loss", "target", "qty", "status", "exit_date", "exit_price", "realized_pnl",
    ]
    valid = [c for c in cols if c in df.columns]
    if not {"ticker", "entry_price", "qty"}.issubset(set(valid)):
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM paper_trades")
        placeholders = ",".join(["?"] * len(valid))
        col_sql = ",".join(valid)
        for _, r in df.iterrows():
            vals = [None if pd.isna(r[c]) else r[c] for c in valid]
            conn.execute(f"INSERT INTO paper_trades ({col_sql}) VALUES ({placeholders})", vals)
        conn.commit()
    return len(df)


# ============================================================================
# Page Setup & Mobile-Friendly Styling
# ============================================================================
_LOGO_FILE = "logo.png" if os.path.exists("logo.png") else "📈"
st.set_page_config(
    page_title="Maddy AI — Personal Quant Terminal",
    page_icon=_LOGO_FILE,
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
    if st.button("Refresh Live Data Now", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    auto_refresh_choice = st.selectbox(
        "Auto-Refresh Interval",
        ["Off", "Every 5 min", "Every 15 min"],
        index=0,
        help="Automatically reloads fresh NSE prices while this tab stays open.",
    )
    use_demo_mode = st.toggle(
        "Offline Demo Data Mode",
        value=False,
        help="Instant synthetic NSE-style candles if live yfinance is slow or unavailable.",
    )
    market_mode = st.selectbox(
        "Market Mode (Multi-Asset Radar)",
        ["NSE Stocks + Metals (₹)", "Crypto 24/7 ($)", "Commodities & Global ($)"],
        index=0,
    )
    is_crypto = "Crypto" in market_mode
    is_global = "Commodities" in market_mode
    allow_frac = is_crypto or is_global
    curr = "$" if (is_crypto or is_global) else "₹"

    if is_crypto:
        bench_ticker = CRYPTO_BENCHMARK
        bench_name = "Bitcoin (BTC-USD)"
        max_atr_pct = 12.0
        default_wl = DEFAULT_CRYPTO_WATCHLIST
        def_cap = 1_000
    elif is_global:
        bench_ticker = GLOBAL_BENCHMARK
        bench_name = "S&P 500 (SPY)"
        max_atr_pct = 6.5
        default_wl = DEFAULT_COMMODITIES_GLOBAL_WATCHLIST
        def_cap = 5_000
    else:
        bench_ticker = BENCHMARK_TICKER
        bench_name = "Nifty 50"
        max_atr_pct = 4.5
        default_wl = DEFAULT_WATCHLIST
        def_cap = 100_000

    capital = st.number_input(
        f"Portfolio Capital ({curr})",
        min_value=100 if allow_frac else 10_000,
        max_value=50_000_000,
        value=def_cap,
        step=500 if allow_frac else 10_000,
        key=f"cap_{market_mode}",
    )
    strategy_mode = st.selectbox(
        "Strategy Engine Mode",
        ["Top-5 Momentum Rotation (Monthly Leaders)", "Alpha Growth (2.2:1 R:R + Chandelier Trail)", "High Accuracy (EMA200 + RSI2 Snapback)", "High Win-Rate (Pullback + Partial Lock)", "Balanced (Fixed TP)"],
        index=0,
        help="Alpha Growth filters for Nifty-outperforming RS Leaders, shifts SL to breakeven at +1.5R, and trails winners with a 2x ATR Chandelier stop.",
    )
    sl_mult = st.slider("Stop-Loss ATR Multiplier", 1.0, 3.0, 2.0, 0.1)
    tp_mult = st.slider("Take-Profit ATR Multiplier", 1.5, 6.0, 3.5, 0.25)
    period = st.selectbox("Backtest Window", ["1y", "2y", "3y"], index=2)

    watchlist_text = st.text_area(
        f"{market_mode.split(' (')[0]} Watchlist (1 ticker per line)",
        value="\n".join(default_wl),
        height=170,
        key=f"wl_{market_mode}",
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
if os.path.exists("logo.png"):
    c_logo, c_ttl = st.columns([1, 8])
    c_logo.image("logo.png", width=64)
    c_ttl.title("Maddy AI — Personal Quant Terminal")
else:
    st.title("Maddy AI — Personal Quant Terminal")
ist_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y · %I:%M %p IST")
st.caption(f"Dual-Engine White-Box Quant Terminal · {market_mode} · Benchmark: {bench_name} · Updated: {ist_time}")

if auto_refresh_choice != "Off":
    refresh_ms = 300_000 if "5" in auto_refresh_choice else 900_000
    components.html(
        f"<script>setTimeout(function(){{ window.parent.location.reload(); }}, {refresh_ms});</script>",
        height=0,
    )

if use_demo_mode:
    nifty_df = generate_demo_ohlcv(bench_ticker, days=380, seed=42)
    stock_data = {t: generate_demo_ohlcv(t, days=380, seed=100 + i) for i, t in enumerate(watchlist)}
else:
    with st.spinner("Loading NSE market data..."):
        nifty_df = fetch_ohlcv(bench_ticker, period=period)
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
    strategy_mode=strategy_mode,
    risk_config=risk_cfg,
    max_atr_pct=max_atr_pct,
    allow_fractional=allow_frac,
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
            f"**{bench_name} Regime: {regime['label']}** | Close: {regime['close']:,.2f} | EMA20: {regime['ema20']:,.2f} | EMA50: {regime['ema50']:,.2f}"
        )
    else:
        st.error(
            f"**{bench_name} Regime: {regime['label']}** | Close: {regime['close']:,.2f} | EMA20: {regime['ema20']:,.2f} | EMA50: {regime['ema50']:,.2f}"
        )

    scan_df = scan_watchlist(
        stock_data=stock_data,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        regime_bullish=regime["is_bullish"],
        nifty_roc20=regime.get("roc20", 0.0),
        max_atr_pct=max_atr_pct,
    )

    if not scan_df.empty:
        scan_df["Qty (1% Risk)"] = scan_df.apply(
            lambda r: calculate_position_size(float(capital), r["Entry"], r["StopLoss"], risk_cfg, allow_fractional=allow_frac)
            if r["Status"] == "READY"
            else 0,
            axis=1,
        )

        st.subheader("Live Signal Table")
        st.dataframe(scan_df, width="stretch", hide_index=True)

        # Upgrade #2: Sector Strength Radar (Top-Down Filter)
        sec_df = compute_sector_strength(scan_df)
        if not sec_df.empty:
            with st.expander("Sector Strength Radar (Top-Down Market Leaders)", expanded=True):
                bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in sec_df["Avg_RS%"]]
                fig_sec = go.Figure(
                    go.Bar(
                        x=sec_df["Avg_RS%"],
                        y=sec_df["Sector"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.2f}% ({ldr})" for v, ldr in zip(sec_df["Avg_RS%"], sec_df["Top_Leader"])],
                        textposition="auto",
                    )
                )
                fig_sec.update_layout(
                    height=max(240, 34 * len(sec_df)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title=f"Avg 20D Relative Strength vs {bench_name} (%)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_sec, width="stretch")

        ready_df = scan_df[scan_df["Status"] == "READY"]
        st.subheader(f"Exact Trade Cards ({len(ready_df)} READY)")

        if ready_df.empty:
            st.info("Abhi koi stock READY criteria pass nahi kar raha (ya Nifty 50 macro bearish hai).")
        else:
            cols = st.columns(min(3, len(ready_df)))
            for idx, (_, row) in enumerate(ready_df.iterrows()):
                qty = round(float(row["Qty (1% Risk)"]), 4) if allow_frac else int(row["Qty (1% Risk)"])
                with cols[idx % len(cols)]:
                    with st.container(border=True):
                        st.markdown(f"### {row['Ticker']} · {row['Score']}/100")
                        st.caption(f"{row['Sector']} · {row['Setup']}")
                        st.metric("Entry Price", f"{curr}{row['Entry']:,.2f}")
                        st.write(f"**ATR Stop-Loss:** {curr}{row['StopLoss']:,.2f}")
                        st.write(f"**Target:** {curr}{row['Target']:,.2f}")
                        st.write(f"**1% Risk Qty:** `{qty}` {'units' if allow_frac else 'shares'}")

                        b1, b2 = st.columns(2)
                        if b1.button("Paper Buy", key=f"pb_{row['Ticker']}", width="stretch"):
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
                        if b2.button("Telegram", key=f"tg_{row['Ticker']}", width="stretch"):
                            msg = format_signal_message(row.to_dict(), qty)
                            ok, info = send_telegram_alert(msg)
                            if ok:
                                st.success(info)
                            else:
                                st.info(info)
                                st.code(msg, language="text")

# ----------------------------------------------------------------------------
# TAB 2: INTERACTIVE SMART CHARTS (TRADINGVIEW PRO STYLE)
# ----------------------------------------------------------------------------
with tab2:
    c_sel, c_rng = st.columns([2, 1])
    selected_ticker = c_sel.selectbox("Select Symbol", options=list(stock_data.keys()))
    range_choice = c_rng.selectbox("Range", options=["1M (22D)", "3M (65D)", "6M (130D)", "1Y (252D)"], index=1)
    bars_map = {"1M (22D)": 22, "3M (65D)": 65, "6M (130D)": 130, "1Y (252D)": 252}
    bars_to_show = bars_map[range_choice]

    # TradingView-style compact layer toggles
    tg1, tg2, tg3, tg4 = st.columns(4)
    show_ema = tg1.checkbox("EMA 20/50", value=True, key="tv_ema")
    show_fvg = tg2.checkbox("FVG Zones", value=True, key="tv_fvg")
    show_gaps = tg3.checkbox("Market Gaps", value=True, key="tv_gaps")
    show_vol = tg4.checkbox("Volume", value=True, key="tv_vol")

    df_full = add_indicators(stock_data[selected_ticker])
    df_chart = df_full.tail(bars_to_show)
    fvg_zones = detect_fvg_zones(df_chart, lookback=bars_to_show)
    gap_zones = detect_market_gaps(df_chart, lookback=bars_to_show)

    # TradingView-style OHLC Live Header Strip
    last_bar = df_chart.iloc[-1]
    prev_bar = df_chart.iloc[-2] if len(df_chart) >= 2 else last_bar
    ltp = safe_float(last_bar["Close"])
    prev_c = safe_float(prev_bar["Close"], default=ltp)
    chg = ltp - prev_c
    chg_pct = (chg / prev_c * 100.0) if prev_c > 0 else 0.0
    chg_color = "#26a69a" if chg >= 0 else "#ef5350"
    st.markdown(
        f"""
        <div style="background:#131722;border:1px solid #2a2e39;border-radius:10px;padding:10px 14px;margin-bottom:8px;color:#d1d4dc;font-family:Inter,sans-serif;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:6px;">
            <div>
              <span style="font-size:17px;font-weight:700;color:#ffffff;">{selected_ticker}</span>
              <span style="font-size:12px;color:#787b86;margin-left:6px;">1D · {get_sector(selected_ticker)}</span>
            </div>
            <div>
              <span style="font-size:18px;font-weight:700;color:#ffffff;">{curr}{ltp:,.2f}</span>
              <span style="font-size:13px;font-weight:600;color:{chg_color};margin-left:6px;">{chg:+.2f} ({chg_pct:+.2f}%)</span>
            </div>
          </div>
          <div style="font-size:11px;color:#9598a1;margin-top:4px;">
            O: <b style="color:#d1d4dc;">{curr}{safe_float(last_bar['Open']):,.2f}</b> &nbsp;
            H: <b style="color:#26a69a;">{curr}{safe_float(last_bar['High']):,.2f}</b> &nbsp;
            L: <b style="color:#ef5350;">{curr}{safe_float(last_bar['Low']):,.2f}</b> &nbsp;
            C: <b style="color:#d1d4dc;">{curr}{ltp:,.2f}</b> &nbsp;
            RSI14: <b style="color:#f59e0b;">{safe_float(last_bar.get('RSI14', 50)):.1f}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_vol:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name=selected_ticker,
            increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        ),
        row=1,
        col=1,
    )

    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["EMA20"],
                mode="lines",
                name="EMA20",
                line=dict(color="#2962ff", width=1.8),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["EMA50"],
                mode="lines",
                name="EMA50",
                line=dict(color="#ff9800", width=1.4, dash="dot"),
            ),
            row=1,
            col=1,
        )

    if show_vol:
        vol_colors = [
            "rgba(38,166,154,0.55)" if c >= o else "rgba(239,83,80,0.55)"
            for o, c in zip(df_chart["Open"], df_chart["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["Volume"],
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    last_dt = df_chart.index[-1]

    # Plot Bullish/Bearish FVG Boxes + 50% Midpoint Line
    if show_fvg:
        for z in fvg_zones[-8:]:
            is_bull = z["type"] == "Bullish FVG"
            fill_col = "rgba(38, 166, 154, 0.16)" if is_bull else "rgba(239, 83, 80, 0.16)"
            line_col = "rgba(38, 166, 154, 0.65)" if is_bull else "rgba(239, 83, 80, 0.65)"
            fig.add_shape(
                type="rect",
                x0=z["start_date"],
                x1=last_dt,
                y0=z["bottom"],
                y1=z["top"],
                fillcolor=fill_col,
                line=dict(color=line_col, width=1),
                row=1,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=z["start_date"],
                x1=last_dt,
                y0=z["midpoint"],
                y1=z["midpoint"],
                line=dict(color=line_col, width=1, dash="dash"),
                row=1,
                col=1,
            )

    # Plot Market Gap Zones
    if show_gaps:
        for g in gap_zones[-6:]:
            g_col = "rgba(41, 98, 255, 0.20)" if g["direction"] == "Gap Up" else "rgba(255, 152, 0, 0.20)"
            fig.add_shape(
                type="rect",
                x0=g["prev_date"],
                x1=g["date"],
                y0=g["bottom"],
                y1=g["top"],
                fillcolor=g_col,
                line=dict(color="rgba(41, 98, 255, 0.75)", width=1),
                row=1,
                col=1,
            )

    # TradingView Horizontal Last-Price Dotted Line + Right-Axis Badge
    fig.add_hline(
        y=ltp,
        line_dash="dot",
        line_color=chg_color,
        line_width=1.2,
        annotation_text=f" {curr}{ltp:,.2f} ",
        annotation_position="right",
        annotation_bgcolor=chg_color,
        annotation_font_color="#ffffff",
        annotation_font_size=11,
        row=1,
        col=1,
    )

    # Overlay Backtest Buy / Sell Markers
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
                        name="Buy",
                        marker=dict(symbol="triangle-up", size=12, color="#26a69a", line=dict(color="#ffffff", width=1)),
                    ),
                    row=1,
                    col=1,
                )
            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x,
                        y=sell_y,
                        mode="markers",
                        name="Sell",
                        marker=dict(symbol="triangle-down", size=12, color="#ef5350", line=dict(color="#ffffff", width=1)),
                    ),
                    row=1,
                    col=1,
                )

    fig.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="Inter, sans-serif", size=11),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=58, t=16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
    )
    fig.update_xaxes(
        gridcolor="#1e222d",
        showline=True,
        linecolor="#2a2e39",
        showspikes=True,
        spikemode="across",
        spikecolor="#787b86",
        spikethickness=1,
        spikedash="dash",
    )
    fig.update_yaxes(
        side="right",
        gridcolor="#1e222d",
        showline=True,
        linecolor="#2a2e39",
        showspikes=True,
        spikemode="across",
        spikecolor="#787b86",
        spikethickness=1,
        spikedash="dash",
        row=1,
        col=1,
    )
    if show_vol:
        fig.update_yaxes(side="right", gridcolor="#1e222d", showticklabels=False, row=2, col=1)

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

    c_fvg, c_gap = st.columns(2)
    with c_fvg:
        st.caption(f"Detected FVG Zones ({len(fvg_zones)}) — 50% midpoint shown as dashed line")
        if fvg_zones:
            st.dataframe(
                pd.DataFrame(fvg_zones[-6:])[["type", "bottom", "midpoint", "top", "mitigated_50", "is_fresh"]],
                width="stretch",
                hide_index=True,
            )
    with c_gap:
        st.caption(f"Detected Market Gaps ({len(gap_zones)})")
        if gap_zones:
            st.dataframe(
                pd.DataFrame(gap_zones[-6:])[["direction", "gap_class", "gap_pct", "vol_ratio", "filled"]],
                width="stretch",
                hide_index=True,
            )

    # Upgrade #4: News & Earnings Risk Shield
    with st.expander(f"News & Earnings Risk Shield — {selected_ticker}", expanded=False):
        if use_demo_mode:
            st.caption("Turn off Offline Demo Data Mode to load live headlines & earnings dates.")
        else:
            ns = get_news_shield(selected_ticker)
            if ns["earnings_warning"]:
                st.warning(f"Upcoming Earnings Alert ({ns['earnings_date']}) — watch for overnight gap risk!")
            elif ns["earnings_date"]:
                st.info(f"Next Earnings Date: {ns['earnings_date']}")
            if ns["caution_count"] > 0:
                st.warning(f"{ns['caution_count']} headline(s) flagged with CAUTION keywords — check before entering.")
            if ns["headlines"]:
                for h in ns["headlines"]:
                    badge = "🔴" if h["tag"] == "CAUTION" else ("🟢" if h["tag"] == "POSITIVE" else "⚪")
                    if h["link"]:
                        st.markdown(f"{badge} **[{h['title']}]({h['link']})** — _{h['publisher']}_")
                    else:
                        st.markdown(f"{badge} **{h['title']}** — _{h['publisher']}_")
            else:
                st.caption("No recent headlines returned for this ticker right now.")

# ----------------------------------------------------------------------------
# TAB 3: PORTFOLIO BACKTEST & ANALYTICS
# ----------------------------------------------------------------------------
with tab3:
    if "error" in bt_results:
        st.error(bt_results["error"])
    else:
        m = bt_results["metrics"]
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Final Equity", f"{curr}{m['FinalEquity']:,.0f}", f"{m['NetReturn%']}% ({m.get('CAGR%', 0)}%/yr)")
        k2.metric("Max Drawdown", f"{m['MaxDrawdown%']}%")
        k3.metric("Win Rate", f"{m['WinRate%']}%")
        k4.metric("Profit Factor", f"{m['ProfitFactor']}", f"Payoff {m.get('PayoffRatio', 0)}x")
        k5.metric("CAPM Alpha", f"{m['Alpha%']}%")
        k6.metric("Beta", f"{m['Beta']}")

        eq_df = bt_results["equity_curve"]
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=eq_df["Equity"], name=f"Maddy AI ({curr})", line=dict(color="#4f46e5", width=2.4)),
            row=1,
            col=1,
        )
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=eq_df["NiftyNorm"], name=f"{bench_name} Benchmark ({curr})", line=dict(color="#94a3b8", width=1.8, dash="dot")),
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
        st.plotly_chart(fig_eq, width="stretch")

        if "yearly_breakdown" in bt_results and not bt_results["yearly_breakdown"].empty:
            st.subheader("Yearly Performance Breakdown (Maddy AI vs Benchmark)")
            st.dataframe(bt_results["yearly_breakdown"], width="stretch", hide_index=True)

        c_tl, c_rj = st.columns(2)
        with c_tl:
            st.subheader(f"Complete Trade Log ({m['TotalTrades']})")
            if not bt_results["trade_log"].empty:
                st.dataframe(bt_results["trade_log"], width="stretch", hide_index=True)
            else:
                st.info("No closed trades in this window.")
        with c_rj:
            st.subheader(f"Rejection Ledger ({m['Rejections']})")
            if not bt_results["rejection_ledger"].empty:
                st.dataframe(bt_results["rejection_ledger"], width="stretch", hide_index=True)
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
            mc3.metric("5th %ile Worst Equity", f"{curr}{mc['p5_final_equity']:,.0f}")
            mc4.metric("Median Final Equity", f"{curr}{mc['median_final_equity']:,.0f}")

            x_days = list(range(1, mc["n_days"] + 1))
            fig_mc = go.Figure()
            for path in mc["sample_paths"]:
                fig_mc.add_trace(
                    go.Scatter(x=x_days, y=path, mode="lines", line=dict(color="rgba(148,163,184,0.22)", width=1), showlegend=False)
                )
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p95_path"], name="95th %ile", line=dict(color="#10b981", width=2)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p50_path"], name="Median (50th)", line=dict(color="#4f46e5", width=2.5)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p5_path"], name="5th %ile Worst-Case", line=dict(color="#ef4444", width=2)))
            fig_mc.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Trading Days", yaxis_title=f"Equity ({curr})")
            st.plotly_chart(fig_mc, width="stretch")

    st.divider()
    st.subheader("2. Parameter Stability Heatmap & Walk-Forward Optimization (WFO)")
    if st.button("Run Heatmap + Walk-Forward Validation", type="primary"):
        with st.spinner("Running SL x TP stability grid & WFO splits..."):
            hm_df = run_parameter_heatmap(stock_data, nifty_df, initial_capital=float(capital), strategy_mode=strategy_mode, max_atr_pct=max_atr_pct, allow_fractional=allow_frac)
            wfo_df = run_walk_forward_validation(stock_data, nifty_df, initial_capital=float(capital), strategy_mode=strategy_mode, n_splits=3, max_atr_pct=max_atr_pct, allow_fractional=allow_frac)

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
        st.plotly_chart(fig_hm, width="stretch")

        st.markdown("**Walk-Forward Optimization (In-Sample → Unseen Out-of-Sample)**")
        if not wfo_df.empty:
            st.dataframe(wfo_df, width="stretch", hide_index=True)
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
            unrealized_pnl += (ltp - float(r["entry_price"])) * float(r["qty"])
            open_pos_dict[r["ticker"]] = {
                "entry": float(r["entry_price"]),
                "stop_loss": float(r["stop_loss"]),
                "qty": float(r["qty"]),
                "sector": r["sector"],
            }

    virtual_equity = float(capital) + realized_pnl + unrealized_pnl
    peak_virtual = max(float(capital), virtual_equity)
    dd_pct = ((peak_virtual - virtual_equity) / peak_virtual * 100.0) if peak_virtual > 0 else 0.0
    kill_switch_hit = dd_pct >= (risk_cfg.kill_switch_dd_pct * 100.0)

    port_heat_pct = compute_portfolio_heat(open_pos_dict, virtual_equity) * 100.0
    sector_exp = compute_sector_exposure(open_pos_dict, virtual_equity, latest_prices)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Virtual Portfolio Equity", f"{curr}{virtual_equity:,.2f}", f"{curr}{(realized_pnl + unrealized_pnl):+,.2f}")
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
        st.plotly_chart(fig_heat, width="stretch")

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
            u_pnl = (ltp - float(r["entry_price"])) * float(r["qty"])
            c_info, c_act = st.columns([4, 1])
            c_info.write(
                f"**{r['ticker']}** ({r['sector']}) · Qty `{r['qty']}` · Entry {curr}{r['entry_price']:,.2f} · LTP {curr}{ltp:,.2f} · SL {curr}{r['stop_loss']:,.2f} · Target {curr}{r['target']:,.2f} · **P&L: {curr}{u_pnl:+,.2f}**"
            )
            if c_act.button("Close @ LTP", key=f"close_{r['id']}", width="stretch"):
                close_paper_trade(int(r["id"]), ltp)
                st.rerun()

    # Closed Trades + CSV Backup + Reset
    with st.expander(f"Closed Paper Trades ({len(closed_df)}) & Permanent Cloud Journal"):
        if not closed_df.empty:
            st.dataframe(closed_df, width="stretch", hide_index=True)
        if not paper_df.empty:
            st.download_button(
                "Download Paper Trade History (CSV)",
                data=paper_df.to_csv(index=False).encode("utf-8"),
                file_name="maddy_ai_paper_trades.csv",
                mime="text/csv",
            )
        st.divider()
        st.markdown("**Permanent Cloud Journal (Survives Reboots)**")
        c_sync1, c_sync2 = st.columns(2)
        if c_sync1.button("Save Journal to GitHub Cloud", width="stretch"):
            ok_s, msg_s = push_journal_to_github(paper_df)
            st.success(msg_s) if ok_s else st.info(msg_s)
        if c_sync2.button("Restore from GitHub Cloud", width="stretch"):
            cloud_df, msg_c = pull_journal_from_github()
            if cloud_df is not None:
                n_rest = restore_paper_trades_from_df(cloud_df)
                st.success(f"{msg_c} ({n_rest} trades loaded)")
                st.rerun()
            else:
                st.info(msg_c)

        uploaded_csv = st.file_uploader("Or Restore from Saved CSV Backup", type=["csv"], key="restore_csv")
        if uploaded_csv is not None:
            try:
                imp_df = pd.read_csv(uploaded_csv)
                n_imp = restore_paper_trades_from_df(imp_df)
                st.success(f"Restored {n_imp} trades from CSV!")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

        if st.button("Reset Paper Portfolio"):
            reset_paper_db()
            st.rerun()
