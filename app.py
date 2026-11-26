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
from journal_sync import check_and_apply_gtt_exits, is_cloud_sync_configured, pull_journal_from_github, push_journal_to_github
from market_intel import compute_market_mood, get_company_health
from news_shield import fetch_market_news_feed, get_news_shield
from backtest import (
    get_current_top5_allocation,
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
                realized_pnl REAL DEFAULT 0.0,
                net_pnl_pct REAL DEFAULT 0.0,
                result TEXT DEFAULT 'OPEN'
            )
            """
        )
        for col_sql in (
            "ALTER TABLE paper_trades ADD COLUMN net_pnl_pct REAL DEFAULT 0.0",
            "ALTER TABLE paper_trades ADD COLUMN result TEXT DEFAULT 'OPEN'",
        ):
            try:
                conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def load_paper_trades() -> pd.DataFrame:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC", conn)



def jump_to_chart(ticker: str) -> None:
    st.session_state["_pending_chart_symbol"] = ticker
    st.session_state["_jump_to_chart_tab"] = True
    st.rerun()


def style_tradingview(fig: go.Figure, height: int = 440, right_axis: bool = True, show_crosshair: bool = True) -> go.Figure:
    """Apply consistent TradingView Pro Dark styling (#131722 canvas, crisp grids, right price axis)."""
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="Inter, sans-serif", size=11),
        hovermode="x unified" if show_crosshair else "closest",
        margin=dict(l=10, r=52 if right_axis else 14, t=30, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    if show_crosshair:
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
            side="right" if right_axis else "left",
            gridcolor="#1e222d",
            showline=True,
            linecolor="#2a2e39",
            showspikes=True,
            spikemode="across",
            spikecolor="#787b86",
            spikethickness=1,
            spikedash="dash",
        )
    else:
        fig.update_xaxes(gridcolor="#1e222d", linecolor="#2a2e39")
        fig.update_yaxes(gridcolor="#1e222d", linecolor="#2a2e39")
    return fig



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
    if is_cloud_sync_configured():
        try:
            push_journal_to_github(load_paper_trades())
        except Exception:
            pass


def close_paper_trade(trade_id: int, exit_price: float) -> None:
    init_paper_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT entry_price, qty FROM paper_trades WHERE id = ?", (trade_id,))
        row = cur.fetchone()
        if not row:
            return
        entry_p, qty = float(row[0]), float(row[1])
        exit_p = float(exit_price)
        charges = (entry_p * qty) * 0.0015  # 0.15% brokerage + STT + slippage
        net_pnl = round(((exit_p - entry_p) * qty) - charges, 2)
        net_pnl_pct = round((((exit_p - entry_p) / entry_p) * 100.0) - 0.15, 2) if entry_p > 0 else 0.0
        res_lbl = "WIN" if net_pnl > 0 else "LOSS"
        conn.execute(
            """
            UPDATE paper_trades
            SET status = 'CLOSED', exit_date = ?, exit_price = ?, realized_pnl = ?, net_pnl_pct = ?, result = ?
            WHERE id = ?
            """,
            (datetime.now().strftime("%Y-%m-%d %H:%M"), round(exit_p, 2), net_pnl, net_pnl_pct, res_lbl, trade_id),
        )
        conn.commit()
    if is_cloud_sync_configured():
        try:
            push_journal_to_github(load_paper_trades())
        except Exception:
            pass


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
        "stop_loss", "target", "qty", "status", "exit_date", "exit_price",
        "realized_pnl", "net_pnl_pct", "result",
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
      .block-container { padding-top: 0.6rem; padding-bottom: 2rem; }
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
if "cloud_journal_checked" not in st.session_state:
    st.session_state["cloud_journal_checked"] = True
    try:
        if is_cloud_sync_configured():
            _cdf, _ = pull_journal_from_github()
            if _cdf is not None and not _cdf.empty:
                restore_paper_trades_from_df(_cdf)
    except Exception:
        pass

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
    risk_per_trade_pct = st.slider(
        "Risk per Trade (%)",
        0.5, 2.5, 1.0, 0.25,
        help="Higher risk proportionally scales stock cap (up to 30%) and portfolio heat (up to 10%) while keeping the 25% Kill-Switch active.",
    )
    top_n_choice = st.selectbox(
        "Rotation Concentration (Top-N)",
        ["Top 5 (Balanced · 20% each)", "Top 3 (High Growth · 33% each)"],
        index=0,
    )
    top_n_val = 3 if "Top 3" in top_n_choice else 5
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
eff_stock_cap = min(0.30, 0.20 + max(0.0, risk_per_trade_pct - 1.0) * 0.08)
eff_heat_cap = min(0.10, max(0.05, (risk_per_trade_pct * 4.0) / 100.0))
risk_cfg = RiskConfig(
    risk_per_trade_pct=risk_per_trade_pct / 100.0,
    max_stock_alloc_pct=eff_stock_cap,
    max_portfolio_heat_pct=eff_heat_cap,
)

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
    top_n=top_n_val,
)

# ============================================================================
# 5-Tab Dashboard Layout
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🎯 Scan", "📊 Chart", "📈 Test", "🧪 Lab", "💼 Paper", "📰 News"]
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

    # Market Mood Meter (0-100) + Market Breadth Card (TradingView Dark Style)
    mood = compute_market_mood(
        stock_data=stock_data,
        bench_df=nifty_df,
        is_nse=(not is_crypto and not is_global),
        use_demo_mode=use_demo_mode,
    )
    br = mood["breadth"]
    # Tickertape-Style Semi-Circle Market Mood Index (MMI) Card
    import math as _math
    _sc = mood["score"]
    _angle_deg = 180.0 - (_sc / 100.0) * 180.0
    _rad = _math.radians(_angle_deg)
    _nx = 150 + 82 * _math.cos(_rad)
    _ny = 140 - 82 * _math.sin(_rad)
    _vix_pill = f"India VIX: <b>{mood['vix']}</b>" if mood.get("vix") else "Vol: <b>Active</b>"
    _gold_pill = f"Gold vs Bench: <b>{mood['gold_rel']:+.1f}%</b>" if mood.get("gold_rel") is not None else "Gold: <b>N/A</b>"

    m_col1, m_col2 = st.columns([1.25, 1])
    with m_col1:
        st.markdown(
            f"""
            <div style="background:#131722;border:1px solid #2a2e39;border-radius:14px;padding:14px 16px;color:#e2e8f0;font-family:Inter,sans-serif;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:13px;font-weight:700;letter-spacing:0.3px;color:#ffffff;">MARKET MOOD INDEX (MMI)</span>
                <span style="font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px;background:{mood['color']}22;color:{mood['color']};border:1px solid {mood['color']}66;">{mood['zone']}</span>
              </div>
              <div style="text-align:center;">
                <svg viewBox="0 0 300 162" style="width:100%;max-width:290px;height:auto;">
                  <!-- 4 Tickertape Color Arcs: Extreme Fear (Green) -> Fear (Yellow) -> Greed (Orange) -> Extreme Greed (Red) -->
                  <path d="M 40 140 A 110 110 0 0 1 85.3 51.0" fill="none" stroke="#26a69a" stroke-width="18" stroke-linecap="butt"/>
                  <path d="M 88.5 48.8 A 110 110 0 0 1 148.0 30.0" fill="none" stroke="#facc15" stroke-width="18" stroke-linecap="butt"/>
                  <path d="M 152.0 30.0 A 110 110 0 0 1 211.5 48.8" fill="none" stroke="#fb923c" stroke-width="18" stroke-linecap="butt"/>
                  <path d="M 214.7 51.0 A 110 110 0 0 1 260 140" fill="none" stroke="#ef5350" stroke-width="18" stroke-linecap="butt"/>
                  <!-- Zone Labels -->
                  <text x="26" y="156" fill="#26a69a" font-size="9" font-weight="600">0 Fear</text>
                  <text x="135" y="16" fill="#94a3b8" font-size="9" font-weight="600">50</text>
                  <text x="234" y="156" fill="#ef5350" font-size="9" font-weight="600">Greed 100</text>
                  <!-- Needle -->
                  <line x1="150" y1="140" x2="{_nx:.1f}" y2="{_ny:.1f}" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round"/>
                  <circle cx="150" cy="140" r="8" fill="{mood['color']}" stroke="#ffffff" stroke-width="2.5"/>
                  <!-- Center Score -->
                  <text x="150" y="114" text-anchor="middle" fill="{mood['color']}" font-size="30" font-weight="800">{_sc}</text>
                </svg>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;font-size:11px;color:#cbd5e1;margin-top:2px;">
                <span style="background:#1e222d;padding:3px 8px;border-radius:6px;">Breadth: <b>{br['pct_ema50']}%</b></span>
                <span style="background:#1e222d;padding:3px 8px;border-radius:6px;">Momentum: <b>{mood.get('mom_score', 50)}</b></span>
                <span style="background:#1e222d;padding:3px 8px;border-radius:6px;">{_vix_pill}</span>
                <span style="background:#1e222d;padding:3px 8px;border-radius:6px;">{_gold_pill}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        with st.container(border=True):
            st.markdown("#### Market Breadth (Asli Tezi)")
            st.metric(
                "Assets Above EMA50",
                f"{br['above_ema50']} / {br['total']} ({br['pct_ema50']}%)",
                br["label"],
            )
            st.progress(min(1.0, br["pct_ema50"] / 100.0))
            st.caption(mood["tip"])

        # Upgrade #2: This Month's Top-5 Momentum Action Card (100% synced with Tab 3)
    top5_plan = get_current_top5_allocation(
        stock_data=stock_data,
        nifty_df=nifty_df,
        capital=float(capital),
        allow_fractional=allow_frac,
    )
    with st.container(border=True):
        st.subheader(f"This Month's Top-{top_n_val} Portfolio Action Plan")
        if top5_plan["shield_active"]:
            st.warning(f"**Status:** {top5_plan['mode']}")
        else:
            st.success(f"**Status:** {top5_plan['mode']}")
        if not top5_plan["table"].empty:
            st.dataframe(top5_plan["table"], width="stretch", hide_index=True)
            st.caption(f"Monthly Rebalance Guide: Allocate {curr}{float(capital)*(1.0/top_n_val):,.0f} ({int(100/top_n_val)}%) per leader · Daily Emergency Stop at -10%.")
        else:
            st.info("No active Top-5 allocations right now — keep capital protected in Cash.")

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
        st.caption("Kisi bhi stock ki row par tap karo ya neeche **📉 Chart** dabao — seedha Smart Chart mein khulega.")
        _tbl_evt = st.dataframe(
            scan_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="scan_table_sel",
        )
        _sel_rows = _tbl_evt.selection.rows if (_tbl_evt and hasattr(_tbl_evt, "selection")) else []
        if _sel_rows:
            _picked_t = str(scan_df.iloc[_sel_rows[0]]["Ticker"])
            if st.session_state.get("_last_table_picked") != _picked_t:
                st.session_state["_last_table_picked"] = _picked_t
                jump_to_chart(_picked_t)

        # Upgrade #2: Sector Strength Radar (Top-Down Filter)
        sec_df = compute_sector_strength(scan_df)
        if not sec_df.empty:
            with st.expander("Sector Strength Radar (Top-Down Market Leaders)", expanded=True):
                bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in sec_df["Avg_RS%"]]
                fig_sec = go.Figure(
                    go.Bar(
                        x=sec_df["Avg_RS%"],
                        y=sec_df["Sector"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.2f}% ({ldr})" for v, ldr in zip(sec_df["Avg_RS%"], sec_df["Top_Leader"])],
                        textposition="auto",
                        textfont=dict(color="#ffffff"),
                    )
                )
                style_tradingview(fig_sec, height=max(260, 36 * len(sec_df)), right_axis=False, show_crosshair=False)
                fig_sec.update_layout(
                    xaxis_title=f"Avg 20D Relative Strength vs {bench_name} (%)",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=14, t=14, b=14),
                )
                st.plotly_chart(fig_sec, width="stretch", config={"displaylogo": False})

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
                        pos_cost = qty * float(row['Entry'])
                        max_risk_amt = qty * max(0.0, float(row['Entry']) - float(row['StopLoss']))
                        est_reward_amt = qty * max(0.0, float(row['Target']) - float(row['Entry']))
                        st.write(f"**ATR Stop-Loss:** {curr}{row['StopLoss']:,.2f} · **Target:** {curr}{row['Target']:,.2f}")
                        st.write(f"**{risk_per_trade_pct:.2f}% Risk Qty:** `{qty}` {'units' if allow_frac else 'shares'} (Cost: **{curr}{pos_cost:,.0f}**)")
                        st.caption(f"Max Risk: {curr}{max_risk_amt:,.0f} · Est. Reward: {curr}{est_reward_amt:,.0f}")

                        b1, b2, b3 = st.columns(3)
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
                        if b3.button("📉 Chart", key=f"ch_{row['Ticker']}", width="stretch"):
                            jump_to_chart(str(row["Ticker"]))

# ----------------------------------------------------------------------------
# TAB 2: INTERACTIVE SMART CHARTS (TRADINGVIEW PRO STYLE)
# ----------------------------------------------------------------------------
with tab2:
    c_sel, c_rng = st.columns([2, 1])
    _avail_syms = list(stock_data.keys())
    if "_pending_chart_symbol" in st.session_state:
        _pend_sym = st.session_state.pop("_pending_chart_symbol")
        if _pend_sym in _avail_syms:
            st.session_state["chart_symbol_select"] = _pend_sym
    if st.session_state.get("chart_symbol_select") not in _avail_syms and _avail_syms:
        st.session_state["chart_symbol_select"] = _avail_syms[0]
    selected_ticker = c_sel.selectbox("Select Symbol", options=_avail_syms, key="chart_symbol_select")
    range_choice = c_rng.selectbox("Range", options=["1M (22D)", "3M (65D)", "6M (130D)", "1Y (252D)"], index=1)
    bars_map = {"1M (22D)": 22, "3M (65D)": 65, "6M (130D)": 130, "1Y (252D)": 252}
    bars_to_show = bars_map[range_choice]

    # TradingView-style compact layer toggles
    tg1, tg2, tg3, tg4, tg5 = st.columns(5)
    show_ema = tg1.checkbox("EMA 20/50", value=True, key="tv_ema")
    show_fvg = tg2.checkbox("FVG Zones", value=True, key="tv_fvg")
    show_gaps = tg3.checkbox("Market Gaps", value=True, key="tv_gaps")
    show_vol = tg4.checkbox("Volume", value=True, key="tv_vol")
    show_levels = tg5.checkbox("TP / SL", value=True, key="tv_levels")

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

    # Plot Bullish/Bearish FVG Boxes + 50% Midpoint Line (only 3 most recent fresh zones)
    if show_fvg:
        fresh_fvgs = [z for z in fvg_zones if z.get("is_fresh", True)][-3:]
        for z in fresh_fvgs:
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

    # Plot Market Gap Zones (only 2 most recent unfilled gaps)
    if show_gaps:
        unfilled_gaps = [g for g in gap_zones if not g.get("filled", False)][-2:]
        for g in unfilled_gaps:
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

    # Active Paper Trade (or READY Signal) Entry / SL / TP Lines
    if show_levels:
        _chart_trade_level = None
        try:
            _p_all = load_paper_trades()
            if not _p_all.empty:
                _p_open = _p_all[
                    (_p_all["ticker"] == selected_ticker)
                    & (_p_all["status"].astype(str).str.upper() == "OPEN")
                ]
                if not _p_open.empty:
                    _tr = _p_open.iloc[0]
                    _chart_trade_level = {
                        "mode": "OPEN",
                        "entry": safe_float(_tr["entry_price"]),
                        "sl": safe_float(_tr["stop_loss"]),
                        "tp": safe_float(_tr["target"]),
                        "qty": safe_float(_tr["qty"]),
                    }
        except Exception:
            _chart_trade_level = None

        if _chart_trade_level is None and not scan_df.empty:
            _s_match = scan_df[(scan_df["Ticker"] == selected_ticker) & (scan_df["Status"] == "READY")]
            if not _s_match.empty:
                _sr = _s_match.iloc[0]
                _chart_trade_level = {
                    "mode": "READY",
                    "entry": safe_float(_sr["Entry"]),
                    "sl": safe_float(_sr["StopLoss"]),
                    "tp": safe_float(_sr["Target"]),
                    "qty": safe_float(_sr.get("Qty (1% Risk)", 0)),
                }

        if _chart_trade_level is not None and _chart_trade_level["entry"] > 0:
            _is_open = _chart_trade_level["mode"] == "OPEN"
            _e_lbl = "Entry" if _is_open else "Plan Entry"
            _sl_lbl = "SL" if _is_open else "Plan SL"
            _tp_lbl = "TP" if _is_open else "Plan TP"

            if _chart_trade_level["tp"] > 0:
                fig.add_hline(
                    y=_chart_trade_level["tp"],
                    line_dash="dash",
                    line_color="#26a69a",
                    line_width=1.6,
                    annotation_text=f" {_tp_lbl} {curr}{_chart_trade_level['tp']:,.2f} ",
                    annotation_position="top left",
                    annotation_bgcolor="rgba(38, 166, 154, 0.90)",
                    annotation_font_color="#ffffff",
                    annotation_font_size=10,
                    row=1,
                    col=1,
                )
            fig.add_hline(
                y=_chart_trade_level["entry"],
                line_dash="solid" if _is_open else "dot",
                line_color="#2962ff",
                line_width=1.6,
                annotation_text=f" {_e_lbl} {curr}{_chart_trade_level['entry']:,.2f} ",
                annotation_position="top left",
                annotation_bgcolor="rgba(41, 98, 255, 0.90)",
                annotation_font_color="#ffffff",
                annotation_font_size=10,
                row=1,
                col=1,
            )
            if _chart_trade_level["sl"] > 0:
                fig.add_hline(
                    y=_chart_trade_level["sl"],
                    line_dash="dash",
                    line_color="#ef5350",
                    line_width=1.6,
                    annotation_text=f" {_sl_lbl} {curr}{_chart_trade_level['sl']:,.2f} ",
                    annotation_position="bottom left",
                    annotation_bgcolor="rgba(239, 83, 80, 0.90)",
                    annotation_font_color="#ffffff",
                    annotation_font_size=10,
                    row=1,
                    col=1,
                )
            if _is_open:
                _live_pnl = (ltp - _chart_trade_level["entry"]) * _chart_trade_level["qty"]
                st.caption(
                    f"📌 **Active Paper Position:** Qty `{_chart_trade_level['qty']}` · "
                    f"Entry `{curr}{_chart_trade_level['entry']:,.2f}` · "
                    f"SL `{curr}{_chart_trade_level['sl']:,.2f}` · "
                    f"TP `{curr}{_chart_trade_level['tp']:,.2f}` · "
                    f"Live P&L: **{curr}{_live_pnl:+,.2f}**"
                )
            else:
                st.caption(
                    f"🎯 **READY Signal Levels:** Plan Entry `{curr}{_chart_trade_level['entry']:,.2f}` · "
                    f"SL `{curr}{_chart_trade_level['sl']:,.2f}` · "
                    f"TP `{curr}{_chart_trade_level['tp']:,.2f}`"
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

    # Auto-pad Y-axis so TP & SL lines are never cut off at top/bottom
    _y_lows = [float(df_chart["Low"].min())]
    _y_highs = [float(df_chart["High"].max())]
    if show_levels and "_chart_trade_level" in locals() and _chart_trade_level is not None:
        if _chart_trade_level.get("sl", 0) > 0:
            _y_lows.append(float(_chart_trade_level["sl"]))
        if _chart_trade_level.get("tp", 0) > 0:
            _y_highs.append(float(_chart_trade_level["tp"]))
    _ymin, _ymax = min(_y_lows), max(_y_highs)
    _ypad = max((_ymax - _ymin) * 0.08, _ymax * 0.01)

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="Inter, sans-serif", size=11),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=68, t=18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
    )
    fig.update_yaxes(range=[_ymin - _ypad, _ymax + _ypad], row=1, col=1)
    if not is_crypto:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
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

    # Company Health Badge (Fundamental Check)
    with st.expander(f"Company Health Badge — {selected_ticker}", expanded=True):
        if use_demo_mode:
            st.caption("Turn off Offline Demo Data Mode to view live company financial ratios.")
        else:
            ch = get_company_health(selected_ticker)
            if ch.get("applicable"):
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Health Verdict", ch["badge"], f"Score {ch['score']}/100")
                hc2.metric("P/E Ratio", f"{ch['pe']}" if ch.get("pe") else "N/A")
                hc3.metric("ROE %", f"{ch['roe']}%" if ch.get("roe") is not None else "N/A")
                hc4.metric("Debt / Equity", f"{ch['dte']}%" if ch.get("dte") is not None else "N/A")
                if ch.get("strengths"):
                    st.caption("Strengths: " + " · ".join(ch["strengths"]))
                if ch.get("flags"):
                    st.warning("Watchouts: " + " · ".join(ch["flags"]))
            else:
                st.caption(f"**{ch['badge']}** — {ch.get('note', '')}")

    # Upgrade #4: News & Earnings Risk Shield
    with st.expander(f"News & Earnings Risk Shield — {selected_ticker}", expanded=False):
        if use_demo_mode:
            st.caption("Turn off Offline Demo Data Mode to load live headlines & earnings dates.")
        else:
            ns = get_news_shield(selected_ticker)
            last_ed = ns.get("last_earnings_date") or "N/A"
            next_ed = ns.get("next_earnings_date") or "TBA"
            if ns.get("earnings_warning"):
                st.warning(f"Upcoming Earnings Alert — **Next:** {next_ed} | **Last:** {last_ed} (watch for gap risk!)")
            else:
                st.info(f"**Last Earnings:** {last_ed} &nbsp;·&nbsp; **Next Earnings:** {next_ed}")
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
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.70, 0.30])
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df.index,
                y=eq_df["Equity"],
                name=f"Maddy AI ({curr})",
                line=dict(color="#26a69a", width=2.4),
                fill="tozeroy",
                fillcolor="rgba(38, 166, 154, 0.08)",
            ),
            row=1,
            col=1,
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df.index,
                y=eq_df["NiftyNorm"],
                name=f"{bench_name} ({curr})",
                line=dict(color="#787b86", width=1.6, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df.index,
                y=eq_df["Drawdown%"],
                name="Drawdown %",
                fill="tozeroy",
                fillcolor="rgba(239, 83, 80, 0.22)",
                line=dict(color="#ef5350", width=1.5),
            ),
            row=2,
            col=1,
        )
        style_tradingview(fig_eq, height=520, right_axis=True, show_crosshair=True)
        fig_eq.update_layout(dragmode="pan")
        st.plotly_chart(fig_eq, width="stretch", config={"scrollZoom": True, "displaylogo": False})

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
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p95_path"], name="95th %ile", line=dict(color="#26a69a", width=2)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p50_path"], name="Median (50th)", line=dict(color="#2962ff", width=2.5)))
            fig_mc.add_trace(go.Scatter(x=x_days, y=mc["p5_path"], name="5th %ile Worst-Case", line=dict(color="#ef5350", width=2)))
            style_tradingview(fig_mc, height=400, right_axis=True, show_crosshair=True)
            fig_mc.update_layout(xaxis_title="Trading Days", yaxis_title=f"Equity ({curr})")
            st.plotly_chart(fig_mc, width="stretch", config={"displaylogo": False})

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
        style_tradingview(fig_hm, height=360, right_axis=False, show_crosshair=False)
        fig_hm.update_layout(title="Net Return % (SL vs TP ATR Multipliers)")
        st.plotly_chart(fig_hm, width="stretch", config={"displaylogo": False})

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
    if not paper_df.empty and not use_demo_mode:
        updated_paper_df, gtt_events = check_and_apply_gtt_exits(paper_df, stock_data)
        if gtt_events:
            restore_paper_trades_from_df(updated_paper_df)
            if is_cloud_sync_configured():
                try:
                    push_journal_to_github(
                        load_paper_trades(),
                        commit_msg=f"Auto-GTT close ({len(gtt_events)} trade(s))",
                    )
                except Exception:
                    pass
            paper_df = load_paper_trades()
            for ev in gtt_events:
                pct_s = f"{ev.get('net_pnl_pct', 0.0):+.2f}%"
                res_s = ev.get("result", "")
                if "TARGET" in ev["reason"]:
                    st.success(f"🎯 {ev['reason']} ({res_s}): **{ev['ticker']}** auto-closed @ {curr}{ev['exit_price']:,.2f} | Net P&L (after 0.15% charges): {curr}{ev['realized_pnl']:+,.2f} ({pct_s})")
                else:
                    st.warning(f"🛑 {ev['reason']} ({res_s}): **{ev['ticker']}** auto-closed @ {curr}{ev['exit_price']:,.2f} | Net P&L (after 0.15% charges): {curr}{ev['realized_pnl']:+,.2f} ({pct_s})")
    open_df = paper_df[paper_df["status"] == "OPEN"] if not paper_df.empty else pd.DataFrame()
    closed_df = paper_df[paper_df["status"] == "CLOSED"] if not paper_df.empty else pd.DataFrame()

    latest_prices: Dict[str, float] = {
        t: safe_float(df["Close"].iloc[-1]) for t, df in stock_data.items() if not df.empty
    }

    realized_pnl = safe_float(closed_df["realized_pnl"].sum()) if not closed_df.empty else 0.0
    unrealized_pnl = 0.0
    invested_cost = 0.0
    open_pos_dict: Dict[str, dict] = {}

    if not open_df.empty:
        for _, r in open_df.iterrows():
            ltp = latest_prices.get(r["ticker"], float(r["entry_price"]))
            invested_cost += float(r["entry_price"]) * float(r["qty"])
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

    free_cash = float(capital) + realized_pnl - invested_cost
    holdings_val = invested_cost + unrealized_pnl
    cash_pct = (free_cash / virtual_equity * 100.0) if virtual_equity > 0 else 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Equity (Stocks + Cash)", f"{curr}{virtual_equity:,.2f}", f"{curr}{(realized_pnl + unrealized_pnl):+,.2f}")
    p2.metric("Portfolio Heat (Max 5%)", f"{port_heat_pct:.2f}%")
    p3.metric("Open Positions", f"{len(open_df)}")
    p4.metric("25% Kill-Switch", "HALTED" if kill_switch_hit else "ACTIVE · SAFE")

    c_inv, c_cash = st.columns(2)
    c_inv.metric("Invested in Stocks (Current Value)", f"{curr}{holdings_val:,.2f}", f"Cost: {curr}{invested_cost:,.2f}")
    c_cash.metric("Free Cash Available", f"{curr}{free_cash:,.2f}", f"{cash_pct:.1f}% liquid")

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
        fig_heat.update_layout(
            height=260,
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            font=dict(color="#d1d4dc", family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=44, b=10),
        )
        st.plotly_chart(fig_heat, width="stretch", config={"displaylogo": False})

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
    st.caption("Auto-GTT Active: SL ya Target touch hote hi position apne aap close hoke GitHub Cloud mein save ho jayegi (Manual Close bhi available hai).")
    if open_df.empty:
        st.caption("Zero open positions.")
    else:
        for _, r in open_df.iterrows():
            ltp = latest_prices.get(r["ticker"], float(r["entry_price"]))
            u_pnl = (ltp - float(r["entry_price"])) * float(r["qty"])
            c_info, c_ch, c_act = st.columns([3.5, 1, 1])
            c_info.write(
                f"**{r['ticker']}** ({r['sector']}) · Qty `{r['qty']}` · Entry {curr}{r['entry_price']:,.2f} · LTP {curr}{ltp:,.2f} · SL {curr}{r['stop_loss']:,.2f} · Target {curr}{r['target']:,.2f} · **P&L: {curr}{u_pnl:+,.2f}**"
            )
            if c_ch.button("📉 Chart", key=f"p5_ch_{r['id']}", width="stretch"):
                jump_to_chart(str(r["ticker"]))
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
            if ok_s:
                st.success(msg_s)
            else:
                st.info(msg_s)
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


# ----------------------------------------------------------------------------
# TAB 6: 1-MINUTE DAILY MARKET BRIEFING (LINK-FREE, IN-APP READING)
# ----------------------------------------------------------------------------
with tab6:
    st.subheader(f"1-Minute Daily Briefing — {market_mode}")

    # Compute live movers & sector leaders for the plain-English briefing
    movers = []
    for t_mv, df_mv in stock_data.items():
        if len(df_mv) >= 2:
            c_now = safe_float(df_mv["Close"].iloc[-1])
            c_prev = safe_float(df_mv["Close"].iloc[-2])
            if c_prev > 0:
                chg_p = ((c_now - c_prev) / c_prev) * 100.0
                movers.append({"Ticker": t_mv, "Sector": get_sector(t_mv), "Price": round(c_now, 2), "Day%": round(chg_p, 2)})
    mv_df = pd.DataFrame(movers).sort_values(by="Day%", ascending=False) if movers else pd.DataFrame()

    sec_brief_df = compute_sector_strength(scan_df) if not scan_df.empty else pd.DataFrame()
    best_sec = f"{sec_brief_df.iloc[0]['Sector']} ({sec_brief_df.iloc[0]['Avg_RS%']:+.1f}%)" if not sec_brief_df.empty else "N/A"
    worst_sec = f"{sec_brief_df.iloc[-1]['Sector']} ({sec_brief_df.iloc[-1]['Avg_RS%']:+.1f}%)" if len(sec_brief_df) > 1 else "N/A"

    # 1. Executive 60-Second Verdict Box
    with st.container(border=True):
        st.markdown("#### Aaj Ka Market Summary (60 Seconds)")
        if regime["is_bullish"] and mood["score"] < 70:
            st.success(f"**Overall Mood:** Bullish & Healthy · {bench_name}: `{regime['close']:,.2f}` · MMI: `{mood['score']}/100 ({mood['zone']})`")
        elif regime["is_bullish"] and mood["score"] >= 70:
            st.warning(f"**Overall Mood:** Garam / Overheated · {bench_name}: `{regime['close']:,.2f}` · MMI: `{mood['score']}/100 ({mood['zone']})` — naye chase se bacho.")
        else:
            st.error(f"**Overall Mood:** Weak / Defensive · {bench_name}: `{regime['close']:,.2f}` · MMI: `{mood['score']}/100 ({mood['zone']})` — Cash / Gold Shield safe rakho.")

        st.markdown(
            f"- **Market Breadth:** `{br['above_ema50']} / {br['total']}` assets (`{br['pct_ema50']}%`) uptrend (`EMA50` ke upar) mein hain — *{br['label']}*.\n"
            f"- **Sabse Strong Sector:** **{best_sec}** &nbsp;|&nbsp; **Sabse Weak Sector:** **{worst_sec}**.\n"
            f"- **Action Tip:** {mood['tip']}"
        )

    # 2. Kisko Dekhna Chahiye (Watch/Strong) vs Kisko Avoid Karein (Weak)
    col_watch, col_avoid = st.columns(2)
    with col_watch:
        with st.container(border=True):
            st.markdown("#### 🟢 Kisko Dekhna Chahiye (Strongest)")
            if not scan_df.empty:
                strong_picks = scan_df[scan_df["Score"] >= 60].head(4)
                if strong_picks.empty:
                    strong_picks = scan_df.head(3)
                for _, r in strong_picks.iterrows():
                    st.markdown(
                        f"- **{r['Ticker']}** ({r['Sector']}) — Score **`{r['Score']}/100`**, RS `{r['RS_vs_Bench']:+.1f}%` · _{r['Setup']}_"
                    )
            else:
                st.caption("No scan data available.")

    with col_avoid:
        with st.container(border=True):
            st.markdown("#### 🔴 Kisko Avoid Karna Chahiye (Weakest)")
            if not scan_df.empty:
                weak_picks = scan_df.sort_values(by=["Score", "RS_vs_Bench"], ascending=[True, True]).head(4)
                for _, r in weak_picks.iterrows():
                    st.markdown(
                        f"- **{r['Ticker']}** ({r['Sector']}) — Score **`{r['Score']}/100`**, RS `{r['RS_vs_Bench']:+.1f}%` · _{r['Status']}_"
                    )
            else:
                st.caption("No scan data available.")

    if not mv_df.empty:
        mc1, mc2 = st.columns(2)
        with mc1:
            top_g = mv_df.head(3)
            g_str = " · ".join(f"**{r['Ticker']}** (`{r['Day%']:+.2f}%`)" for _, r in top_g.iterrows())
            st.info(f"**Aaj Ke Top Gainers:** {g_str}")
        with mc2:
            top_l = mv_df.tail(3).sort_values(by="Day%", ascending=True)
            l_str = " · ".join(f"**{r['Ticker']}** (`{r['Day%']:+.2f}%`)" for _, r in top_l.iterrows())
            st.warning(f"**Aaj Ke Top Losers:** {l_str}")

    st.divider()

    # 3. Important Market Updates (Link-Free Readable Cards!)
    st.subheader("Important Market Updates (No Links — Read Right Here)")
    if use_demo_mode:
        st.caption("Offline Demo Data Mode ON hai — live updates padhne ke liye sidebar mein Demo Mode OFF karo.")
    else:
        news_items = fetch_market_news_feed(market_mode)
        if news_items:
            caut_n = sum(1 for x in news_items if x["tag"] == "CAUTION")
            pos_n = sum(1 for x in news_items if x["tag"] == "POSITIVE")
            st.caption(f"Quick Pulse: 🟢 {pos_n} Positive · 🔴 {caut_n} Caution · ⚪ {len(news_items) - pos_n - caut_n} Neutral")
            for item in news_items:
                tag_lbl = "🔴 CAUTION" if item["tag"] == "CAUTION" else ("🟢 POSITIVE" if item["tag"] == "POSITIVE" else "⚪ UPDATE")
                with st.container(border=True):
                    st.markdown(f"**{tag_lbl} · {item['title']}**")
                    if item.get("summary"):
                        st.write(item["summary"])
                    st.caption(f"{item['publisher']} · {item['date']}")
        else:
            st.caption("Abhi koi fresh update load nahi hui — thodi der baad Refresh dabao.")

if st.session_state.pop("_jump_to_chart_tab", False):
    components.html(
        """
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        if (tabs && tabs.length > 1) {
            tabs[1].click();
        }
        </script>
        """,
        height=0,
    )
