"""
run_daily_alerts.py — Automated Background GTT Monitor & Daily NSE Scanner for Maddy AI.
Runs via GitHub Actions (100% Free):
- Market-hours Auto-GTT checks (MADDY_RUN_MODE=gtt_only): Checks OPEN cloud paper trades against latest
  High/Low, auto-closes if SL or Target is hit, updates paper_trades_journal.csv on GitHub, and sends
  instant Telegram alerts (silent if no trade triggered).
- Daily 3:10 PM IST Full Scan (MADDY_RUN_MODE=daily_full): Runs GTT check + full NSE READY signal scan.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from alerts import format_signal_message, send_telegram_alert
from data_feed import BENCHMARK_TICKER, DEFAULT_WATCHLIST, fetch_ohlcv, load_watchlist_data
import pandas as pd
from journal_sync import (
    append_signals_to_github,
    check_and_apply_gtt_exits,
    is_cloud_sync_configured,
    pull_journal_from_github,
    push_journal_to_github,
)
from risk import RiskConfig, calculate_position_size
from scanner import check_nifty_regime, scan_watchlist


def run_background_gtt_check(ist_now: str, preloaded_data: dict | None = None) -> None:
    """Pull cloud paper journal, check SL/Target GTT triggers, auto-close, sync, and alert on Telegram."""
    if not is_cloud_sync_configured():
        print("GTT check skipped: GITHUB_TOKEN or GITHUB_REPO not set.")
        return

    journal_df, msg = pull_journal_from_github()
    if journal_df is None or journal_df.empty:
        print("GTT check: no cloud journal found.", msg)
        return

    open_df = journal_df[journal_df["status"].astype(str).str.upper() == "OPEN"]
    if open_df.empty:
        print("GTT check: zero OPEN paper positions.")
        return

    price_data = dict(preloaded_data) if preloaded_data else {}
    for ticker in open_df["ticker"].dropna().unique():
        t_str = str(ticker).strip()
        if t_str and t_str not in price_data:
            df_t = fetch_ohlcv(t_str, period="1mo")
            if not df_t.empty:
                price_data[t_str] = df_t

    updated_df, closed_events = check_and_apply_gtt_exits(journal_df, price_data)
    if not closed_events:
        print(f"GTT check ({ist_now}): {len(open_df)} open position(s) checked — no SL/Target hit.")
        return

    ok_push, push_msg = push_journal_to_github(
        updated_df,
        commit_msg=f"Auto-GTT close ({len(closed_events)} trade(s)) at {ist_now}",
    )
    print("GTT Cloud Sync:", ok_push, push_msg)

    for ev in closed_events:
        curr = "₹" if ev["ticker"].endswith(".NS") else "$"
        icon = "🎯" if "TARGET" in ev["reason"] else "🛑"
        pnl_sign = "+" if ev["realized_pnl"] >= 0 else ""
        net_pct = ev.get("net_pnl_pct", 0.0)
        res_lbl = ev.get("result", "CLOSED")
        alert_msg = (
            f"{icon} MADDY AI — {ev['reason']} [{res_lbl}] ({ist_now})\n"
            f"Stock: {ev['ticker']} ({ev['sector']})\n"
            f"Entry: {curr}{ev['entry_price']:,.2f} -> Exit: {curr}{ev['exit_price']:,.2f}\n"
            f"Qty: {ev['qty']} | Net P&L (after 0.15%): {curr}{pnl_sign}{ev['realized_pnl']:,.2f} ({net_pct:+.2f}%)\n"
            f"Status: Auto-closed & saved to GitHub Cloud Journal."
        )
        ok_tg, info_tg = send_telegram_alert(alert_msg)
        print(f"Sent GTT Telegram alert for {ev['ticker']}:", ok_tg, info_tg)


def main() -> None:
    run_mode = os.environ.get("MADDY_RUN_MODE", "daily_full").strip().lower()
    capital = float(os.environ.get("MADDY_CAPITAL", "100000"))
    sl_mult = float(os.environ.get("MADDY_SL_MULT", "2.0"))
    tp_mult = float(os.environ.get("MADDY_TP_MULT", "3.5"))

    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p IST")

    # Mode 1: Fast silent background GTT monitor (only alerts if SL/Target hits)
    if run_mode == "gtt_only":
        run_background_gtt_check(ist_now)
        return

    # Mode 2: Full Daily NSE Scan + GTT monitor (3:10 PM IST or manual run)
    nifty_df = fetch_ohlcv(BENCHMARK_TICKER, period="1y")
    stock_data = load_watchlist_data(DEFAULT_WATCHLIST, period="1y")

    # Run GTT Auto-Close check first using freshly downloaded prices
    run_background_gtt_check(ist_now, preloaded_data=stock_data)

    if nifty_df.empty or not stock_data:
        send_telegram_alert(f"MADDY AI ({ist_now})\nCould not fetch NSE data today. Please check dashboard.")
        return

    regime = check_nifty_regime(nifty_df)
    scan_df = scan_watchlist(
        stock_data=stock_data,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        regime_bullish=regime["is_bullish"],
        nifty_roc20=regime.get("roc20", 0.0),
    )

    header = (
        f"MADDY AI — DAILY NSE SCAN ({ist_now})\n"
        f"Nifty 50 Regime: {regime['label']} (Close: {regime['close']:,.2f})"
    )

    if not regime["is_bullish"]:
        ok, info = send_telegram_alert(f"{header}\nMacro Bearish — Cash Safe mode active. No new trades today.")
        print("Sent bearish summary:", ok, info)
        return

    ready_df = scan_df[scan_df["Status"] == "READY"] if not scan_df.empty else scan_df
    if ready_df.empty:
        top_watch = scan_df.head(3)[["Ticker", "Score", "Setup"]].to_dict("records") if not scan_df.empty else []
        watch_lines = "\n".join(f"- {r['Ticker']}: {r['Score']}/100 ({r['Setup']})" for r in top_watch)
        msg = f"{header}\nNo READY signals (Score >= 75) today.\nTop Watchlist:\n{watch_lines}"
        ok, info = send_telegram_alert(msg)
        print("Sent no-signal summary:", ok, info)
        return

    risk_cfg = RiskConfig()
    today_str = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")
    sig_rows = []
    send_telegram_alert(f"{header}\n{len(ready_df)} READY Signal(s) detected:")
    for _, row in ready_df.iterrows():
        qty = calculate_position_size(capital, float(row["Entry"]), float(row["StopLoss"]), risk_cfg)
        alert_text = format_signal_message(row.to_dict(), qty)
        ok, info = send_telegram_alert(alert_text)
        print(f"Sent alert for {row['Ticker']}:", ok, info)
        sig_rows.append(
            {
                "Date": today_str,
                "Ticker": row["Ticker"],
                "Sector": row["Sector"],
                "Setup": row["Setup"],
                "Score": row["Score"],
                "Signal_Price": round(float(row["Entry"]), 2),
                "SL": round(float(row["StopLoss"]), 2),
                "Target": round(float(row["Target"]), 2),
                "Status": "OPEN",
            }
        )
    if sig_rows and is_cloud_sync_configured():
        ok_sig, msg_sig = append_signals_to_github(pd.DataFrame(sig_rows))
        print("Daily Signals Log:", ok_sig, msg_sig)


if __name__ == "__main__":
    main()
