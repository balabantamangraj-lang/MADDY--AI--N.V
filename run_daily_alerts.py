"""
run_daily_alerts.py — Automated Daily NSE Scanner & Telegram Alert Runner for Maddy AI.
Designed to run via GitHub Actions (Mon-Fri at 3:10 PM IST) or manually anytime.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment variables.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from alerts import format_signal_message, send_telegram_alert
from data_feed import BENCHMARK_TICKER, DEFAULT_WATCHLIST, fetch_ohlcv, load_watchlist_data
from risk import RiskConfig, calculate_position_size
from scanner import check_nifty_regime, scan_watchlist


def main() -> None:
    capital = float(os.environ.get("MADDY_CAPITAL", "100000"))
    sl_mult = float(os.environ.get("MADDY_SL_MULT", "2.0"))
    tp_mult = float(os.environ.get("MADDY_TP_MULT", "3.5"))

    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p IST")

    nifty_df = fetch_ohlcv(BENCHMARK_TICKER, period="1y")
    stock_data = load_watchlist_data(DEFAULT_WATCHLIST, period="1y")

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
    send_telegram_alert(f"{header}\n{len(ready_df)} READY Signal(s) detected:")
    for _, row in ready_df.iterrows():
        qty = calculate_position_size(capital, float(row["Entry"]), float(row["StopLoss"]), risk_cfg)
        alert_text = format_signal_message(row.to_dict(), qty)
        ok, info = send_telegram_alert(alert_text)
        print(f"Sent alert for {row['Ticker']}:", ok, info)


if __name__ == "__main__":
    main()
