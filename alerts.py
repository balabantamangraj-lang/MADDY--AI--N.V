"""
alerts.py — Automatic Telegram Bot Alert Module for Maddy AI.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from Streamlit Secrets or env vars.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple
import urllib.parse
import urllib.request
import json

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def get_telegram_credentials() -> Tuple[str, str]:
    """Fetch Telegram Bot Token and Chat ID from st.secrets or environment."""
    token = ""
    chat_id = ""
    if _HAS_STREAMLIT:
        try:
            token = str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
            chat_id = str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
        except Exception:
            pass
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def format_signal_message(signal_row: Dict[str, Any], qty: int) -> str:
    """Format a clean, mobile-readable Telegram alert for a READY trade signal."""
    ticker = signal_row.get("Ticker", "N/A")
    score = signal_row.get("Score", 0)
    setup = signal_row.get("Setup", "Trend + SMC")
    entry = float(signal_row.get("Entry", 0.0))
    sl = float(signal_row.get("StopLoss", 0.0))
    target = float(signal_row.get("Target", 0.0))
    sector = signal_row.get("Sector", "Other")
    risk_per_share = max(0.01, entry - sl)
    rr = (target - entry) / risk_per_share

    return (
        f"MADDY AI — READY SIGNAL\n"
        f"Stock: {ticker} ({sector})\n"
        f"Score: {score}/100 | Setup: {setup}\n"
        f"Entry: Rs.{entry:,.2f}\n"
        f"ATR Stop-Loss: Rs.{sl:,.2f}\n"
        f"Target: Rs.{target:,.2f} (R:R 1:{rr:.1f})\n"
        f"1% Risk Qty: {int(qty)} shares"
    )


def send_telegram_alert(message: str) -> Tuple[bool, str]:
    """Send a Telegram message via Bot API. Returns (success, status_message)."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return (
            False,
            "Telegram keys not configured yet. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Streamlit Secrets.",
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if body.get("ok"):
                return True, "Telegram alert sent successfully!"
            return False, f"Telegram API error: {body.get('description', 'Unknown error')}"
    except Exception as exc:
        return False, f"Could not send Telegram alert: {exc}"
