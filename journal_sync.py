"""
journal_sync.py — Permanent Cloud Backup & Real-Life Auto-GTT (SL/Target) Engine for Maddy AI.
Uses GitHub REST API (zero extra packages!) when GITHUB_TOKEN + GITHUB_REPO are set.
Falls back cleanly to local SQLite + CSV Download/Restore when secrets are not set.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


JOURNAL_REPO_PATH = "paper_trades_journal.csv"
SIGNALS_REPO_PATH = "daily_signals_log.csv"
ROUND_TRIP_COST_PCT = 0.0015  # 0.15% brokerage + STT + slippage
IST = timezone(timedelta(hours=5, minutes=30))


def _get_secret(key: str) -> str:
    if _HAS_STREAMLIT:
        try:
            if key in st.secrets:
                return str(st.secrets[key]).strip()
        except Exception:
            pass
    return os.environ.get(key, "").strip()


def is_cloud_sync_configured() -> bool:
    return bool(_get_secret("GITHUB_TOKEN") and _get_secret("GITHUB_REPO"))


def push_journal_to_github(df: pd.DataFrame, commit_msg: str = "Sync Maddy AI paper trading journal") -> Tuple[bool, str]:
    """Save paper_trades_journal.csv to the user's GitHub repo so trades survive Streamlit reboots."""
    token = _get_secret("GITHUB_TOKEN")
    repo = _get_secret("GITHUB_REPO")
    if not token or not repo:
        return False, "Add GITHUB_TOKEN and GITHUB_REPO in Streamlit Secrets for auto cloud backup."

    api_url = f"https://api.github.com/repos/{repo}/contents/{JOURNAL_REPO_PATH}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "MaddyAI-Terminal",
    }

    # Check if file already exists to get its SHA
    sha = None
    try:
        req_get = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=10) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
            sha = meta.get("sha")
    except Exception:
        sha = None

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    payload = {
        "message": commit_msg,
        "content": base64.b64encode(csv_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha

    try:
        req_put = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req_put, timeout=12) as resp:
            if resp.status in (200, 201):
                return True, "Paper trade journal synced to GitHub cloud!"
            return False, f"GitHub sync returned status {resp.status}"
    except Exception as exc:
        return False, f"GitHub sync failed: {exc}"


def pull_journal_from_github() -> Tuple[Optional[pd.DataFrame], str]:
    """Fetch paper_trades_journal.csv from GitHub repo if configured."""
    token = _get_secret("GITHUB_TOKEN")
    repo = _get_secret("GITHUB_REPO")
    if not token or not repo:
        return None, "GitHub cloud sync not configured."

    api_url = f"https://api.github.com/repos/{repo}/contents/{JOURNAL_REPO_PATH}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "MaddyAI-Terminal",
    }
    try:
        req = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
            raw = base64.b64decode(meta["content"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            return df, "Restored journal from GitHub cloud!"
    except Exception as exc:
        return None, f"No cloud journal found yet: {exc}"


def check_and_apply_gtt_exits(
    journal_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Real-life Broker GTT (OCO) engine for OPEN paper trades:
    - Checks OHLC bars on/after the trade's entry_date.
    - Same-day bar (bar_day == entry_day): checks Close <= SL or Close >= Target (prevents false trigger on earlier intraday wick).
    - Next trading days (bar_day > entry_day): checks Low <= SL first (conservative risk rule), then High >= Target,
      with realistic gap-down/gap-up Open fill when price gaps past the GTT trigger.
    Returns (updated_journal_df, list_of_closed_event_dicts).
    """
    if journal_df is None or journal_df.empty:
        return journal_df, []

    updated = journal_df.copy()
    if "net_pnl_pct" not in updated.columns:
        updated["net_pnl_pct"] = 0.0
    if "result" not in updated.columns:
        updated["result"] = updated["status"].apply(
            lambda s: "OPEN" if str(s).upper() == "OPEN" else "CLOSED"
        )
    closed_events: List[dict] = []

    for idx, row in updated.iterrows():
        if str(row.get("status", "")).upper() != "OPEN":
            continue

        ticker = str(row.get("ticker", "")).strip()
        df_px = price_data.get(ticker)
        if df_px is None or df_px.empty:
            continue

        try:
            entry_ts = pd.Timestamp(str(row["entry_date"]))
            entry_day = entry_ts.normalize()
            entry_p = float(row["entry_price"])
            sl_p = float(row["stop_loss"])
            tp_p = float(row["target"])
            qty = float(row["qty"])
        except Exception:
            continue

        if sl_p <= 0 or tp_p <= 0 or qty <= 0:
            continue

        # Ensure index is tz-naive DatetimeIndex for clean date comparisons
        px = df_px.copy()
        try:
            px.index = pd.to_datetime(px.index)
            if getattr(px.index, "tz", None) is not None:
                px.index = px.index.tz_localize(None)
        except Exception:
            continue

        hit_reason = None
        exit_p = None
        exit_dt_str = None

        for bar_ts, bar in px.iterrows():
            bar_day = bar_ts.normalize()
            if bar_day < entry_day:
                continue

            o_val = float(bar["Open"]) if pd.notna(bar.get("Open")) else float(bar["Close"])
            h_val = float(bar["High"]) if pd.notna(bar.get("High")) else float(bar["Close"])
            l_val = float(bar["Low"]) if pd.notna(bar.get("Low")) else float(bar["Close"])
            c_val = float(bar["Close"])

            if bar_day == entry_day:
                # Only trigger on entry day if close itself breached SL/Target
                if c_val <= sl_p:
                    hit_reason = "AUTO SL HIT"
                    exit_p = round(sl_p, 2)
                elif c_val >= tp_p:
                    hit_reason = "AUTO TARGET HIT"
                    exit_p = round(tp_p, 2)
            else:
                # Subsequent trading days: full GTT High/Low check (SL checked first for safety)
                if l_val <= sl_p:
                    hit_reason = "AUTO SL HIT"
                    exit_p = round(min(o_val, sl_p) if o_val < sl_p else sl_p, 2)
                elif h_val >= tp_p:
                    hit_reason = "AUTO TARGET HIT"
                    exit_p = round(max(o_val, tp_p) if o_val > tp_p else tp_p, 2)

            if hit_reason is not None:
                exit_dt_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
                break

        if hit_reason is not None and exit_p is not None:
            gross_pnl = (exit_p - entry_p) * qty
            charges = (entry_p * qty) * ROUND_TRIP_COST_PCT
            pnl = round(gross_pnl - charges, 2)
            net_pnl_pct = round((((exit_p - entry_p) / entry_p) * 100.0) - (ROUND_TRIP_COST_PCT * 100.0), 2) if entry_p > 0 else 0.0
            res_lbl = "WIN" if pnl > 0 else "LOSS"
            orig_setup = str(row.get("setup", "") or "").strip()
            new_setup = f"{orig_setup} · {hit_reason}" if orig_setup else hit_reason

            updated.at[idx, "status"] = "CLOSED"
            updated.at[idx, "exit_date"] = exit_dt_str
            updated.at[idx, "exit_price"] = exit_p
            updated.at[idx, "realized_pnl"] = pnl
            updated.at[idx, "net_pnl_pct"] = net_pnl_pct
            updated.at[idx, "result"] = res_lbl
            updated.at[idx, "setup"] = new_setup

            closed_events.append(
                {
                    "id": row.get("id"),
                    "ticker": ticker,
                    "sector": str(row.get("sector", "")),
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "stop_loss": sl_p,
                    "target": tp_p,
                    "qty": qty,
                    "realized_pnl": pnl,
                    "net_pnl_pct": net_pnl_pct,
                    "result": res_lbl,
                    "reason": hit_reason,
                    "exit_date": exit_dt_str,
                }
            )

    return updated, closed_events


def append_signals_to_github(new_signals_df: pd.DataFrame) -> Tuple[bool, str]:
    """Append daily READY scanner signals to daily_signals_log.csv on GitHub (deduplicated by Date+Ticker)."""
    if new_signals_df is None or new_signals_df.empty:
        return False, "No signals to log."
    token = _get_secret("GITHUB_TOKEN")
    repo = _get_secret("GITHUB_REPO")
    if not token or not repo:
        return False, "GitHub cloud sync not configured."

    api_url = f"https://api.github.com/repos/{repo}/contents/{SIGNALS_REPO_PATH}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "MaddyAI-Terminal",
    }

    sha = None
    existing_df = pd.DataFrame()
    try:
        req_get = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=10) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
            sha = meta.get("sha")
            raw = base64.b64decode(meta["content"]).decode("utf-8")
            existing_df = pd.read_csv(io.StringIO(raw))
    except Exception:
        sha = None

    combined = pd.concat([existing_df, new_signals_df], ignore_index=True)
    if {"Date", "Ticker"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["Date", "Ticker"], keep="last")

    csv_bytes = combined.to_csv(index=False).encode("utf-8")
    payload = {
        "message": f"Log {len(new_signals_df)} Maddy AI daily signal(s)",
        "content": base64.b64encode(csv_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha

    try:
        req_put = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req_put, timeout=12) as resp:
            if resp.status in (200, 201):
                return True, f"Logged {len(new_signals_df)} signal(s) to {SIGNALS_REPO_PATH}!"
            return False, f"GitHub signals log returned status {resp.status}"
    except Exception as exc:
        return False, f"GitHub signals log failed: {exc}"
