"""
journal_sync.py — Optional Permanent Cloud Backup for Maddy AI Paper Trading Journal.
Uses GitHub REST API (zero extra packages!) when GITHUB_TOKEN + GITHUB_REPO are set in Streamlit Secrets.
Falls back cleanly to local SQLite + CSV Download/Restore when secrets are not set.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Optional, Tuple
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


JOURNAL_REPO_PATH = "paper_trades_journal.csv"


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


def push_journal_to_github(df: pd.DataFrame) -> Tuple[bool, str]:
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
        "message": "Sync Maddy AI paper trading journal",
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
