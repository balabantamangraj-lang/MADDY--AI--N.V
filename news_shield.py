"""
news_shield.py — Latest News Headlines, Sentiment Tag & Upcoming Earnings Risk Alert for Maddy AI.
Uses yfinance safely with try/except and Streamlit caching so it never slows down or crashes the terminal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


CAUTION_KEYWORDS = (
    "lawsuit", "probe", "penalty", "fraud", "ban", "hack", "downgrade",
    "default", "crash", "sebi", "notice", "fine", "investigation", "warning",
    "loss", "slump", "plunge", "layoff", "recall",
)
POSITIVE_KEYWORDS = (
    "surge", "rally", "upgrade", "beat", "record", "profit", "growth",
    "buyback", "dividend", "order win", "partnership", "breakout", "approval",
)


def _classify_headline(title: str) -> str:
    low = title.lower()
    if any(k in low for k in CAUTION_KEYWORDS):
        return "CAUTION"
    if any(k in low for k in POSITIVE_KEYWORDS):
        return "POSITIVE"
    return "NEUTRAL"


def _fetch_news_and_earnings_raw(ticker: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ticker": ticker,
        "headlines": [],
        "caution_count": 0,
        "last_earnings_date": None,
        "next_earnings_date": None,
        "earnings_warning": False,
    }
    if yf is None:
        return result

    try:
        tk = yf.Ticker(ticker)
        # 1. Latest News Headlines (handles both legacy and new yfinance news schemas)
        raw_news = getattr(tk, "news", None) or []
        headlines: List[Dict[str, str]] = []
        for item in raw_news[:5]:
            if not isinstance(item, dict):
                continue
            content = item.get("content") if isinstance(item.get("content"), dict) else item
            title = str(content.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            pub_obj = content.get("provider") if isinstance(content.get("provider"), dict) else {}
            publisher = str(pub_obj.get("displayName") or item.get("publisher") or "Market News")
            url_obj = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
            link = str(url_obj.get("url") or item.get("link") or "")
            tag = _classify_headline(title)
            if tag == "CAUTION":
                result["caution_count"] += 1
            headlines.append({"title": title, "publisher": publisher, "link": link, "tag": tag})
        result["headlines"] = headlines[:4]

        # 2. Last & Next Earnings Date check (equities only)
        if "-USD" not in ticker and "=F" not in ticker:
            now_utc = datetime.now(timezone.utc).date()
            parsed_dates = []
            cal = getattr(tk, "calendar", None)
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                ed_list = list(ed) if isinstance(ed, (list, tuple)) else [ed]
                for item in ed_list:
                    if item is not None:
                        try:
                            parsed_dates.append(datetime.strptime(str(item)[:10], "%Y-%m-%d").date())
                        except Exception:
                            pass
            try:
                ed_df = getattr(tk, "earnings_dates", None)
                if ed_df is not None and not ed_df.empty:
                    for idx_dt in ed_df.index[:8]:
                        try:
                            parsed_dates.append(idx_dt.date())
                        except Exception:
                            pass
            except Exception:
                pass

            past = sorted({d for d in parsed_dates if d < now_utc})
            future = sorted({d for d in parsed_dates if d >= now_utc})
            if past:
                result["last_earnings_date"] = past[-1].strftime("%Y-%m-%d")
            if future:
                nxt = future[0]
                result["next_earnings_date"] = nxt.strftime("%Y-%m-%d")
                if 0 <= (nxt - now_utc).days <= 10:
                    result["earnings_warning"] = True
    except Exception:
        pass

    return result


if _HAS_STREAMLIT:
    @st.cache_data(ttl=1800, show_spinner=False)
    def get_news_shield(ticker: str) -> Dict[str, Any]:
        return _fetch_news_and_earnings_raw(ticker)
else:
    def get_news_shield(ticker: str) -> Dict[str, Any]:
        return _fetch_news_and_earnings_raw(ticker)
