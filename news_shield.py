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


import html as _html
import re as _re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def _strip_html(raw_text: str) -> str:
    txt = _html.unescape(raw_text or "")
    txt = _re.sub(r"<[^>]+>", " ", txt)
    txt = _re.sub(r"\s+", " ", txt).strip()
    return txt


def _fetch_rss_headlines(query: str, max_items: int = 8) -> List[Dict[str, str]]:
    """Fetch clean, link-free readable market headlines + short summaries from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    items: List[Dict[str, str]] = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for el in root.findall(".//item")[:max_items]:
            raw_title = (el.findtext("title") or "").strip()
            source_el = el.find("source")
            publisher = (source_el.text if source_el is not None and source_el.text else "Market Desk").strip()
            # Remove trailing " - Publisher" from Google News title for cleaner reading
            clean_title = raw_title
            if publisher and clean_title.endswith(f" - {publisher}"):
                clean_title = clean_title[: -len(f" - {publisher}")].strip()
            desc = _strip_html(el.findtext("description") or "")
            # Avoid repeating identical title as summary
            if desc.lower().startswith(clean_title.lower()[:35]):
                summary = ""
            else:
                summary = desc[:180] + ("..." if len(desc) > 180 else "")
            pub_date = (el.findtext("pubDate") or "").strip()[:16]
            if clean_title:
                items.append(
                    {
                        "title": clean_title,
                        "summary": summary,
                        "publisher": publisher,
                        "date": pub_date,
                        "tag": _classify_headline(clean_title + " " + summary),
                    }
                )
    except Exception:
        pass
    return items


def _fetch_market_news_raw(market_mode: str) -> List[Dict[str, str]]:
    if "Crypto" in market_mode:
        query = "Bitcoin OR Ethereum OR Crypto market news today"
    elif "Commodities" in market_mode:
        query = "Gold price OR Crude Oil OR S&P 500 market news today"
    else:
        query = "Nifty 50 OR Sensex OR Indian stock market NSE news today"
    return _fetch_rss_headlines(query, max_items=8)


if _HAS_STREAMLIT:
    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_market_news_feed(market_mode: str) -> List[Dict[str, str]]:
        return _fetch_market_news_raw(market_mode)
else:
    def fetch_market_news_feed(market_mode: str) -> List[Dict[str, str]]:
        return _fetch_market_news_raw(market_mode)
