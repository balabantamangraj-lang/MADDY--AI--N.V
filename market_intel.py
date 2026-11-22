"""
market_intel.py — Market Mood Meter (0-100), Market Breadth, and Company Health Badge for Maddy AI.
Zero extra dependencies (uses yfinance + pandas + numpy with safe fallbacks and caching).
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd

from data_feed import fetch_ohlcv, safe_float
from scanner import add_indicators

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def compute_market_breadth(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Count how many watchlist assets are trading above their EMA50."""
    total = 0
    above_ema50 = 0
    above_ema20 = 0

    for df in stock_data.values():
        ind = add_indicators(df)
        if ind.empty or len(ind) < 50:
            continue
        last = ind.iloc[-1]
        c = safe_float(last["Close"])
        e20 = safe_float(last["EMA20"])
        e50 = safe_float(last["EMA50"])
        if c <= 0:
            continue
        total += 1
        if c > e50:
            above_ema50 += 1
        if c > e20:
            above_ema20 += 1

    pct_50 = round((above_ema50 / total) * 100.0, 1) if total > 0 else 0.0
    pct_20 = round((above_ema20 / total) * 100.0, 1) if total > 0 else 0.0

    if pct_50 >= 65.0:
        label = "Broad Rally (Mazboot)"
    elif pct_50 >= 40.0:
        label = "Mixed Market (Selective)"
    else:
        label = "Weak Breadth (Kamzor)"

    return {
        "above_ema50": above_ema50,
        "above_ema20": above_ema20,
        "total": total,
        "pct_ema50": pct_50,
        "pct_ema20": pct_20,
        "label": label,
    }


def compute_market_mood(
    stock_data: Dict[str, pd.DataFrame],
    bench_df: pd.DataFrame,
    is_nse: bool = True,
    use_demo_mode: bool = False,
) -> Dict[str, Any]:
    """
    Compute 0-100 Market Mood Index (Fear & Greed):
      1. Breadth (% above EMA50) — 40% weight
      2. Benchmark Momentum (Close vs EMA50 + 20D ROC) — 35% weight
      3. Volatility / India VIX — 25% weight
    """
    breadth = compute_market_breadth(stock_data)
    breadth_score = float(breadth["pct_ema50"])  # 0-100

    # Benchmark momentum score (0-100)
    mom_score = 50.0
    b_ind = add_indicators(bench_df)
    if not b_ind.empty and len(b_ind) >= 50:
        b_last = b_ind.iloc[-1]
        b_close = safe_float(b_last["Close"])
        b_e50 = safe_float(b_last["EMA50"], default=b_close)
        b_roc20 = safe_float(b_last["ROC20"])
        dist_50 = ((b_close - b_e50) / b_e50 * 100.0) if b_e50 > 0 else 0.0
        mom_score = float(np.clip(50.0 + dist_50 * 4.0 + b_roc20 * 2.5, 5.0, 95.0))

    # Volatility / VIX score (low VIX -> Greed/higher score, high VIX -> Fear/lower score)
    vol_score = 55.0
    vix_val = None
    if is_nse and not use_demo_mode:
        try:
            vix_df = fetch_ohlcv("^INDIAVIX", period="3mo")
            if not vix_df.empty:
                vix_val = round(safe_float(vix_df["Close"].iloc[-1]), 2)
                # India VIX ~12 -> calm/greed (~80), VIX ~25+ -> high fear (~15)
                vol_score = float(np.clip(100.0 - ((vix_val - 10.0) * 5.0), 10.0, 95.0))
        except Exception:
            vix_val = None
    elif not b_ind.empty:
        atr_pct = safe_float(b_ind["ATR_Pct"].iloc[-1], default=1.2)
        vol_score = float(np.clip(85.0 - atr_pct * 15.0, 15.0, 90.0))

    mood_score = int(round(np.clip(0.40 * breadth_score + 0.35 * mom_score + 0.25 * vol_score, 0, 100)))

    if mood_score < 25:
        zone = "Extreme Fear"
        color = "#ef5350"
        tip = "Bazaar mein zyada darr hai — jaldbazi mat karo, sirf A+ quality setups par nazar rakho."
    elif mood_score < 45:
        zone = "Fear"
        color = "#ff9800"
        tip = "Market cautious hai — defensive sizing aur Bear Shield par dhyan rakho."
    elif mood_score <= 55:
        zone = "Neutral"
        color = "#94a3b8"
        tip = "Balanced mood — sirf strong RS leaders aur clean FVG setups follow karo."
    elif mood_score <= 75:
        zone = "Greed"
        color = "#26a69a"
        tip = "Healthy bullish trend — Chandelier trailing stops ke saath winners ride karo."
    else:
        zone = "Extreme Greed"
        color = "#00e676"
        tip = "Bazaar garam (overheated) hai — naya breakout chase mat karo, trailing stop tight rakho."

    return {
        "score": mood_score,
        "zone": zone,
        "color": color,
        "tip": tip,
        "vix": vix_val,
        "breadth": breadth,
    }


def _fetch_company_health_raw(ticker: str) -> Dict[str, Any]:
    """Fetch fundamental health metrics (P/E, ROE, Debt/Equity, Profit Margin) via yfinance."""
    if yf is None or "-USD" in ticker or "=F" in ticker or "BEES" in ticker or ticker in ("SPY", "QQQ", "GLD", "SLV"):
        return {
            "applicable": False,
            "badge": "ETF / Index / Crypto",
            "score": None,
            "note": "Fundamental company balance-sheet ratios sirf individual stocks par apply hote hain.",
        }

    try:
        info = yf.Ticker(ticker).info or {}
        pe = safe_float(info.get("trailingPE"), default=0.0)
        roe = safe_float(info.get("returnOnEquity"), default=0.0) * 100.0
        dte = safe_float(info.get("debtToEquity"), default=-1.0)
        margin = safe_float(info.get("profitMargins"), default=0.0) * 100.0
        rev_growth = safe_float(info.get("revenueGrowth"), default=0.0) * 100.0

        score = 50
        flags = []
        strengths = []

        if roe >= 15.0:
            score += 18
            strengths.append(f"Strong ROE ({roe:.1f}%)")
        elif 0 < roe < 8.0:
            score -= 12
            flags.append(f"Low ROE ({roe:.1f}%)")
        elif roe < 0:
            score -= 22
            flags.append("Negative ROE")

        if 0 <= dte <= 60.0:
            score += 14
            strengths.append(f"Low Debt (D/E {dte:.0f}%)")
        elif dte > 150.0:
            score -= 18
            flags.append(f"High Debt (D/E {dte:.0f}%)")

        if margin >= 10.0:
            score += 10
            strengths.append(f"Healthy Margin ({margin:.1f}%)")
        elif margin < 0:
            score -= 20
            flags.append("Loss-Making")

        if rev_growth >= 10.0:
            score += 8
            strengths.append(f"Revenue +{rev_growth:.1f}%")
        elif rev_growth < 0:
            score -= 8
            flags.append(f"Revenue {rev_growth:.1f}%")

        if pe > 80.0:
            score -= 8
            flags.append(f"Expensive P/E ({pe:.1f})")

        score = int(np.clip(score, 0, 100))
        if score >= 70:
            badge = "🟢 Strong Company"
        elif score >= 45:
            badge = "🟡 Moderate Health"
        else:
            badge = "🔴 Risky / Caution"

        return {
            "applicable": True,
            "badge": badge,
            "score": score,
            "pe": round(pe, 1) if pe > 0 else None,
            "roe": round(roe, 1) if roe != 0 else None,
            "dte": round(dte, 1) if dte >= 0 else None,
            "margin": round(margin, 1) if margin != 0 else None,
            "rev_growth": round(rev_growth, 1) if rev_growth != 0 else None,
            "strengths": strengths,
            "flags": flags,
        }
    except Exception:
        return {
            "applicable": False,
            "badge": "Data Unavailable",
            "score": None,
            "note": "Fundamental data abhi fetch nahi ho paya.",
        }


if _HAS_STREAMLIT:
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_company_health(ticker: str) -> Dict[str, Any]:
        return _fetch_company_health_raw(ticker)
else:
    def get_company_health(ticker: str) -> Dict[str, Any]:
        return _fetch_company_health_raw(ticker)
