"""
scanner.py — Dual-Engine White-Box Strategy & SMC Scanner for Maddy AI (Alpha Growth Upgrade).
Implements:
  - Technical Indicators: EMA20, EMA50, ATR14, ATR%, RSI14, Volume MA20
  - Relative Strength (20D RS vs Nifty 50) — focuses on market leaders
  - Smart Money Concepts (SMC):
      * 3-Candle Bullish & Bearish Fair Value Gap (FVG) detector + 50% midpoint mitigation
      * Common vs. Breakaway Market Gap detector (Gap Fill + Gap & Go)
  - Macro Filter: Nifty 50 Regime Banner
  - Confluence Scoring System (0-100) with Conviction Grade (A+ / A / Watch)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from data_feed import get_sector, safe_float


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA20, EMA50, ATR14, ATR%, RSI14, and 20-day Volume MA cleanly."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    prev_close = close.shift(1)

    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()
    out["High5"] = high.rolling(window=5, min_periods=3).max().shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = true_range.rolling(window=14, min_periods=14).mean()
    out["ATR_Pct"] = np.where(close > 0, (out["ATR14"] / close) * 100.0, 0.0)
    out["Vol_MA20"] = out["Volume"].astype(float).rolling(window=20, min_periods=10).mean()

    # RSI14 (Wilder smoothing via ewm)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    out["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    ag2 = gain.ewm(alpha=1 / 2, min_periods=2, adjust=False).mean()
    al2 = loss.ewm(alpha=1 / 2, min_periods=2, adjust=False).mean()
    rs2 = np.where(al2 > 0, ag2 / al2, 100.0)
    out["RSI2"] = 100.0 - (100.0 / (1.0 + rs2))
    out["ROC20"] = close.pct_change(20) * 100.0
    out["EMA20_Slope"] = out["EMA20"].diff(5)

    # ADX14 (Chop / Trend-Strength Filter — avoids flat sideways markets)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe = out["ATR14"].replace(0, np.nan)
    plus_di = 100.0 * (pd.Series(plus_dm, index=out.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_safe)
    minus_di = 100.0 * (pd.Series(minus_dm, index=out.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_safe)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    out["ADX14"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(0.0)
    return out


def detect_fvg_zones(
    df: pd.DataFrame,
    lookback: int = 80,
    min_atr_displacement: float = 0.30,
) -> List[Dict[str, Any]]:
    """3-Candle Fair Value Gap (FVG) detector requiring >= 0.3 ATR displacement + 50% midpoint."""
    enriched = add_indicators(df) if "ATR14" not in df.columns else df
    if enriched is None or len(enriched) < 5:
        return []

    sub = enriched.tail(lookback).copy()
    highs = sub["High"].to_numpy(dtype=float)
    lows = sub["Low"].to_numpy(dtype=float)
    closes = sub["Close"].to_numpy(dtype=float)
    atrs = sub["ATR14"].bfill().fillna(0.0).to_numpy(dtype=float)
    dates = sub.index

    zones: List[Dict[str, Any]] = []
    n = len(sub)

    for i in range(2, n):
        atr_ref = atrs[i - 1] if atrs[i - 1] > 0 else (closes[i - 1] * 0.01)
        min_disp = min_atr_displacement * atr_ref
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            if gap_size >= min_disp:
                bottom = float(highs[i - 2])
                top = float(lows[i])
                mid = (top + bottom) / 2.0
                mitigated_50 = False
                mitigated_full = False
                mitigation_date = None
                for j in range(i + 1, n):
                    if lows[j] <= mid:
                        mitigated_50 = True
                        if mitigation_date is None:
                            mitigation_date = dates[j]
                    if lows[j] <= bottom:
                        mitigated_full = True
                        break
                zones.append(
                    {
                        "type": "Bullish FVG",
                        "start_date": dates[i - 2],
                        "end_date": dates[i],
                        "top": round(top, 2),
                        "bottom": round(bottom, 2),
                        "midpoint": round(mid, 2),
                        "mitigated_50": mitigated_50,
                        "mitigated_full": mitigated_full,
                        "mitigation_date": mitigation_date,
                        "is_fresh": not mitigated_full,
                    }
                )
        elif highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            if gap_size >= min_disp:
                bottom = float(highs[i])
                top = float(lows[i - 2])
                mid = (top + bottom) / 2.0
                mitigated_50 = False
                mitigated_full = False
                mitigation_date = None
                for j in range(i + 1, n):
                    if highs[j] >= mid:
                        mitigated_50 = True
                        if mitigation_date is None:
                            mitigation_date = dates[j]
                    if highs[j] >= top:
                        mitigated_full = True
                        break
                zones.append(
                    {
                        "type": "Bearish FVG",
                        "start_date": dates[i - 2],
                        "end_date": dates[i],
                        "top": round(top, 2),
                        "bottom": round(bottom, 2),
                        "midpoint": round(mid, 2),
                        "mitigated_50": mitigated_50,
                        "mitigated_full": mitigated_full,
                        "mitigation_date": mitigation_date,
                        "is_fresh": not mitigated_full,
                    }
                )

    return zones


def detect_market_gaps(
    df: pd.DataFrame,
    lookback: int = 80,
    min_gap_pct: float = 0.010,
) -> List[Dict[str, Any]]:
    """Detect Breakaway (Gap & Go) vs Common (Gap Fill) Market Gaps."""
    enriched = add_indicators(df) if "Vol_MA20" not in df.columns else df
    if enriched is None or len(enriched) < 5:
        return []

    sub = enriched.tail(lookback).copy()
    opens = sub["Open"].to_numpy(dtype=float)
    closes = sub["Close"].to_numpy(dtype=float)
    highs = sub["High"].to_numpy(dtype=float)
    lows = sub["Low"].to_numpy(dtype=float)
    vols = sub["Volume"].to_numpy(dtype=float)
    vol_ma = sub["Vol_MA20"].fillna(sub["Volume"]).to_numpy(dtype=float)
    dates = sub.index

    gaps: List[Dict[str, Any]] = []
    n = len(sub)

    for i in range(1, n):
        prev_c = closes[i - 1]
        if prev_c <= 0:
            continue
        gap_pct = (opens[i] - prev_c) / prev_c
        if abs(gap_pct) < min_gap_pct:
            continue

        direction = "Gap Up" if gap_pct > 0 else "Gap Down"
        vol_ratio = vols[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        followed_through = (closes[i] >= opens[i]) if gap_pct > 0 else (closes[i] <= opens[i])
        gap_class = "Breakaway Gap (Gap & Go)" if (vol_ratio >= 1.3 and followed_through) else "Common Gap (Gap Fill)"

        top = max(prev_c, opens[i])
        bottom = min(prev_c, opens[i])
        filled = False
        for j in range(i, n):
            if gap_pct > 0 and lows[j] <= prev_c:
                filled = True
                break
            if gap_pct < 0 and highs[j] >= prev_c:
                filled = True
                break

        gaps.append(
            {
                "date": dates[i],
                "prev_date": dates[i - 1],
                "direction": direction,
                "gap_class": gap_class,
                "gap_pct": round(gap_pct * 100.0, 2),
                "top": round(float(top), 2),
                "bottom": round(float(bottom), 2),
                "vol_ratio": round(float(vol_ratio), 2),
                "filled": filled,
            }
        )

    return gaps


def check_nifty_regime(nifty_df: pd.DataFrame) -> Dict[str, Any]:
    """Macro Filter: Nifty 50 Bullish (Trade On) vs Bearish (Cash Safe)."""
    enriched = add_indicators(nifty_df)
    if enriched.empty or len(enriched) < 50:
        return {
            "is_bullish": False,
            "label": "DATA UNAVAILABLE",
            "close": 0.0,
            "ema20": 0.0,
            "ema50": 0.0,
            "roc20": 0.0,
        }

    last = enriched.iloc[-1]
    close = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    roc20 = safe_float(last["ROC20"])
    is_bullish = bool(close > ema20 and ema20 >= ema50)

    return {
        "is_bullish": is_bullish,
        "label": "BULLISH · TRADE ON" if is_bullish else "BEARISH · CASH SAFE",
        "close": round(close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "roc20": round(roc20, 2),
    }


def score_stock_snapshot(
    ticker: str,
    df: pd.DataFrame,
    sl_mult: float = 2.0,
    tp_mult: float = 3.5,
    regime_bullish: bool = True,
    nifty_roc20: float = 0.0,
    max_atr_pct: float = 4.5,
) -> Dict[str, Any]:
    """
    Dual-Engine White-Box Scoring (0-100) with RS Leader + RSI Quality Filter:
      - Layer 1 (Trend + RS Leader + RSI, up to 55 pts)
      - Layer 2 (SMC Booster: FVG 50% + Breakaway Gap, up to 45 pts)
    """
    enriched = add_indicators(df)
    if enriched.empty or len(enriched) < 50:
        return {
            "Ticker": ticker,
            "Sector": get_sector(ticker),
            "Status": "NO DATA",
            "Grade": "-",
            "Score": 0,
            "RS_vs_Bench": 0.0,
            "ADX14": 0.0,
            "RSI14": 0.0,
            "Setup": "Insufficient History",
            "SMC_Signal": "None",
            "Entry": 0.0,
            "StopLoss": 0.0,
            "Target": 0.0,
            "ATR14": 0.0,
        }

    last = enriched.iloc[-1]
    close = safe_float(last["Close"])
    low = safe_float(last["Low"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    atr14 = safe_float(last["ATR14"])
    atr_pct = safe_float(last["ATR_Pct"])
    rsi14 = safe_float(last["RSI14"], default=50.0)
    roc20 = safe_float(last["ROC20"])
    rs_vs_nifty = roc20 - nifty_roc20
    vol = safe_float(last["Volume"])
    vol_ma20 = safe_float(last["Vol_MA20"], default=1.0)

    score = 0
    setup_tags: List[str] = []
    smc_tags: List[str] = []

    # --- Layer 1: Trend + Relative Strength + RSI Quality (max 55 pts) ---
    if close > ema20:
        score += 15
        setup_tags.append("EMA20 Trend")
    if ema20 > ema50:
        score += 12
    if rs_vs_nifty > 0:
        score += 12
        setup_tags.append("RS Leader")
    if 48.0 <= rsi14 <= 70.0:
        score += 10
    elif rsi14 > 75.0:
        score = max(0, score - 8)
        smc_tags.append("Overbought RSI")
    if 0.8 <= atr_pct <= max_atr_pct and vol_ma20 > 0 and vol >= vol_ma20:
        score += 6

    # --- Layer 2: SMC Booster (max 45 pts) ---
    fvg_zones = detect_fvg_zones(enriched, lookback=45)
    bull_fvgs = [z for z in fvg_zones if z["type"] == "Bullish FVG" and z["is_fresh"]]
    bear_fvgs = [z for z in fvg_zones if z["type"] == "Bearish FVG" and z["is_fresh"]]

    fvg_hit = False
    for z in reversed(bull_fvgs[-4:]):
        if z["bottom"] <= low <= z["top"] * 1.015 and close >= z["midpoint"]:
            score += 25
            setup_tags.append("Bullish FVG 50% Test")
            smc_tags.append(f"Bull FVG ({z['bottom']}–{z['top']})")
            fvg_hit = True
            break
    if not fvg_hit and bull_fvgs:
        nearest = bull_fvgs[-1]
        if close > nearest["top"] and ((close - nearest["top"]) / close) <= 0.04:
            score += 15
            setup_tags.append("Bullish FVG Support")
            smc_tags.append(f"Bull FVG ({nearest['bottom']}–{nearest['top']})")

    for z in reversed(bear_fvgs[-3:]):
        if z["bottom"] * 0.99 <= close <= z["top"]:
            score = max(0, score - 12)
            smc_tags.append("Bearish FVG Overhead")
            break

    recent_gaps = detect_market_gaps(enriched.tail(15), lookback=15)
    for g in reversed(recent_gaps[-3:]):
        if g["direction"] == "Gap Up" and "Breakaway" in g["gap_class"] and not g["filled"]:
            score += 20
            setup_tags.append("Breakaway Gap & Go")
            smc_tags.append(f"Gap Up +{g['gap_pct']}%")
            break
        if g["direction"] == "Gap Up" and "Common" in g["gap_class"] and g["filled"] and close > ema20:
            score += 12
            setup_tags.append("Gap Fill Reclaim")
            smc_tags.append(f"Filled Gap +{g['gap_pct']}%")
            break

    score = int(min(max(score, 0), 100))
    stop_loss = max(0.05, close - sl_mult * atr14)
    risk_unit = max(0.01, close - stop_loss)
    target = max(close + tp_mult * atr14, close + 2.2 * risk_unit)

    adx14 = safe_float(last.get("ADX14", 0.0))
    ema_slope = safe_float(last.get("EMA20_Slope", 0.0))
    trend_strong = adx14 >= 20.0 and ema_slope > 0.0

    if not regime_bullish:
        status = "PAUSED (MACRO)"
        grade = "-"
    elif score >= 85 and close > ema20 and atr14 > 0 and trend_strong:
        status = "READY"
        grade = "A+"
    elif score >= 75 and close > ema20 and atr14 > 0 and trend_strong:
        status = "READY"
        grade = "A"
    elif score >= 55:
        status = "WATCH"
        grade = "B"
    else:
        status = "AVOID"
        grade = "C"

    return {
        "Ticker": ticker,
        "Sector": get_sector(ticker),
        "Status": status,
        "Grade": grade,
        "Score": score,
        "RS_vs_Bench": round(rs_vs_nifty, 2),
        "ADX14": round(safe_float(last.get("ADX14", 0.0)), 1),
        "RSI14": round(rsi14, 1),
        "Setup": " + ".join(setup_tags) if setup_tags else "No Confluence",
        "SMC_Signal": " | ".join(smc_tags) if smc_tags else "None",
        "Entry": round(close, 2),
        "StopLoss": round(stop_loss, 2),
        "Target": round(target, 2),
        "ATR14": round(atr14, 2),
    }


def scan_watchlist(
    stock_data: Dict[str, pd.DataFrame],
    sl_mult: float = 2.0,
    tp_mult: float = 3.5,
    regime_bullish: bool = True,
    nifty_roc20: float = 0.0,
    max_atr_pct: float = 4.5,
) -> pd.DataFrame:
    """Scan all watchlist assets and sort by Score and Relative Strength descending."""
    rows = [
        score_stock_snapshot(
            ticker=t,
            df=df,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            regime_bullish=regime_bullish,
            nifty_roc20=nifty_roc20,
            max_atr_pct=max_atr_pct,
        )
        for t, df in stock_data.items()
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["Score", "RS_vs_Bench"], ascending=[False, False]).reset_index(drop=True)


def compute_sector_strength(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate average RS_vs_Bench, average Score, and Top Leader per Sector."""
    if scan_df is None or scan_df.empty or "Sector" not in scan_df.columns:
        return pd.DataFrame()
    rows = []
    for sector, grp in scan_df.groupby("Sector"):
        top_row = grp.sort_values(by=["RS_vs_Bench", "Score"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "Sector": sector,
                "Avg_RS%": round(float(grp["RS_vs_Bench"].mean()), 2),
                "Avg_Score": round(float(grp["Score"].mean()), 1),
                "Assets": int(len(grp)),
                "Top_Leader": str(top_row["Ticker"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by="Avg_RS%", ascending=False).reset_index(drop=True)
