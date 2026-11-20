"""
scanner.py — Dual-Engine White-Box Strategy & SMC Scanner for Maddy AI.
Implements:
  - Technical Indicators: EMA20, EMA50, ATR14, ATR%, Volume MA20
  - Smart Money Concepts (SMC):
      * 3-Candle Bullish & Bearish Fair Value Gap (FVG) detector
      * 50% Midpoint calculation & fresh mitigation check
      * Common vs. Breakaway Market Gap detector (Gap Fill + Gap & Go)
  - Macro Filter: Nifty 50 Regime Banner
  - Confluence Scoring System (0-100)
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from data_feed import get_sector, safe_float


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA20, EMA50, ATR14, ATR%, and 20-day Volume MA cleanly."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    prev_close = close.shift(1)

    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = true_range.rolling(window=14, min_periods=14).mean()
    out["ATR_Pct"] = np.where(close > 0, (out["ATR14"] / close) * 100.0, 0.0)
    out["Vol_MA20"] = out["Volume"].astype(float).rolling(window=20, min_periods=10).mean()
    return out


def detect_fvg_zones(
    df: pd.DataFrame,
    lookback: int = 80,
    min_gap_pct: float = 0.002,
) -> List[Dict[str, Any]]:
    """
    3-Candle Fair Value Gap (FVG) detector with 50% midpoint & mitigation tracking.
      - Bullish FVG: Low[i] > High[i-2]
      - Bearish FVG: High[i] < Low[i-2]
    Tracks whether subsequent candles mitigated the 50% midpoint level.
    """
    if df is None or len(df) < 5:
        return []

    sub = df.tail(lookback).copy()
    highs = sub["High"].to_numpy(dtype=float)
    lows = sub["Low"].to_numpy(dtype=float)
    closes = sub["Close"].to_numpy(dtype=float)
    dates = sub.index

    zones: List[Dict[str, Any]] = []
    n = len(sub)

    for i in range(2, n):
        # Bullish FVG: candle[i].low > candle[i-2].high
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            if closes[i - 1] > 0 and (gap_size / closes[i - 1]) >= min_gap_pct:
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

        # Bearish FVG: candle[i].high < candle[i-2].low
        elif highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            if closes[i - 1] > 0 and (gap_size / closes[i - 1]) >= min_gap_pct:
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
    """
    Detect overnight Market Gaps (Gap Up / Gap Down) and classify them as:
      - Breakaway Gap (Gap & Go): Volume >= 1.3x 20D avg AND closes in gap direction
      - Common Gap (Gap Fill candidate): Lower volume or fades back toward prior close
    """
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

        if vol_ratio >= 1.3 and followed_through:
            gap_class = "Breakaway Gap (Gap & Go)"
        else:
            gap_class = "Common Gap (Gap Fill)"

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
        }

    last = enriched.iloc[-1]
    close = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    is_bullish = bool(close > ema20 and ema20 >= ema50)

    return {
        "is_bullish": is_bullish,
        "label": "BULLISH · TRADE ON" if is_bullish else "BEARISH · CASH SAFE",
        "close": round(close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
    }


def score_stock_snapshot(
    ticker: str,
    df: pd.DataFrame,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    regime_bullish: bool = True,
) -> Dict[str, Any]:
    """
    Dual-Engine White-Box Scoring (0-100):
      - Layer 1 (Trend Base, up to 55 pts): Price > EMA20, EMA20 > EMA50, Healthy ATR, Volume
      - Layer 2 (SMC Booster, up to 45 pts): Fresh Bullish FVG 50% mitigation test + Breakaway Gap momentum
    """
    enriched = add_indicators(df)
    if enriched.empty or len(enriched) < 50:
        return {
            "Ticker": ticker,
            "Sector": get_sector(ticker),
            "Status": "NO DATA",
            "Score": 0,
            "Setup": "Insufficient History",
            "SMC_Signal": "None",
            "Entry": 0.0,
            "StopLoss": 0.0,
            "Target": 0.0,
            "ATR14": 0.0,
            "ATR_Pct": 0.0,
        }

    last = enriched.iloc[-1]
    close = safe_float(last["Close"])
    low = safe_float(last["Low"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    atr14 = safe_float(last["ATR14"])
    atr_pct = safe_float(last["ATR_Pct"])
    vol = safe_float(last["Volume"])
    vol_ma20 = safe_float(last["Vol_MA20"], default=1.0)

    score = 0
    setup_tags: List[str] = []
    smc_tags: List[str] = []

    # --- Layer 1: Trend Base (max 55 pts) ---
    if close > ema20:
        score += 20
        setup_tags.append("EMA20 Trend")
    if ema20 > ema50:
        score += 15
    if 0.8 <= atr_pct <= 4.5:
        score += 12
    elif atr_pct > 0:
        score += 5
    if vol_ma20 > 0 and vol >= vol_ma20:
        score += 8

    # --- Layer 2: SMC Booster (max 45 pts) ---
    fvg_zones = detect_fvg_zones(enriched, lookback=45)
    bull_fvgs = [z for z in fvg_zones if z["type"] == "Bullish FVG" and z["is_fresh"]]
    bear_fvgs = [z for z in fvg_zones if z["type"] == "Bearish FVG" and z["is_fresh"]]

    # Check if price is testing/mitigating a fresh Bullish FVG 50% midpoint or just above it
    fvg_hit = False
    for z in reversed(bull_fvgs[-4:]):
        if z["bottom"] <= low <= z["top"] * 1.015 and close >= z["midpoint"]:
            score += 25
            setup_tags.append("Bullish FVG 50% Test")
            smc_tags.append(f"Bull FVG ({z['bottom']}–{z['top']})")
            fvg_hit = True
            break
    if not fvg_hit and bull_fvgs:
        # Fresh bullish FVG support below current price
        nearest = bull_fvgs[-1]
        if close > nearest["top"] and ((close - nearest["top"]) / close) <= 0.04:
            score += 15
            setup_tags.append("Bullish FVG Support")
            smc_tags.append(f"Bull FVG ({nearest['bottom']}–{nearest['top']})")

    # Penalize if immediately under a fresh Bearish FVG resistance
    for z in reversed(bear_fvgs[-3:]):
        if z["bottom"] * 0.99 <= close <= z["top"]:
            score = max(0, score - 12)
            smc_tags.append("Bearish FVG Overhead")
            break

    # Check recent Market Gaps (last 10 bars)
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
    target = close + tp_mult * atr14

    if not regime_bullish:
        status = "PAUSED (MACRO)"
    elif score >= 60 and close > ema20 and atr14 > 0:
        status = "READY"
    elif score >= 40:
        status = "WATCH"
    else:
        status = "AVOID"

    return {
        "Ticker": ticker,
        "Sector": get_sector(ticker),
        "Status": status,
        "Score": score,
        "Setup": " + ".join(setup_tags) if setup_tags else "No Confluence",
        "SMC_Signal": " | ".join(smc_tags) if smc_tags else "None",
        "Entry": round(close, 2),
        "StopLoss": round(stop_loss, 2),
        "Target": round(target, 2),
        "ATR14": round(atr14, 2),
        "ATR_Pct": round(atr_pct, 2),
    }


def scan_watchlist(
    stock_data: Dict[str, pd.DataFrame],
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    regime_bullish: bool = True,
) -> pd.DataFrame:
    """Scan all watchlist stocks and sort by Score descending."""
    rows = [
        score_stock_snapshot(
            ticker=t,
            df=df,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            regime_bullish=regime_bullish,
        )
        for t, df in stock_data.items()
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
