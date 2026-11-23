"""
data_feed.py — Safe market data loader & Broker API Bridge for Maddy AI.
Features:
  - yfinance historical & latest NSE loader with MultiIndex flattening
  - float(np.squeeze()) scalar protection (zero Pandas Series errors)
  - Streamlit caching with standalone fallback
  - Future-ready BrokerBridge (Angel One SmartAPI / Upstox / Dhan) via st.secrets
  - Deterministic offline demo data generator for instant testing
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


BENCHMARK_TICKER = "^NSEI"
CRYPTO_BENCHMARK = "BTC-USD"
GLOBAL_BENCHMARK = "SPY"

DEFAULT_WATCHLIST: List[str] = [
    # Core 22 Growth Leaders + Precious Metals ETFs
    "TRENT.NS",
    "HAL.NS",
    "BEL.NS",
    "DIXON.NS",
    "PERSISTENT.NS",
    "TVSMOTOR.NS",
    "INDIGO.NS",
    "CHOLAFIN.NS",
    "CGPOWER.NS",
    "CUMMINSIND.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "BAJFINANCE.NS",
    "BHARTIARTL.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "RELIANCE.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "GOLDBEES.NS",
    "SILVERBEES.NS",
    # Top 30 Liquid Nifty 50 Bluechips (Total: 52 NSE Assets)
    "HDFCBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "INDUSINDBK.NS",
    "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS",
    "JIOFIN.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "MARUTI.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "APOLLOHOSP.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "NESTLEIND.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "ONGC.NS",
]

DEFAULT_CRYPTO_WATCHLIST: List[str] = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
    "AVAX-USD",
    "LINK-USD",
    "DOGE-USD",
    "SUI-USD",
]

DEFAULT_COMMODITIES_GLOBAL_WATCHLIST: List[str] = [
    "GC=F",
    "SI=F",
    "CL=F",
    "HG=F",
    "GLD",
    "SLV",
    "SPY",
    "QQQ",
]

SECTOR_MAP: Dict[str, str] = {
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "TATAMOTORS.NS": "Auto",
    "M&M.NS": "Auto",
    "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "BAJFINANCE.NS": "Finance",
    "TITAN.NS": "Consumer",
    "BEL.NS": "Defence",
    "NTPC.NS": "Power",
    "BTC-USD": "Layer-1",
    "ETH-USD": "Layer-1",
    "SOL-USD": "Layer-1",
    "ADA-USD": "Layer-1",
    "AVAX-USD": "Layer-1",
    "SUI-USD": "Layer-1",
    "BNB-USD": "Exchange",
    "XRP-USD": "Payments",
    "LINK-USD": "Oracle/DeFi",
    "DOGE-USD": "Meme",
    "GOLDBEES.NS": "Precious Metals",
    "SILVERBEES.NS": "Precious Metals",
    "GC=F": "Precious Metals",
    "SI=F": "Precious Metals",
    "GLD": "Precious Metals",
    "SLV": "Precious Metals",
    "CL=F": "Energy",
    "HG=F": "Industrial Metals",
    "SPY": "US Index",
    "TRENT.NS": "Consumer",
    "HAL.NS": "Defence",
    "DIXON.NS": "Electronics",
    "PERSISTENT.NS": "IT",
    "TVSMOTOR.NS": "Auto",
    "INDIGO.NS": "Aviation",
    "CHOLAFIN.NS": "Finance",
    "CGPOWER.NS": "Capital Goods",
    "CUMMINSIND.NS": "Capital Goods",
    "POWERGRID.NS": "Power",
    "QQQ": "US Tech",
    "INDUSINDBK.NS": "Banking",
    "BAJAJFINSV.NS": "Finance",
    "SHRIRAMFIN.NS": "Finance",
    "JIOFIN.NS": "Finance",
    "SBILIFE.NS": "Insurance",
    "HDFCLIFE.NS": "Insurance",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "CIPLA.NS": "Pharma",
    "APOLLOHOSP.NS": "Healthcare",
    "NESTLEIND.NS": "FMCG",
    "ASIANPAINT.NS": "Consumer",
    "ULTRACEMCO.NS": "Cement",
    "TATASTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals",
    "JSWSTEEL.NS": "Metals",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any scalar, 1-element Series, or numpy array to Python float."""
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series, pd.DataFrame)):
            arr = value.to_numpy()
            if arr.size == 0:
                return default
            val = float(np.squeeze(arr.flat[-1]))
        else:
            val = float(np.squeeze(value))
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError, IndexError):
        return default


def get_sector(ticker: str) -> str:
    """Return sector label for an NSE ticker."""
    return SECTOR_MAP.get(ticker.upper().strip(), "Other")


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and guarantee clean float OHLCV DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]).strip() for col in out.columns]
    else:
        out.columns = [str(col).strip() for col in out.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()

    out = out[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _download_raw(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        return _clean_ohlcv(raw)
    except Exception:
        return pd.DataFrame()


if _HAS_STREAMLIT:
    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        return _download_raw(ticker, period=period, interval=interval)
else:
    def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        return _download_raw(ticker, period=period, interval=interval)


def load_watchlist_data(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Load clean OHLCV DataFrames for all valid tickers in the watchlist."""
    data: Dict[str, pd.DataFrame] = {}
    for raw_t in tickers:
        ticker = raw_t.upper().strip()
        if not ticker:
            continue
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        if not df.empty and len(df) >= 60:
            data[ticker] = df
    return data


def generate_demo_ohlcv(
    ticker: str,
    days: int = 360,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Deterministic offline OHLCV generator (with realistic occasional gaps & swings)
    so the complete terminal can be tested instantly without internet.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp("2026-11-20"), periods=days)
    base_price = float(rng.uniform(450, 2800))

    opens = np.zeros(days)
    highs = np.zeros(days)
    lows = np.zeros(days)
    closes = np.zeros(days)
    volumes = rng.integers(600_000, 6_000_000, size=days).astype(float)

    prev_close = base_price
    for i in range(days):
        # Occasional overnight gap (~6% of sessions)
        gap = 0.0
        if rng.random() < 0.06:
            gap = float(rng.choice([-1, 1]) * rng.uniform(0.014, 0.032))
            volumes[i] *= rng.uniform(1.5, 2.3)

        o = prev_close * (1.0 + gap)
        # Mild bullish drift with cyclical swings
        drift = 0.0006 + 0.0008 * np.sin(i / 28.0)
        intraday_ret = float(rng.normal(drift, 0.014))
        c = max(10.0, o * (1.0 + intraday_ret))
        wick_up = abs(float(rng.normal(0.006, 0.003))) * max(o, c)
        wick_dn = abs(float(rng.normal(0.006, 0.003))) * min(o, c)

        opens[i] = o
        closes[i] = c
        highs[i] = max(o, c) + wick_up
        lows[i] = max(1.0, min(o, c) - wick_dn)
        prev_close = c

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes.astype(int),
        },
        index=dates,
    )


class BrokerBridge:
    """
    Phase 5 Live Broker Bridge (Angel One SmartAPI / Upstox / Dhan).
    Reads credentials safely from Streamlit Secrets or environment variables.
    Falls back cleanly to yfinance / dry-run mode when keys are not configured.
    """

    SUPPORTED_BROKERS = ("None (yfinance only)", "Angel One SmartAPI", "Upstox", "Dhan")

    def __init__(self, broker_name: str = "None (yfinance only)"):
        self.broker_name = broker_name
        self.api_key = self._get_secret("BROKER_API_KEY")
        self.access_token = self._get_secret("BROKER_ACCESS_TOKEN")
        self.client_id = self._get_secret("BROKER_CLIENT_ID")

    @staticmethod
    def _get_secret(key: str) -> str:
        if _HAS_STREAMLIT:
            try:
                if key in st.secrets:
                    return str(st.secrets[key]).strip()
            except Exception:
                pass
        return os.environ.get(key, "").strip()

    @property
    def is_configured(self) -> bool:
        return (
            self.broker_name != "None (yfinance only)"
            and bool(self.api_key)
            and bool(self.access_token)
        )

    def status_summary(self) -> Dict[str, Any]:
        if self.broker_name == "None (yfinance only)":
            return {
                "connected": False,
                "mode": "yfinance Historical / Delayed Feed",
                "message": "Select Angel One, Upstox, or Dhan and add BROKER_API_KEY + BROKER_ACCESS_TOKEN in Streamlit Secrets to enable live order routing.",
            }
        if not self.is_configured:
            return {
                "connected": False,
                "mode": f"{self.broker_name} (Keys Missing — Dry-Run Mode)",
                "message": "Add BROKER_API_KEY and BROKER_ACCESS_TOKEN in Streamlit Secrets to activate live execution.",
            }
        return {
            "connected": True,
            "mode": f"{self.broker_name} (Authenticated)",
            "message": "Credentials detected. Live order payloads ready (paper-first safety lock recommended).",
        }

    def build_order_payload(
        self,
        ticker: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        target: float,
    ) -> Dict[str, Any]:
        """Construct a broker-normalized NSE order payload for review or execution."""
        clean_symbol = ticker.replace(".NS", "").upper()
        return {
            "broker": self.broker_name,
            "exchange": "NSE",
            "tradingsymbol": clean_symbol,
            "transaction_type": side.upper(),
            "quantity": int(qty),
            "order_type": "LIMIT",
            "price": round(float(entry_price), 2),
            "stoploss": round(float(stop_loss), 2),
            "target": round(float(target), 2),
            "product": "CNC",
            "dry_run": not self.is_configured,
        }
