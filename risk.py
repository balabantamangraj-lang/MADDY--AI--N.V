"""
risk.py — Institutional Risk Gate & Block-Bootstrap Monte Carlo Engine for Maddy AI.
Rules enforced:
  1. 1% Risk per trade position sizing
  2. 20% Single-stock capital cap
  3. 30% Sector exposure cap
  4. 5% Max total portfolio heat
  5. 30-day rolling Correlation Shield (>0.80 reject)
  6. 25% Drawdown Kill-Switch
Plus:
  - 1,000-run Block-Bootstrap Monte Carlo stress tester (Risk of Ruin, Median Max DD, 5th %ile Equity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

from data_feed import get_sector, safe_float


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    max_stock_alloc_pct: float = 0.20
    max_sector_alloc_pct: float = 0.30
    max_portfolio_heat_pct: float = 0.05
    max_correlation: float = 0.80
    kill_switch_dd_pct: float = 0.25
    min_rr_ratio: float = 2.2              # Lock minimum 2.2:1 Reward-to-Risk
    be_trigger_r: float = 1.6              # Shift SL to Break-Even at +1.2R
    chandelier_atr_mult: float = 2.5       # Chandelier Trailing Stop distance


def enforce_min_rr(
    entry_price: float,
    stop_loss: float,
    target: float,
    min_rr: float = 2.2,
) -> float:
    """Lock target so Reward-to-Risk is always at least min_rr (default 2.2:1)."""
    risk_unit = max(0.01, entry_price - stop_loss)
    min_target = entry_price + (min_rr * risk_unit)
    return round(max(float(target), min_target), 2)


def update_chandelier_stop(
    entry_price: float,
    current_sl: float,
    highest_high: float,
    atr_now: float,
    init_risk: float,
    config: RiskConfig = RiskConfig(),
    buffer_pct: float = 0.0015,
) -> Tuple[float, str]:
    """
    Two-stage Chandelier Trailing Stop-Loss:
      Stage 1 (Break-Even): Once profit reaches +be_trigger_r (1.2R), shift SL to entry + fee buffer.
      Stage 2 (Chandelier Trail): Trail SL at (highest_high - chandelier_atr_mult * ATR14), ratchet up only.
    Returns (new_stop_loss, stage_label).
    """
    if init_risk <= 0 or highest_high <= entry_price:
        return current_sl, "INITIAL"

    r_multiple = (highest_high - entry_price) / init_risk
    if r_multiple < config.be_trigger_r:
        return current_sl, "INITIAL"

    be_stop = entry_price * (1.0 + buffer_pct)
    chandelier_stop = highest_high - (config.chandelier_atr_mult * max(atr_now, 0.01))
    new_sl = max(current_sl, be_stop, chandelier_stop)

    if chandelier_stop > be_stop and new_sl > current_sl:
        return round(new_sl, 2), "CHANDELIER"
    if new_sl > current_sl:
        return round(new_sl, 2), "BREAKEVEN"
    return current_sl, ("CHANDELIER" if current_sl > be_stop else "BREAKEVEN")


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    config: RiskConfig = RiskConfig(),
    allow_fractional: bool = False,
) -> float:
    """Position size respecting 1% risk and 20% single-asset cap (whole shares for NSE, fractional for Crypto)."""
    if equity <= 0 or entry_price <= 0 or stop_loss >= entry_price:
        return 0
    risk_per_unit = entry_price - stop_loss
    if risk_per_unit <= 0:
        return 0
    raw_risk = (equity * config.risk_per_trade_pct) / risk_per_unit
    raw_cap = (equity * config.max_stock_alloc_pct) / entry_price
    qty = max(0.0, min(raw_risk, raw_cap))
    if allow_fractional:
        return round(qty, 4) if qty >= 0.0001 else 0.0
    return int(math.floor(qty))


def compute_portfolio_heat(open_positions: Dict[str, dict], equity: float) -> float:
    """Current open risk as fraction of total equity."""
    if equity <= 0 or not open_positions:
        return 0.0
    total_risk = sum(
        max(0.0, (safe_float(p["entry"]) - safe_float(p["stop_loss"])) * float(p["qty"]))
        for p in open_positions.values()
    )
    return total_risk / equity


def compute_sector_exposure(
    open_positions: Dict[str, dict],
    equity: float,
    current_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Sector exposure fractions keyed by sector name."""
    if equity <= 0 or not open_positions:
        return {}
    sector_vals: Dict[str, float] = {}
    for ticker, pos in open_positions.items():
        sector = pos.get("sector") or get_sector(ticker)
        px = safe_float(current_prices.get(ticker)) if current_prices and ticker in current_prices else safe_float(pos["entry"])
        sector_vals[sector] = sector_vals.get(sector, 0.0) + px * float(pos["qty"])
    return {sec: val / equity for sec, val in sector_vals.items()}


def check_correlation_shield(
    candidate_ticker: str,
    open_tickers: List[str],
    returns_df: Optional[pd.DataFrame],
    max_corr: float = 0.80,
    window: int = 30,
) -> Tuple[bool, float, Optional[str]]:
    """Reject if 30-day rolling correlation with any open position exceeds max_corr."""
    if returns_df is None or returns_df.empty or not open_tickers or candidate_ticker not in returns_df.columns:
        return True, 0.0, None

    recent = returns_df.tail(window)
    if len(recent[candidate_ticker].dropna()) < 10:
        return True, 0.0, None

    highest = 0.0
    conflict = None
    for open_t in open_tickers:
        if open_t == candidate_ticker or open_t not in recent.columns:
            continue
        pair = recent[[candidate_ticker, open_t]].dropna()
        if len(pair) < 10:
            continue
        c = safe_float(pair[candidate_ticker].corr(pair[open_t]))
        if c > highest:
            highest = c
            conflict = open_t
    return (highest <= max_corr), round(highest, 2), conflict


def evaluate_trade_gate(
    ticker: str,
    entry_price: float,
    stop_loss: float,
    equity: float,
    peak_equity: float,
    open_positions: Dict[str, dict],
    returns_df: Optional[pd.DataFrame] = None,
    current_prices: Optional[Dict[str, float]] = None,
    config: RiskConfig = RiskConfig(),
    allow_fractional: bool = False,
) -> Tuple[bool, float, str]:
    """Master 5-Rule Institutional Risk Gate + Kill-Switch."""
    if ticker in open_positions:
        return False, 0, "Already in Position"

    if peak_equity > 0:
        dd = (peak_equity - equity) / peak_equity
        if dd >= config.kill_switch_dd_pct:
            return False, 0, f"Kill-Switch Active (DD {dd*100:.1f}% >= {config.kill_switch_dd_pct*100:.0f}%)"

    qty = calculate_position_size(equity, entry_price, stop_loss, config, allow_fractional=allow_fractional)
    if qty <= 0:
        return False, 0, "Position Size Zero"

    heat = compute_portfolio_heat(open_positions, equity)
    trade_risk = ((entry_price - stop_loss) * qty) / equity
    if heat + trade_risk > config.max_portfolio_heat_pct + 1e-9:
        return False, 0, f"Portfolio Heat Cap ({(heat + trade_risk)*100:.1f}% > {config.max_portfolio_heat_pct*100:.0f}%)"

    sector = get_sector(ticker)
    sec_exp = compute_sector_exposure(open_positions, equity, current_prices)
    new_sec = sec_exp.get(sector, 0.0) + (entry_price * qty) / equity
    if new_sec > config.max_sector_alloc_pct + 1e-9:
        return False, 0, f"Sector Cap [{sector}] ({new_sec*100:.1f}% > {config.max_sector_alloc_pct*100:.0f}%)"

    corr_ok, max_corr, conflict = check_correlation_shield(
        candidate_ticker=ticker,
        open_tickers=list(open_positions.keys()),
        returns_df=returns_df,
        max_corr=config.max_correlation,
    )
    if not corr_ok:
        return False, 0, f"High Correlation ({max_corr:.2f} with {conflict})"

    return True, qty, "APPROVED"


def run_block_bootstrap_mc(
    equity_series: pd.Series,
    initial_capital: float = 100_000.0,
    n_sims: int = 1000,
    block_size: int = 5,
    ruin_dd_pct: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Block-Bootstrap Monte Carlo Simulation (preserves volatility clustering).
    Returns percentile equity paths, Risk of Ruin %, Median Max DD %, and 5th Percentile Equity.
    """
    if equity_series is None or len(equity_series) < 20:
        return {"error": "Need at least 20 equity bars for Monte Carlo."}

    daily_rets = equity_series.pct_change().dropna().to_numpy(dtype=float)
    n_days = len(daily_rets)
    if n_days < 10:
        return {"error": "Insufficient return series."}

    rng = np.random.default_rng(seed)
    blk = max(2, min(block_size, n_days // 3))
    n_blocks = math.ceil(n_days / blk)
    max_start = max(1, n_days - blk + 1)

    paths = np.zeros((n_sims, n_days), dtype=float)
    max_dds = np.zeros(n_sims, dtype=float)
    ruin_hits = 0

    for s in range(n_sims):
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled = np.concatenate([daily_rets[st : st + blk] for st in starts])[:n_days]
        eq = initial_capital * np.cumprod(1.0 + sampled)
        paths[s] = eq

        peaks = np.maximum.accumulate(eq)
        dds = (eq - peaks) / np.where(peaks > 0, peaks, 1.0)
        min_dd = float(np.min(dds))
        max_dds[s] = min_dd * 100.0
        if min_dd <= -ruin_dd_pct:
            ruin_hits += 1

    final_equities = paths[:, -1]
    p5_path = np.percentile(paths, 5, axis=0)
    p50_path = np.percentile(paths, 50, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)

    # Keep up to 40 sample paths for lightweight mobile Plotly rendering
    sample_idx = np.linspace(0, n_sims - 1, min(40, n_sims), dtype=int)

    return {
        "n_sims": n_sims,
        "n_days": n_days,
        "risk_of_ruin_pct": round((ruin_hits / n_sims) * 100.0, 2),
        "median_max_dd_pct": round(float(np.median(max_dds)), 2),
        "p5_final_equity": round(float(np.percentile(final_equities, 5)), 2),
        "median_final_equity": round(float(np.median(final_equities)), 2),
        "p95_final_equity": round(float(np.percentile(final_equities, 95)), 2),
        "p5_path": p5_path,
        "p50_path": p50_path,
        "p95_path": p95_path,
        "sample_paths": paths[sample_idx],
    }
