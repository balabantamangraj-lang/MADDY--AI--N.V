"""
backtest.py — Multi-Asset Dual-Engine Portfolio Backtest, Parameter Heatmap,
and Walk-Forward Optimization (WFO) engine for Maddy AI.
Features:
  - Realistic slippage (0.10%) and brokerage/commission (0.05%)
  - Stale signal expiry (pending setups expire after max_signal_age bars)
  - Full Trade Log + Institutional Risk Gate Rejection Ledger
  - CAPM Alpha & Beta vs Nifty 50 Benchmark
  - Parameter Stability Heatmap (SL vs TP ATR multipliers)
  - Rolling Walk-Forward Optimization (In-Sample -> Out-of-Sample)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from data_feed import get_sector, safe_float
from risk import RiskConfig, evaluate_trade_gate
from scanner import add_indicators


def _precompute_smc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized / fast bar-by-bar Bullish FVG & Breakaway Gap flags for backtesting."""
    out = add_indicators(df)
    if out.empty:
        return out

    highs = out["High"].to_numpy(dtype=float)
    lows = out["Low"].to_numpy(dtype=float)
    opens = out["Open"].to_numpy(dtype=float)
    closes = out["Close"].to_numpy(dtype=float)
    vols = out["Volume"].to_numpy(dtype=float)
    vol_ma = out["Vol_MA20"].fillna(out["Volume"]).to_numpy(dtype=float)
    n = len(out)

    fvg_support = np.zeros(n, dtype=bool)
    breakaway_gap = np.zeros(n, dtype=bool)

    active_fvgs: List[Tuple[float, float, float, int]] = []  # (bottom, top, mid, born_idx)

    for i in range(2, n):
        # Detect new Bullish FVG at bar i
        if lows[i] > highs[i - 2] and closes[i - 1] > 0:
            if (lows[i] - highs[i - 2]) / closes[i - 1] >= 0.002:
                b = highs[i - 2]
                t = lows[i]
                active_fvgs.append((b, t, (b + t) / 2.0, i))

        # Check fresh FVG 50% mitigation or near support (max 25 bars old)
        still_active = []
        for b, t, mid, born in active_fvgs:
            if i - born > 25 or lows[i] <= b:
                continue
            if b <= lows[i] <= t * 1.015 and closes[i] >= mid:
                fvg_support[i] = True
            still_active.append((b, t, mid, born))
        active_fvgs = still_active[-5:]

        # Breakaway Gap Up detection
        prev_c = closes[i - 1]
        if prev_c > 0:
            gp = (opens[i] - prev_c) / prev_c
            vr = vols[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
            if gp >= 0.010 and vr >= 1.3 and closes[i] >= opens[i]:
                breakaway_gap[i] = True

    out["SMC_FVG_Hit"] = fvg_support
    out["SMC_Breakaway_Gap"] = breakaway_gap
    return out


def run_portfolio_backtest(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    slippage_pct: float = 0.0010,
    commission_pct: float = 0.0005,
    stale_signal_bars: int = 3,
    risk_config: RiskConfig = RiskConfig(),
) -> Dict[str, Any]:
    """Run multi-asset portfolio backtest with slippage, fees, stale signal expiry, and audit trail."""
    if not stock_data or nifty_df is None or nifty_df.empty:
        return {"error": "Missing stock or Nifty 50 data."}

    nifty = add_indicators(nifty_df)
    enriched: Dict[str, pd.DataFrame] = {}
    close_matrix: Dict[str, pd.Series] = {}

    for ticker, df in stock_data.items():
        ind = _precompute_smc_flags(df)
        if not ind.empty and len(ind) >= 60:
            enriched[ticker] = ind
            close_matrix[ticker] = ind["Close"]

    if not enriched or len(nifty) < 60:
        return {"error": "Need at least 60 daily bars to run backtest."}

    returns_df = pd.DataFrame(close_matrix).pct_change()
    common_dates = nifty.index[50:]

    cash = float(initial_capital)
    peak_equity = float(initial_capital)
    open_positions: Dict[str, dict] = {}
    pending_signals: Dict[str, dict] = {}  # Stale signal expiry tracker

    equity_records: List[Dict[str, Any]] = []
    trade_log: List[Dict[str, Any]] = []
    rejection_ledger: List[Dict[str, Any]] = []

    for bar_idx, dt in enumerate(common_dates):
        current_prices: Dict[str, float] = {}
        to_close: List[str] = []

        # 1. Manage Open Positions (Stop-Loss / Target)
        for ticker, pos in list(open_positions.items()):
            df_t = enriched[ticker]
            if dt not in df_t.index:
                current_prices[ticker] = pos["last_price"]
                continue

            row = df_t.loc[dt]
            low = safe_float(row["Low"])
            high = safe_float(row["High"])
            close = safe_float(row["Close"])
            current_prices[ticker] = close
            pos["last_price"] = close

            exit_price = None
            exit_reason = None

            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"] * (1.0 - slippage_pct)
                exit_reason = "Stop-Loss"
            elif high >= pos["target"]:
                exit_price = pos["target"] * (1.0 - slippage_pct)
                exit_reason = "Target Hit"

            if exit_price is not None:
                gross = exit_price * pos["qty"]
                fee = gross * commission_pct
                net = gross - fee
                cash += net
                pnl = net - pos["cost_basis"]
                pnl_pct = (pnl / pos["cost_basis"] * 100.0) if pos["cost_basis"] > 0 else 0.0

                trade_log.append(
                    {
                        "Ticker": ticker,
                        "Sector": pos["sector"],
                        "Setup": pos["setup"],
                        "EntryDate": pos["entry_date"].strftime("%Y-%m-%d"),
                        "ExitDate": dt.strftime("%Y-%m-%d"),
                        "Qty": pos["qty"],
                        "Entry": round(pos["entry"], 2),
                        "Exit": round(exit_price, 2),
                        "PnL": round(pnl, 2),
                        "Return%": round(pnl_pct, 2),
                        "Reason": exit_reason,
                    }
                )
                to_close.append(ticker)

        for ticker in to_close:
            open_positions.pop(ticker, None)

        holdings_val = sum(
            current_prices.get(t, pos["last_price"]) * pos["qty"]
            for t, pos in open_positions.items()
        )
        equity = cash + holdings_val
        peak_equity = max(peak_equity, equity)

        # 2. Expire stale pending signals
        for t in list(pending_signals.keys()):
            if bar_idx - pending_signals[t]["bar_idx"] > stale_signal_bars:
                pending_signals.pop(t, None)

        # 3. Macro Regime Check on Nifty 50
        n_row = nifty.loc[dt]
        nifty_bullish = (
            safe_float(n_row["Close"]) > safe_float(n_row["EMA20"])
            and safe_float(n_row["EMA20"]) >= safe_float(n_row["EMA50"])
        )

        # 4. Dual-Engine Entry Scan
        if nifty_bullish:
            returns_slice = returns_df.loc[:dt]
            for ticker, df_t in enriched.items():
                if ticker in open_positions or dt not in df_t.index:
                    continue

                row = df_t.loc[dt]
                close = safe_float(row["Close"])
                ema20 = safe_float(row["EMA20"])
                ema50 = safe_float(row["EMA50"])
                atr14 = safe_float(row["ATR14"])
                atr_pct = safe_float(row["ATR_Pct"])
                fvg_hit = bool(row["SMC_FVG_Hit"])
                gap_hit = bool(row["SMC_Breakaway_Gap"])

                dist_pct = ((close - ema20) / ema20 * 100.0) if ema20 > 0 else 99.0
                trend_ok = close > ema20 and ema20 > ema50 and atr14 > 0 and 0.8 <= atr_pct <= 4.5
                smc_boost = fvg_hit or gap_hit

                if not (trend_ok and (dist_pct <= 3.0 or smc_boost)):
                    continue

                setup_label = "EMA Trend"
                if fvg_hit and gap_hit:
                    setup_label = "EMA + FVG 50% + Gap"
                elif fvg_hit:
                    setup_label = "EMA + Bullish FVG 50%"
                elif gap_hit:
                    setup_label = "EMA + Breakaway Gap"

                pending_signals[ticker] = {"bar_idx": bar_idx, "setup": setup_label}

                entry_exec = close * (1.0 + slippage_pct)
                stop_loss = max(0.05, entry_exec - sl_mult * atr14)
                target = entry_exec + tp_mult * atr14

                approved, qty, reason = evaluate_trade_gate(
                    ticker=ticker,
                    entry_price=entry_exec,
                    stop_loss=stop_loss,
                    equity=equity,
                    peak_equity=peak_equity,
                    open_positions=open_positions,
                    returns_df=returns_slice,
                    current_prices=current_prices,
                    config=risk_config,
                )

                if not approved:
                    rejection_ledger.append(
                        {
                            "Date": dt.strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Sector": get_sector(ticker),
                            "Setup": setup_label,
                            "Price": round(close, 2),
                            "RejectionReason": reason,
                        }
                    )
                    continue

                gross_cost = entry_exec * qty
                total_cost = gross_cost * (1.0 + commission_pct)
                if total_cost > cash:
                    rejection_ledger.append(
                        {
                            "Date": dt.strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Sector": get_sector(ticker),
                            "Setup": setup_label,
                            "Price": round(close, 2),
                            "RejectionReason": "Insufficient Cash",
                        }
                    )
                    continue

                cash -= total_cost
                open_positions[ticker] = {
                    "entry_date": dt,
                    "entry": entry_exec,
                    "stop_loss": stop_loss,
                    "target": target,
                    "qty": qty,
                    "cost_basis": total_cost,
                    "sector": get_sector(ticker),
                    "setup": setup_label,
                    "last_price": close,
                }
                pending_signals.pop(ticker, None)
                current_prices[ticker] = close

        holdings_val = sum(
            current_prices.get(t, pos["last_price"]) * pos["qty"]
            for t, pos in open_positions.items()
        )
        equity_records.append(
            {
                "Date": dt,
                "Equity": cash + holdings_val,
                "NiftyClose": safe_float(n_row["Close"]),
            }
        )

    eq_df = pd.DataFrame(equity_records).set_index("Date")
    eq_df["Peak"] = eq_df["Equity"].cummax()
    eq_df["Drawdown%"] = (eq_df["Equity"] - eq_df["Peak"]) / eq_df["Peak"] * 100.0
    eq_df["NiftyNorm"] = (eq_df["NiftyClose"] / eq_df["NiftyClose"].iloc[0]) * initial_capital

    final_equity = safe_float(eq_df["Equity"].iloc[-1], default=initial_capital)
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0
    max_dd_pct = safe_float(eq_df["Drawdown%"].min())

    trades_df = pd.DataFrame(trade_log)
    rejections_df = pd.DataFrame(rejection_ledger)

    if not trades_df.empty:
        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100.0
        gross_profit = safe_float(wins["PnL"].sum())
        gross_loss = abs(safe_float(losses["PnL"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # CAPM Alpha & Beta (6% India risk-free rate)
    port_ret = eq_df["Equity"].pct_change().dropna()
    bench_ret = eq_df["NiftyClose"].pct_change().dropna()
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    aligned.columns = ["port", "bench"]

    if len(aligned) > 10 and aligned["bench"].var() > 0:
        beta = safe_float(aligned["port"].cov(aligned["bench"]) / aligned["bench"].var())
        rf_daily = 0.06 / 252.0
        alpha_daily = (aligned["port"].mean() - rf_daily) - beta * (aligned["bench"].mean() - rf_daily)
        alpha_annual_pct = safe_float(alpha_daily * 252.0 * 100.0)
    else:
        beta = 0.0
        alpha_annual_pct = 0.0

    metrics = {
        "InitialCapital": round(initial_capital, 2),
        "FinalEquity": round(final_equity, 2),
        "NetReturn%": round(net_return_pct, 2),
        "MaxDrawdown%": round(max_dd_pct, 2),
        "TotalTrades": int(len(trades_df)),
        "WinRate%": round(win_rate, 2),
        "ProfitFactor": round(profit_factor, 2),
        "Beta": round(beta, 2),
        "Alpha%": round(alpha_annual_pct, 2),
        "Rejections": int(len(rejections_df)),
    }

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "trade_log": trades_df,
        "rejection_ledger": rejections_df,
    }


def run_parameter_heatmap(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    sl_grid: List[float] = None,
    tp_grid: List[float] = None,
) -> pd.DataFrame:
    """Evaluate Net Return % across a grid of SL x TP ATR multipliers for stability validation."""
    sl_vals = sl_grid or [1.2, 1.5, 2.0, 2.5]
    tp_vals = tp_grid or [2.0, 2.5, 3.0, 4.0]

    matrix = pd.DataFrame(index=[f"SL {s:.1f}x" for s in sl_vals], columns=[f"TP {t:.1f}x" for t in tp_vals], dtype=float)
    for s in sl_vals:
        for t in tp_vals:
            res = run_portfolio_backtest(
                stock_data=stock_data,
                nifty_df=nifty_df,
                initial_capital=initial_capital,
                sl_mult=s,
                tp_mult=t,
            )
            ret = res["metrics"]["NetReturn%"] if "metrics" in res else 0.0
            matrix.loc[f"SL {s:.1f}x", f"TP {t:.1f}x"] = round(float(ret), 2)
    return matrix


def run_walk_forward_validation(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n_splits: int = 3,
) -> pd.DataFrame:
    """
    Walk-Forward Optimization (WFO):
      - For each rolling window, pick best (SL, TP) on In-Sample (IS) slice,
        then evaluate those exact parameters on unseen Out-of-Sample (OOS) slice.
    """
    if nifty_df is None or len(nifty_df) < 180:
        return pd.DataFrame()

    total_len = len(nifty_df)
    chunk = total_len // (n_splits + 1)
    if chunk < 65:
        return pd.DataFrame()

    candidate_pairs = [(1.2, 2.5), (1.5, 3.0), (2.0, 3.5), (2.0, 4.0)]
    rows: List[Dict[str, Any]] = []

    for split in range(n_splits):
        is_start = split * chunk
        is_end = is_start + chunk
        oos_end = min(total_len, is_end + chunk)
        if oos_end - is_end < 65:
            continue

        is_dates = nifty_df.index[is_start:is_end]
        oos_dates = nifty_df.index[is_end:oos_end]

        is_nifty = nifty_df.loc[is_dates]
        oos_nifty = nifty_df.loc[oos_dates]
        is_stocks = {t: df.loc[df.index.intersection(is_dates)] for t, df in stock_data.items()}
        oos_stocks = {t: df.loc[df.index.intersection(oos_dates)] for t, df in stock_data.items()}

        best_pair = (1.5, 3.0)
        best_is_ret = -999.0
        for sl_m, tp_m in candidate_pairs:
            res_is = run_portfolio_backtest(is_stocks, is_nifty, initial_capital=initial_capital, sl_mult=sl_m, tp_mult=tp_m)
            if "metrics" in res_is and res_is["metrics"]["NetReturn%"] > best_is_ret:
                best_is_ret = res_is["metrics"]["NetReturn%"]
                best_pair = (sl_m, tp_m)

        res_oos = run_portfolio_backtest(
            oos_stocks,
            oos_nifty,
            initial_capital=initial_capital,
            sl_mult=best_pair[0],
            tp_mult=best_pair[1],
        )
        oos_m = res_oos.get("metrics", {})

        rows.append(
            {
                "Fold": f"WFO Fold {split + 1}",
                "IS_Window": f"{is_dates[0].strftime('%Y-%m-%d')} → {is_dates[-1].strftime('%Y-%m-%d')}",
                "OOS_Window": f"{oos_dates[0].strftime('%Y-%m-%d')} → {oos_dates[-1].strftime('%Y-%m-%d')}",
                "Best_SL_TP": f"{best_pair[0]:.1f}x / {best_pair[1]:.1f}x",
                "IS_Return%": round(best_is_ret, 2),
                "OOS_Return%": oos_m.get("NetReturn%", 0.0),
                "OOS_WinRate%": oos_m.get("WinRate%", 0.0),
                "OOS_Trades": oos_m.get("TotalTrades", 0),
            }
        )

    return pd.DataFrame(rows)
