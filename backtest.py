"""
backtest.py — Multi-Asset Dual-Engine Portfolio Backtest, High Win-Rate Pullback Engine,
Alpha Growth Trailing Stop Engine, Parameter Heatmap, and Walk-Forward Optimization (WFO).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from data_feed import get_sector, safe_float
from risk import RiskConfig, enforce_min_rr, evaluate_trade_gate, update_chandelier_stop
from scanner import add_indicators


def _precompute_smc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Fast bar-by-bar Bullish FVG & Breakaway Gap flags for backtesting."""
    out = add_indicators(df)
    if out.empty:
        return out

    out = out[~out.index.duplicated(keep="last")].sort_index()
    highs = out["High"].to_numpy(dtype=float)
    lows = out["Low"].to_numpy(dtype=float)
    opens = out["Open"].to_numpy(dtype=float)
    closes = out["Close"].to_numpy(dtype=float)
    vols = out["Volume"].to_numpy(dtype=float)
    vol_ma = out["Vol_MA20"].fillna(out["Volume"]).to_numpy(dtype=float)
    atrs = out["ATR14"].bfill().fillna(0.0).to_numpy(dtype=float)
    n = len(out)

    fvg_support = np.zeros(n, dtype=bool)
    breakaway_gap = np.zeros(n, dtype=bool)
    active_fvgs: List[Tuple[float, float, float, int]] = []

    for i in range(2, n):
        atr_ref = atrs[i - 1] if atrs[i - 1] > 0 else (closes[i - 1] * 0.01)
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= (0.30 * atr_ref):
                b = highs[i - 2]
                t = lows[i]
                active_fvgs.append((b, t, (b + t) / 2.0, i))

        still_active = []
        for b, t, mid, born in active_fvgs:
            if i - born > 25 or lows[i] <= b:
                continue
            if b <= lows[i] <= t * 1.015 and closes[i] >= mid:
                fvg_support[i] = True
            still_active.append((b, t, mid, born))
        active_fvgs = still_active[-5:]

        prev_c = closes[i - 1]
        if prev_c > 0:
            gp = (opens[i] - prev_c) / prev_c
            vr = vols[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
            if gp >= 0.010 and vr >= 1.3 and closes[i] >= opens[i]:
                breakaway_gap[i] = True

    bear_fvg_overhead = np.zeros(n, dtype=bool)
    active_bear_fvgs: List[Tuple[float, float, int]] = []
    for i in range(2, n):
        atr_ref = atrs[i - 1] if atrs[i - 1] > 0 else (closes[i - 1] * 0.01)
        if highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= (0.30 * atr_ref):
            active_bear_fvgs.append((highs[i], lows[i - 2], i))
        still_b = []
        for b_bot, b_top, born in active_bear_fvgs:
            if i - born > 25 or highs[i] >= b_top:
                continue
            if (b_bot <= closes[i] <= b_top) or (closes[i] < b_bot <= closes[i] * 1.02):
                bear_fvg_overhead[i] = True
            still_b.append((b_bot, b_top, born))
        active_bear_fvgs = still_b[-5:]

    out["SMC_FVG_Hit"] = fvg_support
    out["SMC_Breakaway_Gap"] = breakaway_gap
    out["SMC_Bear_FVG_Overhead"] = bear_fvg_overhead
    out["PrevClose"] = out["Close"].shift(1)
    return out




def _compute_yearly_breakdown(eq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar Year-by-Year Return % for Maddy AI vs Benchmark."""
    if eq_df is None or eq_df.empty:
        return pd.DataFrame()
    df = eq_df[["Equity", "NiftyNorm"]].copy()
    df["Year"] = df.index.year
    rows = []
    prev_eq = float(df["Equity"].iloc[0])
    prev_bm = float(df["NiftyNorm"].iloc[0])
    for yr, grp in df.groupby("Year"):
        end_eq = float(grp["Equity"].iloc[-1])
        end_bm = float(grp["NiftyNorm"].iloc[-1])
        m_ret = ((end_eq - prev_eq) / prev_eq * 100.0) if prev_eq > 0 else 0.0
        b_ret = ((end_bm - prev_bm) / prev_bm * 100.0) if prev_bm > 0 else 0.0
        rows.append(
            {
                "Year": str(yr),
                "Maddy_AI%": round(m_ret, 2),
                "Benchmark%": round(b_ret, 2),
                "Outperformance%": round(m_ret - b_ret, 2),
            }
        )
        prev_eq = end_eq
        prev_bm = end_bm
    return pd.DataFrame(rows)


def _run_top5_momentum_rotation(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    slippage_pct: float = 0.0010,
    commission_pct: float = 0.0005,
    allow_fractional: bool = False,
    rebalance_days: int = 21,
    top_n: int = 5,
    emergency_stop_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    All-Weather Top-5 Momentum Rotation (Finalized Rules):
      - 1B Smart Gold Shield: When Benchmark < EMA200, rotate 100% into Gold ETF (GOLDBEES.NS / GLD)
        if Gold > EMA50, else 100% Cash Safe.
      - 2A Monthly Rebalance (21 trading days, Next-Day Open): Hold Top-5 risk-adjusted 63D momentum
        leaders above EMA50 at 20% equal weight.
      - 3 Emergency Crash Stop-Loss (-10%): Checked daily on every open position to prevent sudden crash damage.
    """
    nifty = add_indicators(nifty_df)
    nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()
    enriched: Dict[str, pd.DataFrame] = {}
    for t, df in stock_data.items():
        ind = add_indicators(df)
        if not ind.empty and len(ind) >= 70:
            ind = ind[~ind.index.duplicated(keep="last")].sort_index()
            ind["ROC126"] = ind["Close"].pct_change(126) * 100.0
            ind["ROC252"] = ind["Close"].pct_change(252) * 100.0
            ind["Vol60_Std"] = ind["Close"].pct_change().rolling(60, min_periods=20).std().replace(0, np.nan)
            enriched[t] = ind

    if not enriched or len(nifty) < 70:
        return {"error": "Need at least 70 daily bars for All-Weather Momentum Rotation."}

    # Identify defensive Gold asset in current watchlist (NSE GOLDBEES.NS or Global GLD/GC=F)
    gold_ticker = None
    for cand in ("GOLDBEES.NS", "GLD", "GC=F"):
        if cand in enriched:
            gold_ticker = cand
            break

    warmup = 130 if len(nifty) > 160 else 65
    common_dates = nifty.index[warmup:]
    cash = float(initial_capital)
    open_positions: Dict[str, dict] = {}
    pending_target_tickers: List[str] = []
    pending_is_gold_shield = False
    should_rebalance_next_open = False

    equity_records: List[Dict[str, Any]] = []
    trade_log: List[Dict[str, Any]] = []

    for bar_idx, dt in enumerate(common_dates):
        # 1. Daily Mid-Month Emergency Crash Stop-Loss (-10%) Check
        for t in list(open_positions.keys()):
            if dt not in enriched[t].index:
                continue
            r_day = enriched[t].loc[dt]
            if isinstance(r_day, pd.DataFrame):
                r_day = r_day.iloc[-1]
            low_px = safe_float(r_day["Low"])
            pos = open_positions[t]
            stop_px = pos["entry"] * (1.0 - emergency_stop_pct)
            if low_px > 0 and low_px <= stop_px:
                exec_px = stop_px * (1.0 - slippage_pct)
                open_positions.pop(t)
                net = exec_px * pos["qty"] * (1.0 - commission_pct)
                cash += net
                pnl = net - pos["cost_basis"]
                pnl_pct = (pnl / pos["cost_basis"] * 100.0) if pos["cost_basis"] > 0 else 0.0
                trade_log.append(
                    {
                        "Ticker": t,
                        "Sector": pos["sector"],
                        "Setup": pos["setup"],
                        "EntryDate": pos["entry_date"].strftime("%Y-%m-%d"),
                        "ExitDate": dt.strftime("%Y-%m-%d"),
                        "Qty": pos["qty"],
                        "Entry": round(pos["entry"], 2),
                        "Exit": round(exec_px, 2),
                        "PnL": round(pnl, 2),
                        "Return%": round(pnl_pct, 2),
                        "Reason": "Emergency -10% Stop",
                    }
                )

        # 2. Execute Monthly Rebalance at TODAY'S OPEN
        if should_rebalance_next_open:
            for t in list(open_positions.keys()):
                if t not in pending_target_tickers and dt in enriched[t].index:
                    row_o = enriched[t].loc[dt]
                    if isinstance(row_o, pd.DataFrame):
                        row_o = row_o.iloc[-1]
                    open_px = safe_float(row_o["Open"], default=open_positions[t]["last_price"])
                    exec_px = open_px * (1.0 - slippage_pct)
                    pos = open_positions.pop(t)
                    net = exec_px * pos["qty"] * (1.0 - commission_pct)
                    cash += net
                    pnl = net - pos["cost_basis"]
                    pnl_pct = (pnl / pos["cost_basis"] * 100.0) if pos["cost_basis"] > 0 else 0.0
                    trade_log.append(
                        {
                            "Ticker": t,
                            "Sector": pos["sector"],
                            "Setup": pos["setup"],
                            "EntryDate": pos["entry_date"].strftime("%Y-%m-%d"),
                            "ExitDate": dt.strftime("%Y-%m-%d"),
                            "Qty": pos["qty"],
                            "Entry": round(pos["entry"], 2),
                            "Exit": round(exec_px, 2),
                            "PnL": round(pnl, 2),
                            "Return%": round(pnl_pct, 2),
                            "Reason": "Monthly Rebalance",
                        }
                    )

            open_holdings = 0.0
            for t, pos in open_positions.items():
                if dt in enriched[t].index:
                    r_o = enriched[t].loc[dt]
                    if isinstance(r_o, pd.DataFrame):
                        r_o = r_o.iloc[-1]
                    open_holdings += safe_float(r_o["Open"], default=pos["last_price"]) * pos["qty"]
                else:
                    open_holdings += pos["last_price"] * pos["qty"]
            eq_open = cash + open_holdings

            target_alloc = (eq_open * 0.50) if pending_is_gold_shield else (eq_open * (1.0 / top_n) * 0.98)

            for t in pending_target_tickers:
                if t in open_positions or dt not in enriched[t].index:
                    continue
                r_o = enriched[t].loc[dt]
                if isinstance(r_o, pd.DataFrame):
                    r_o = r_o.iloc[-1]
                open_px = safe_float(r_o["Open"])
                close_px = safe_float(r_o["Close"])
                if open_px <= 0:
                    continue
                exec_px = open_px * (1.0 + slippage_pct)
                alloc = min(target_alloc, cash * 0.995)
                if alloc <= 0:
                    continue
                raw_qty = alloc / (exec_px * (1.0 + commission_pct))
                qty = round(raw_qty, 4) if allow_fractional else int(raw_qty)
                if qty <= 0:
                    continue
                cost = exec_px * qty * (1.0 + commission_pct)
                if cost > cash:
                    continue
                cash -= cost
                open_positions[t] = {
                    "entry_date": dt,
                    "entry": exec_px,
                    "qty": qty,
                    "cost_basis": cost,
                    "sector": get_sector(t),
                    "setup": "Gold Bear Shield" if pending_is_gold_shield else "Top-5 Momentum Leader",
                    "last_price": close_px,
                }
            should_rebalance_next_open = False

        # 3. End-of-Day Mark-to-Market
        holdings_val = 0.0
        for t, pos in open_positions.items():
            if dt in enriched[t].index:
                r_c = enriched[t].loc[dt]
                if isinstance(r_c, pd.DataFrame):
                    r_c = r_c.iloc[-1]
                pos["last_price"] = safe_float(r_c["Close"], default=pos["last_price"])
            holdings_val += pos["last_price"] * pos["qty"]

        n_row = nifty.loc[dt]
        if isinstance(n_row, pd.DataFrame):
            n_row = n_row.iloc[-1]
        bench_close = safe_float(n_row["Close"])
        bench_ema50 = safe_float(n_row["EMA50"])

        equity_records.append(
            {
                "Date": dt,
                "Equity": cash + holdings_val,
                "NiftyClose": bench_close,
            }
        )

        # 4. Every 21 trading days (Monthly): Fast EMA50 Shield + 6M/12M Blended Momentum
        if bar_idx % rebalance_days == 0:
            if bench_close < bench_ema50:
                # Fast EMA50 Bear Shield: 50% Gold ETF (if Gold > EMA50) + 50% Cash
                if gold_ticker and dt in enriched[gold_ticker].index:
                    rg = enriched[gold_ticker].loc[dt]
                    if isinstance(rg, pd.DataFrame):
                        rg = rg.iloc[-1]
                    if safe_float(rg["Close"]) > safe_float(rg["EMA50"]):
                        pending_target_tickers = [gold_ticker]
                        pending_is_gold_shield = True
                    else:
                        pending_target_tickers = []
                        pending_is_gold_shield = False
                else:
                    pending_target_tickers = []
                    pending_is_gold_shield = False
            else:
                ranked = []
                for t, df_t in enriched.items():
                    if dt not in df_t.index:
                        continue
                    r = df_t.loc[dt]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[-1]
                    c = safe_float(r["Close"])
                    e50 = safe_float(r["EMA50"])
                    roc6m = safe_float(r["ROC126"])
                    roc12m = safe_float(r["ROC252"], default=roc6m)
                    blended_mom = 0.5 * roc6m + 0.5 * roc12m
                    vol_s = safe_float(r["Vol60_Std"], default=0.02)
                    if c > e50 and blended_mom > 0 and vol_s > 0:
                        ranked.append((blended_mom / (vol_s * 100.0), t))
                ranked.sort(key=lambda x: x[0], reverse=True)
                pending_target_tickers = [t for _, t in ranked[:top_n]]
                pending_is_gold_shield = False
            should_rebalance_next_open = True

    eq_df = pd.DataFrame(equity_records).set_index("Date")
    eq_df["Peak"] = eq_df["Equity"].cummax()
    eq_df["Drawdown%"] = (eq_df["Equity"] - eq_df["Peak"]) / eq_df["Peak"] * 100.0
    eq_df["NiftyNorm"] = (eq_df["NiftyClose"] / eq_df["NiftyClose"].iloc[0]) * initial_capital

    final_equity = safe_float(eq_df["Equity"].iloc[-1], default=initial_capital)
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0
    n_years = max(len(eq_df) / 252.0, 0.25)
    cagr_pct = (((final_equity / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0 if final_equity > 0 else -100.0
    max_dd_pct = safe_float(eq_df["Drawdown%"].min())

    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100.0
        gp = safe_float(wins["PnL"].sum())
        gl = abs(safe_float(losses["PnL"].sum()))
        avg_w = safe_float(wins["PnL"].mean()) if len(wins) > 0 else 0.0
        avg_l = abs(safe_float(losses["PnL"].mean())) if len(losses) > 0 else 0.0
        payoff = (avg_w / avg_l) if avg_l > 0 else (99.0 if avg_w > 0 else 0.0)
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    else:
        win_rate = pf = avg_w = avg_l = payoff = 0.0

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
        beta = alpha_annual_pct = 0.0

    return {
        "metrics": {
            "InitialCapital": round(initial_capital, 2),
            "FinalEquity": round(final_equity, 2),
            "NetReturn%": round(net_return_pct, 2),
            "CAGR%": round(cagr_pct, 2),
            "MaxDrawdown%": round(max_dd_pct, 2),
            "TotalTrades": int(len(trades_df)),
            "WinRate%": round(win_rate, 2),
            "ProfitFactor": round(pf, 2),
            "PayoffRatio": round(payoff, 2),
            "AvgWin": round(avg_w, 2),
            "AvgLoss": round(avg_l, 2),
            "Beta": round(beta, 2),
            "Alpha%": round(alpha_annual_pct, 2),
            "Rejections": 0,
        },
        "equity_curve": eq_df,
        "trade_log": trades_df,
        "rejection_ledger": pd.DataFrame(),
        "yearly_breakdown": _compute_yearly_breakdown(eq_df),
    }

def run_portfolio_backtest(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    sl_mult: float = 1.8,
    tp_mult: float = 2.4,
    slippage_pct: float = 0.0010,
    commission_pct: float = 0.0005,
    stale_signal_bars: int = 3,
    strategy_mode: str = "Alpha Growth (2.2:1 R:R + Chandelier Trail)",
    risk_config: RiskConfig = RiskConfig(),
    max_atr_pct: float = 4.5,
    allow_fractional: bool = False,
) -> Dict[str, Any]:
    """
    Run multi-asset portfolio backtest across 3 white-box engine modes:
      1. 'High Win-Rate (Pullback + Partial Lock)':
           - Uptrend pullback entries (Close > EMA50, EMA20 > EMA50, RSI 38-60, green bounce candle)
           - Wider noise-proof SL (max(sl_mult, 1.8) * ATR)
           - Locks 50% partial profit at +1.0R and moves remaining 50% SL to Breakeven!
      2. 'Alpha Growth (Trail + RS Leaders)':
           - RS Leaders outperforming Nifty 50 + +1.5R Breakeven + 2x ATR Chandelier Trailing Stop
      3. 'Balanced (Fixed TP)':
           - Standard EMA + SMC fixed SL/TP baseline
    """
    if not stock_data or nifty_df is None or nifty_df.empty:
        return {"error": "Missing stock or Nifty 50 data."}

    mode_str = str(strategy_mode)
    if "Momentum Rotation" in mode_str:
        return _run_top5_momentum_rotation(
            stock_data=stock_data,
            nifty_df=nifty_df,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            allow_fractional=allow_fractional,
        )
    is_accuracy_mode = "High Accuracy" in mode_str
    is_winrate_mode = "High Win-Rate" in mode_str
    is_alpha_mode = "Alpha Growth" in mode_str

    nifty = add_indicators(nifty_df)
    nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()

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
    pending_signals: Dict[str, dict] = {}

    equity_records: List[Dict[str, Any]] = []
    trade_log: List[Dict[str, Any]] = []
    rejection_ledger: List[Dict[str, Any]] = []

    for bar_idx, dt in enumerate(common_dates):
        current_prices: Dict[str, float] = {}
        to_close: List[str] = []

        # 1. Manage Open Positions
        for ticker, pos in list(open_positions.items()):
            df_t = enriched[ticker]
            if dt not in df_t.index:
                current_prices[ticker] = pos["last_price"]
                continue

            row = df_t.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            low = safe_float(row["Low"])
            high = safe_float(row["High"])
            close = safe_float(row["Close"])
            atr_now = safe_float(row["ATR14"], default=pos["atr_entry"])
            current_prices[ticker] = close
            pos["last_price"] = close
            pos["highest_high"] = max(pos["highest_high"], high)

            # Rule 4: +1.0R 50% Partial Profit + Breakeven Lock (Alpha Growth & High Win-Rate)
            min_split_qty = 0.0002 if allow_fractional else 2
            if is_winrate_mode and (not pos["partial_taken"]) and pos["qty"] >= min_split_qty:
                partial_target = pos["entry"] + 1.0 * pos["init_risk"]
                if high >= partial_target:
                    sell_qty = round(pos["qty"] * 0.5, 4) if allow_fractional else (int(pos["qty"]) // 2)
                    rem_qty = round(pos["qty"] - sell_qty, 4) if allow_fractional else (int(pos["qty"]) - sell_qty)
                    exec_px = partial_target * (1.0 - slippage_pct)
                    gross = exec_px * sell_qty
                    net = gross * (1.0 - commission_pct)
                    cash += net

                    part_cost = pos["cost_basis"] * (sell_qty / pos["qty"])
                    part_pnl = net - part_cost
                    pos["realized_partial_pnl"] += part_pnl
                    pos["cost_basis"] -= part_cost
                    pos["qty"] = rem_qty
                    pos["partial_taken"] = True
                    # Shift SL to Breakeven + buffer so remaining shares cannot turn trade red
                    pos["stop_loss"] = max(pos["stop_loss"], pos["entry"] * (1.0 + slippage_pct + commission_pct))
                    pos["trailed"] = True

            exit_price = None
            exit_reason = None

            rsi2_now = safe_float(row["RSI2"], default=50.0)
            high5_now = safe_float(row["High5"], default=high)
            bars_held = bar_idx - pos["entry_bar_idx"]

            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"] * (1.0 - slippage_pct)
                if pos["partial_taken"]:
                    exit_reason = "Partial + Breakeven"
                elif pos["trailed"]:
                    exit_reason = "Trailing Stop"
                else:
                    exit_reason = "Stop-Loss"
            elif is_accuracy_mode and bars_held >= 1 and close >= (pos["entry"] + 1.0 * pos["init_risk"]) and (close >= high5_now or rsi2_now >= 75.0):
                exit_price = close * (1.0 - slippage_pct)
                exit_reason = "Snapback Profit"
            elif is_accuracy_mode and bars_held >= 12:
                exit_price = close * (1.0 - slippage_pct)
                exit_reason = "Time Exit (12D)"
            elif is_winrate_mode and high >= pos["target"] * 1.8:
                exit_price = (pos["target"] * 1.8) * (1.0 - slippage_pct)
                exit_reason = "Runner Target"
            elif (not is_alpha_mode) and (not is_winrate_mode) and high >= pos["target"]:
                exit_price = pos["target"] * (1.0 - slippage_pct)
                exit_reason = "Target Hit"
            elif is_alpha_mode and high >= pos["target"] * 1.6:
                exit_price = (pos["target"] * 1.6) * (1.0 - slippage_pct)
                exit_reason = "Extended Target"

            if exit_price is not None:
                gross = exit_price * pos["qty"]
                net = gross * (1.0 - commission_pct)
                cash += net

                total_pnl = (net - pos["cost_basis"]) + pos["realized_partial_pnl"]
                orig_cost = pos["orig_cost_basis"]
                pnl_pct = (total_pnl / orig_cost * 100.0) if orig_cost > 0 else 0.0

                trade_log.append(
                    {
                        "Ticker": ticker,
                        "Sector": pos["sector"],
                        "Setup": pos["setup"],
                        "EntryDate": pos["entry_date"].strftime("%Y-%m-%d"),
                        "ExitDate": dt.strftime("%Y-%m-%d"),
                        "Qty": pos["orig_qty"],
                        "Entry": round(pos["entry"], 2),
                        "Exit": round(exit_price, 2),
                        "PnL": round(total_pnl, 2),
                        "Return%": round(pnl_pct, 2),
                        "Reason": exit_reason,
                    }
                )
                to_close.append(ticker)
            elif is_alpha_mode or (is_winrate_mode and pos["partial_taken"]):
                new_sl, stage = update_chandelier_stop(
                    entry_price=pos["entry"],
                    current_sl=pos["stop_loss"],
                    highest_high=pos["highest_high"],
                    atr_now=atr_now,
                    init_risk=pos["init_risk"],
                    config=risk_config,
                    buffer_pct=slippage_pct + commission_pct,
                )
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    pos["trailed"] = True

        for ticker in to_close:
            open_positions.pop(ticker, None)

        holdings_val = sum(
            current_prices.get(t, pos["last_price"]) * pos["qty"]
            for t, pos in open_positions.items()
        )
        equity = cash + holdings_val
        peak_equity = max(peak_equity, equity)

        # 2. Execute Yesterday's Pending Signals at TODAY'S OPEN (+ slippage) — 100% Realistic!
        returns_slice = returns_df.loc[:dt]
        sorted_pending = sorted(
            list(pending_signals.items()),
            key=lambda kv: kv[1].get("rs_val", 0.0),
            reverse=True,
        )
        for ticker, sig in sorted_pending:
            if bar_idx - sig["bar_idx"] > stale_signal_bars:
                pending_signals.pop(ticker, None)
                continue
            if ticker in open_positions:
                pending_signals.pop(ticker, None)
                continue
            df_t = enriched[ticker]
            if dt not in df_t.index:
                continue

            row_open = df_t.loc[dt]
            if isinstance(row_open, pd.DataFrame):
                row_open = row_open.iloc[-1]

            open_px = safe_float(row_open["Open"])
            close_px = safe_float(row_open["Close"])
            if open_px <= 0:
                pending_signals.pop(ticker, None)
                continue

            atr14 = sig["atr14"]
            setup_label = sig["setup"]
            fvg_hit = sig["fvg_hit"]
            gap_hit = sig["gap_hit"]

            entry_exec = open_px * (1.0 + slippage_pct)
            if is_accuracy_mode:
                eff_sl_mult = min(sl_mult, 1.6)
                eff_tp_mult = max(tp_mult, 3.0)
            elif is_winrate_mode:
                eff_sl_mult = max(sl_mult, 1.8)
                eff_tp_mult = min(tp_mult, 2.4)
            else:
                eff_sl_mult = sl_mult
                eff_tp_mult = tp_mult

            stop_loss = max(0.05, entry_exec - eff_sl_mult * atr14)
            raw_target = entry_exec + eff_tp_mult * atr14
            target = enforce_min_rr(entry_exec, stop_loss, raw_target, min_rr=risk_config.min_rr_ratio)

            eff_cfg = RiskConfig(
                risk_per_trade_pct=0.015 if ((is_alpha_mode or is_winrate_mode) and (fvg_hit or gap_hit)) else risk_config.risk_per_trade_pct,
                max_stock_alloc_pct=risk_config.max_stock_alloc_pct,
                max_sector_alloc_pct=risk_config.max_sector_alloc_pct,
                max_portfolio_heat_pct=risk_config.max_portfolio_heat_pct,
                max_correlation=risk_config.max_correlation,
                kill_switch_dd_pct=risk_config.kill_switch_dd_pct,
            )

            approved, qty, reason = evaluate_trade_gate(
                ticker=ticker,
                entry_price=entry_exec,
                stop_loss=stop_loss,
                equity=equity,
                peak_equity=peak_equity,
                open_positions=open_positions,
                returns_df=returns_slice,
                current_prices=current_prices,
                config=eff_cfg,
                allow_fractional=allow_fractional,
            )

            if not approved:
                rejection_ledger.append(
                    {
                        "Date": dt.strftime("%Y-%m-%d"),
                        "Ticker": ticker,
                        "Sector": get_sector(ticker),
                        "Setup": setup_label,
                        "Price": round(open_px, 2),
                        "RejectionReason": reason,
                    }
                )
                pending_signals.pop(ticker, None)
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
                        "Price": round(open_px, 2),
                        "RejectionReason": "Insufficient Cash",
                    }
                )
                pending_signals.pop(ticker, None)
                continue

            cash -= total_cost
            open_positions[ticker] = {
                "entry_date": dt,
                "entry_bar_idx": bar_idx,
                "entry": entry_exec,
                "stop_loss": stop_loss,
                "init_risk": max(0.01, entry_exec - stop_loss),
                "atr_entry": atr14,
                "highest_high": max(open_px, close_px),
                "trailed": False,
                "partial_taken": False,
                "realized_partial_pnl": 0.0,
                "target": target,
                "qty": qty,
                "orig_qty": qty,
                "cost_basis": total_cost,
                "orig_cost_basis": total_cost,
                "sector": get_sector(ticker),
                "setup": setup_label,
                "last_price": close_px,
            }
            pending_signals.pop(ticker, None)
            current_prices[ticker] = close_px

        # 3. Macro Regime Check on Benchmark (End-of-Day)
        n_row = nifty.loc[dt]
        if isinstance(n_row, pd.DataFrame):
            n_row = n_row.iloc[-1]
        nifty_roc20 = safe_float(n_row["ROC20"])
        nifty_bullish = (
            safe_float(n_row["Close"]) > safe_float(n_row["EMA20"])
            and safe_float(n_row["EMA20"]) >= safe_float(n_row["EMA50"])
        )

        # 4. End-of-Day Signal Scan -> Queued for NEXT-DAY OPEN Execution
        if nifty_bullish:
            # Rule 3: Compute cross-sectional Sector Avg RS% on day dt
            sec_rs_buckets: Dict[str, List[float]] = {}
            for t_sec, df_sec in enriched.items():
                if dt in df_sec.index:
                    r_sec = df_sec.loc[dt]
                    if isinstance(r_sec, pd.DataFrame):
                        r_sec = r_sec.iloc[-1]
                    sec_name = get_sector(t_sec)
                    sec_rs_buckets.setdefault(sec_name, []).append(safe_float(r_sec["ROC20"]) - nifty_roc20)
            strong_sectors = {
                sec for sec, vals in sec_rs_buckets.items() if vals and (sum(vals) / len(vals)) > 0.0
            }

            for ticker, df_t in enriched.items():
                if ticker in open_positions or ticker in pending_signals or dt not in df_t.index:
                    continue
                row = df_t.loc[dt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                open_p = safe_float(row["Open"])
                close = safe_float(row["Close"])
                low = safe_float(row["Low"])
                prev_c = safe_float(row["PrevClose"], default=close)
                ema20 = safe_float(row["EMA20"])
                ema50 = safe_float(row["EMA50"])
                atr14 = safe_float(row["ATR14"])
                atr_pct = safe_float(row["ATR_Pct"])
                rsi14 = safe_float(row["RSI14"], default=50.0)
                roc20 = safe_float(row["ROC20"])
                rs_vs_nifty = roc20 - nifty_roc20
                fvg_hit = bool(row["SMC_FVG_Hit"])
                gap_hit = bool(row["SMC_Breakaway_Gap"])

                dist_pct = ((close - ema20) / ema20 * 100.0) if ema20 > 0 else 99.0
                smc_boost = fvg_hit or gap_hit

                if is_accuracy_mode:
                    ema200 = safe_float(row["EMA200"], default=ema50)
                    rsi2 = safe_float(row["RSI2"], default=50.0)
                    long_term_up = close > ema200 and ema50 >= ema200 * 0.99 and atr14 > 0
                    oversold_dip = (rsi2 <= 22.0) or (fvg_hit and rsi2 <= 35.0)
                    if not (long_term_up and oversold_dip):
                        continue
                elif is_winrate_mode:
                    uptrend = close > ema50 and ema20 > ema50 and rs_vs_nifty > 0.0 and atr14 > 0 and 0.6 <= atr_pct <= max_atr_pct
                    near_value = (-1.5 <= dist_pct <= 3.5) or (low <= ema20 * 1.015 and close >= ema20) or smc_boost
                    green_bounce = close > open_p and close > prev_c
                    healthy_rsi = 42.0 <= rsi14 <= 68.0
                    if not (uptrend and near_value and green_bounce and healthy_rsi):
                        continue
                elif is_alpha_mode:
                    adx14 = safe_float(row["ADX14"], default=0.0)
                    ema_slope = safe_float(row["EMA20_Slope"], default=0.0)
                    candle_str = safe_float(row["Candle_Strength"], default=0.5)
                    vol_ratio = safe_float(row["Vol_Ratio"], default=1.0)
                    weekly_bull = bool(row["Weekly_Bullish"])
                    bear_fvg_blk = bool(row["SMC_Bear_FVG_Overhead"])
                    sector_strong = get_sector(ticker) in strong_sectors

                    # Clean 4-Rule Alpha Growth (avoids over-filtering idle cash)
                    trend_ok = (
                        weekly_bull
                        and close > ema20
                        and ema20 > ema50
                        and adx14 >= 18.0
                        and (not bear_fvg_blk)
                        and atr14 > 0
                        and 0.8 <= atr_pct <= max_atr_pct
                    )
                    if not (trend_ok and rs_vs_nifty > 0.0 and 45.0 <= rsi14 <= 72.0 and (dist_pct <= 4.5 or smc_boost)):
                        continue
                else:
                    trend_ok = close > ema20 and ema20 > ema50 and atr14 > 0 and 0.8 <= atr_pct <= max_atr_pct
                    if not (trend_ok and (dist_pct <= 3.0 or smc_boost)):
                        continue

                if is_accuracy_mode:
                    setup_label = "EMA200 + FVG Snapback" if fvg_hit else "EMA200 + RSI2 Dip"
                elif is_winrate_mode:
                    setup_label = "Pullback + FVG 50%" if fvg_hit else "EMA20 Pullback Bounce"
                elif is_alpha_mode:
                    setup_label = "A+ FVG/Gap Leader" if (fvg_hit or gap_hit) else "RS Leader + EMA"
                else:
                    setup_label = "EMA + SMC" if (fvg_hit or gap_hit) else "EMA Trend"

                pending_signals[ticker] = {
                    "bar_idx": bar_idx,
                    "setup": setup_label,
                    "rs_val": rs_vs_nifty,
                    "atr14": atr14,
                    "fvg_hit": fvg_hit,
                    "gap_hit": gap_hit,
                }

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
    n_years = max(len(eq_df) / 252.0, 0.25)
    cagr_pct = (((final_equity / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0 if final_equity > 0 else -100.0
    max_dd_pct = safe_float(eq_df["Drawdown%"].min())

    trades_df = pd.DataFrame(trade_log)
    rejections_df = pd.DataFrame(rejection_ledger)

    if not trades_df.empty:
        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100.0
        gross_profit = safe_float(wins["PnL"].sum())
        gross_loss = abs(safe_float(losses["PnL"].sum()))
        avg_win = safe_float(wins["PnL"].mean()) if len(wins) > 0 else 0.0
        avg_loss = abs(safe_float(losses["PnL"].mean())) if len(losses) > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        payoff_ratio = 0.0

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
        "CAGR%": round(cagr_pct, 2),
        "MaxDrawdown%": round(max_dd_pct, 2),
        "TotalTrades": int(len(trades_df)),
        "WinRate%": round(win_rate, 2),
        "ProfitFactor": round(profit_factor, 2),
        "PayoffRatio": round(payoff_ratio, 2),
        "AvgWin": round(avg_win, 2),
        "AvgLoss": round(avg_loss, 2),
        "Beta": round(beta, 2),
        "Alpha%": round(alpha_annual_pct, 2),
        "Rejections": int(len(rejections_df)),
    }

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "trade_log": trades_df,
        "rejection_ledger": rejections_df,
        "yearly_breakdown": _compute_yearly_breakdown(eq_df),
    }


def run_parameter_heatmap(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    strategy_mode: str = "High Accuracy (EMA200 + RSI2 Snapback)",
    sl_grid: List[float] = None,
    tp_grid: List[float] = None,
    max_atr_pct: float = 4.5,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """Evaluate Net Return % across a grid of SL x TP ATR multipliers."""
    sl_vals = sl_grid or [1.5, 1.8, 2.0, 2.5]
    tp_vals = tp_grid or [2.0, 2.4, 3.0, 4.0]

    matrix = pd.DataFrame(index=[f"SL {s:.1f}x" for s in sl_vals], columns=[f"TP {t:.1f}x" for t in tp_vals], dtype=float)
    for s in sl_vals:
        for t in tp_vals:
            res = run_portfolio_backtest(
                stock_data=stock_data,
                nifty_df=nifty_df,
                initial_capital=initial_capital,
                sl_mult=s,
                tp_mult=t,
                strategy_mode=strategy_mode,
                max_atr_pct=max_atr_pct,
                allow_fractional=allow_fractional,
            )
            ret = res["metrics"]["NetReturn%"] if "metrics" in res else 0.0
            matrix.loc[f"SL {s:.1f}x", f"TP {t:.1f}x"] = round(float(ret), 2)
    return matrix


def run_walk_forward_validation(
    stock_data: Dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    strategy_mode: str = "High Accuracy (EMA200 + RSI2 Snapback)",
    n_splits: int = 3,
    max_atr_pct: float = 4.5,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """Rolling Walk-Forward Optimization (In-Sample -> Unseen Out-of-Sample)."""
    if nifty_df is None or len(nifty_df) < 180:
        return pd.DataFrame()

    total_len = len(nifty_df)
    chunk = total_len // (n_splits + 1)
    if chunk < 65:
        return pd.DataFrame()

    candidate_pairs = [(1.5, 2.4), (1.8, 2.4), (2.0, 3.0), (2.0, 4.0)]
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

        best_pair = (1.8, 2.4)
        best_is_ret = -999.0
        for sl_m, tp_m in candidate_pairs:
            res_is = run_portfolio_backtest(
                is_stocks, is_nifty, initial_capital=initial_capital,
                sl_mult=sl_m, tp_mult=tp_m, strategy_mode=strategy_mode, max_atr_pct=max_atr_pct, allow_fractional=allow_fractional,
            )
            if "metrics" in res_is and res_is["metrics"]["NetReturn%"] > best_is_ret:
                best_is_ret = res_is["metrics"]["NetReturn%"]
                best_pair = (sl_m, tp_m)

        res_oos = run_portfolio_backtest(
            oos_stocks, oos_nifty, initial_capital=initial_capital,
            sl_mult=best_pair[0], tp_mult=best_pair[1], strategy_mode=strategy_mode, max_atr_pct=max_atr_pct, allow_fractional=allow_fractional,
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
