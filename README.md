# Maddy AI — Complete Personal Quant Terminal (5-Tab Edition)

100% phone-friendly modular Python + Streamlit quant terminal for NSE / Nifty 50.

## Modular Architecture
- `data_feed.py` — Safe `yfinance` loader (`float(np.squeeze())` protection), sector map, offline demo mode, and `BrokerBridge` (Angel One SmartAPI / Upstox / Dhan).
- `scanner.py` — `EMA20`, `EMA50`, `ATR14`, Nifty 50 Macro Filter, 3-Candle Bullish/Bearish FVG detector (50% midpoint & mitigation), Breakaway/Common Market Gap detector, and 0–100 Confluence Scoring.
- `risk.py` — Institutional Risk Gate (1% risk, 20% stock cap, 30% sector cap, 5% heat, 30D correlation shield >0.80, 25% kill-switch) + 1,000-run Block-Bootstrap Monte Carlo simulator.
- `backtest.py` — Multi-asset portfolio engine (0.10% slippage, 0.05% brokerage, stale signal expiry, CAPM Alpha/Beta, Rejection Ledger), Parameter Stability Heatmap, and Walk-Forward Optimization (WFO).
- `alerts.py` — Telegram Bot notification module (`TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID`).
- `app.py` — Complete 5-Tab Streamlit + Plotly dashboard with SQLite paper trading & CSV backup.

## Phone-Only Free Deployment (GitHub + Streamlit Community Cloud)
1. Unzip `maddy_ai_complete.zip` in your phone's File Manager.
2. Open GitHub in Chrome/Safari (enable **Desktop site** if needed), create a repo (e.g. `maddy-ai`), tap **Add file → Upload files**, select all files inside the folder, and **Commit changes**.
3. Open `https://share.streamlit.io`, sign in with GitHub, tap **Create app**, choose your `maddy-ai` repo and `app.py`, and tap **Deploy**.
4. *(Optional)* In Streamlit Cloud **App Settings → Secrets**, add:
```toml
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
BROKER_API_KEY = "optional_broker_key"
BROKER_ACCESS_TOKEN = "optional_broker_token"
```
