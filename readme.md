# 🧠 TraderDesk

**TraderDesk** is a desktop trading research tool built in Python.
It lets you visualize and backtest simple trading strategies (like SMA crossovers) using **free market data** from Yahoo Finance today, and it is being expanded into an **AI-assisted live-trading workstation**.

---

## 📸 Preview
Example: SPY with 50/200 SMA crossover  
![TraderDesk Screenshot](docs/screenshot.png)

---

## 🚀 Features
- 🟢 Load and visualize historical stock or ETF data (via `yfinance`)
- 📈 Plot price with customizable moving averages
- ⚙️ Backtest basic crossover strategies
- 🧮 Display key performance metrics (CAGR, Sharpe, Sortino, Max Drawdown)
- 🤖 Prototype AI predictor for next-bar returns, confidence, and position sizing guidance
- 🪟 Simple GUI built with `PySide6`

---

## 🧩 Tech Stack
- **Python 3.11+**
- `PySide6` – GUI framework  
- `yfinance` – Free market data  
- `pandas`, `matplotlib` – Data handling and plotting  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/chlburbank/TraderDesk.git
cd TraderDesk
```

### 2 Clone the repository
```bash
Create a virtual environment
python -m venv ta_env
```

3️⃣ Activate it
```bash
🟦 On Windows PowerShell:
.\ta_env\Scripts\Activate.ps1
```

🟩 On macOS/Linux:
```bash
source ta_env/bin/activate
```

4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

5️⃣ Run the app
```bash
python traderdesk.py
```

> ⚠️ **Work in Progress:** TraderDesk is actively evolving toward a live trading platform. The current release focuses on research and backtesting while the team builds out the broker connectivity, execution, and risk controls required for production use.

### 🛣️ Live Trading Roadmap Highlights
- ✅ **Today:** Research workflow with historical market data, signal generation, performance analytics, and an AI predictor powering the live engine prototype.
- 🚧 **In Development:** Modular execution engine hardening, broker API integration, and real-time data ingestion.
- 🗓️ **Planned:** Automated risk management, monitoring dashboards, compliance tooling, and multi-asset portfolio coordination for safe live deployment.

### 🧭 What You Can Do Right Now
- Fetch and clean market data
- Generate trading signals
- Backtest strategies with realistic assumptions
- Interpret performance metrics
- Experiment with the new AI predictor and paper-broker live trading engine scaffolding

---

## 🧠 Getting Started with AI-Assisted Live Trading

The `traderdesk.ai` module introduces a lightweight ridge-regression predictor that learns from historical closing prices and estimates the next-bar return with an associated confidence score. The live trading prototype wires this predictor into a modular engine that can be pointed at a real broker once credentials and compliance checks are ready.

```python
from traderdesk import (
    AIPredictor,
    LiveTradingConfig,
    LiveTradingEngine,
    PaperBroker,
    YahooMarketDataProvider,
)

config = LiveTradingConfig(ticker="SPY", max_trade_notional=1000)
predictor = AIPredictor()
data_provider = YahooMarketDataProvider()
broker = PaperBroker()

engine = LiveTradingEngine(config, predictor, data_provider, broker)
decision = engine.evaluate_and_execute()
print(decision)
```

When the AI signal meets the built-in thresholds, the engine allocates up to the specified
`max_trade_notional` based on a blend of expected return strength and confidence, so you receive a
ready-to-execute share count without tuning expert parameters.

### 🖥️ One-Click AI Trades in the UI

Inside the Qt application you only provide a ticker and an **Investment Budget ($)**. Pressing
**AI Evaluate & Trade** runs the same engine as above, logs the forecast, and (when conditions are
met) sends a paper-trade order sized automatically by the AI model. If the budget is too small to
buy at least one share, the app will prompt you to raise it or pick a lower-priced asset—no expert
settings required.

> ⚠️ **Important:** The live trading components currently target a paper broker and do not handle order routing, authentication, or regulatory checks. They are meant for experimentation while the production integrations are being built.
