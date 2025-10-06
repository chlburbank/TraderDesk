# 🧠 TraderDesk

**TraderDesk** is a desktop trading research tool built in Python.
It lets you visualize and backtest simple trading strategies (like SMA crossovers) using **free market data** from Yahoo Finance today, and it is being expanded into a full live-trading workstation.

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
- ✅ **Today:** Research workflow with historical market data, signal generation, and performance analytics.
- 🚧 **In Development:** Modular execution engine, broker API integration, and real-time data ingestion.
- 🗓️ **Planned:** Automated risk management, monitoring dashboards, and compliance tooling for safe live deployment.

### 🧭 What You Can Do Right Now
- Fetch and clean market data
- Generate trading signals
- Backtest strategies with realistic assumptions
- Interpret performance metrics
