# 📈 Trading Bots Collection (For Academic Use Only)

Welcome to the Trading Bots Repository!  
This collection of algorithmic trading strategies was developed **purely for academic and educational purposes**. The goal of this project was to explore and understand various algorithmic and machine learning-based approaches to financial market modeling, rather than to provide production-ready or profitable trading systems.

> ⚠️ **Disclaimer**  
> None of the strategies included in this repository have shown reliable profitability during backtesting or paper trading.  
> **These bots should not be used in live trading environments or with real money.**  
> Use at your own risk, and always perform your own due diligence.

---

## 🧠 Purpose

The repository serves as a learning tool to:

- Explore classical and modern trading strategies.
- Understand time series modeling and forecasting techniques.
- Gain practical experience with machine learning applied to financial data.
- Learn about strategy evaluation, backtesting, and tuning.

---

## ⚙️ Included Strategies (see Branches)

### 1. **Buy-the-Dip**
A naive mean-reversion strategy that attempts to buy assets after significant short-term losses.  
**Insight**: Simple idea, but performance is highly sensitive to noise and lacks predictive power.

---

### 2. **Hilbert Transform**
Applies signal-processing techniques to price data to extract cyclical information.  
**Insight**: Offers theoretical elegance, but in practice, market data is too non-stationary and erratic for reliable application.

---

### 3. **Ichimoku Cloud**
A technical indicator combining trend-following and momentum.  
**Insight**: Provides visually rich signals, yet failed to deliver consistent edge without extensive filtering or confirmation.

---

### 4. **Probabilistic Models**
Forecasting approaches combining:

- **ARIMA / SARIMA**: For linear time-series trends and seasonality.
- **GARCH / EGARCH**: For modeling volatility clusters.

**Insight**: Interesting for modeling returns and volatility; however, forecasts remained too uncertain for systematic trading.

---

### 5. **Machine Learning Projects**
Two experimental ML-based approaches were included:

- **Supervised classification models** (e.g., Random Forests, SVMs).
- **Neural networks for price movement prediction**.

**Insight**: These were primarily exercises to understand feature engineering, overfitting, and model evaluation in finance.  
**They were not designed with production use in mind.**

---

## 🔧 Contribution and Reworking

We welcome contributions, experiments, and re-tunings. While the strategies as-is did not yield satisfactory results, with:

- Better **hyperparameter tuning**,
- More **robust data pipelines**,
- Adaptive **risk management**, or
- Incorporation of **alternative data**,

there remains potential for discovery.

Feel free to open issues, suggest improvements, or fork the project to build upon these academic foundations.

---

## 🧾 License

This repository is provided **as-is**, without warranty or guarantee of profitability.  
Use and modification are permitted under the MIT License.

---

## 📬 Contact

For questions, discussions, or suggestions, feel free to open an issue or reach out via the discussions tab.

---

> _"In trading, the illusion of simplicity often hides a jungle of uncertainty. May this repo help you explore it wisely."_ 🌱
