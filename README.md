# 🚀 AI-Smart-Trading-Engine

An Institutional-Grade Quantitative Trading & Machine Learning Framework powered by Smart Money Concepts (SMC), Probability Calibration, Explainable AI (XAI), and Walk-Forward Validation.

---

## 🌟 Key Features

* **Smart Money Concepts (SMC):** Liquidity Sweeps, Order Blocks (OB), Break of Structure (BOS), and Fair Value Gaps (FVG).
* **Multi-Timeframe Analysis (MTF):** Aligns low-timeframe execution with higher-timeframe market bias.
* **Probability Calibration:** Implements Isotonic/Platt Calibration to provide true probabilistic confidence outputs.
* **Explainable AI (XAI):** Natural language breakdowns explaining *why* a CALL/PUT/WAIT signal was issued.
* **Walk-Forward Validation:** Prevents backtest overfitting using rolling window out-of-sample data splits.
* **Production-Ready FastAPI:** REST API Endpoints ready for Deployment on Render / Docker.

---

## 📁 Architecture Overview

```text
AI-Smart-Trading-Engine/
├── app.py                  # FastAPI Backend Server
├── config.py               # Global Project Configuration
├── requirements.txt        # Dependencies Optimized for Cloud Deployment
├── README.md               # Documentation
│
├── core/
│   ├── data_feed.py        # Yahoo Finance Data Ingestion
│   ├── market_structure.py # BOS & CHoCH Analytics
│   ├── liquidity.py        # Equal Highs/Lows & Sweeps
│   ├── order_block.py       # Bullish & Bearish OB Engine
│   ├── fair_value_gap.py    # 3-Candle FVG Imbalance Engine
│   ├── feature_engine.py    # Feature Engineering Aggregator
│   ├── calibration.py      # Probability Calibration
│   ├── prepare_dataset.py  # Dataset Generator
│   ├── train_model.py      # ML Model Training Pipeline
│   ├── predict.py          # Real-time Inference Engine (XAI Enabled)
│   ├── backtest.py         # Vectorized Backtesting System
│   └── walk_forward.py     # Rolling Walk-Forward Validator
│
└── models/                 # Saved `.pkl` Models
