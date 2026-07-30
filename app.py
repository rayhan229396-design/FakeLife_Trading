import os
import sys
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Ensure core directory is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_feed import DataFeed
from core.predict import SignalPredictor
from core.train_model import ModelTrainer
from core.backtest import Backtester
from core.walk_forward import WalkForwardValidator
from config import Config

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.VERSION,
    description="Institutional-Grade Quant Trading Engine with SMC, ML Calibration, XAI & Walk-Forward Validation."
)

# Enable CORS for Live Dashboard Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SignalPredictor()

@app.get("/")
def home():
    return {
        "status": "Online",
        "engine": Config.APP_NAME,
        "version": Config.VERSION,
        "docs": "/docs"
    }

@app.get("/api/predict")
def get_prediction(symbol: str = Query("EURUSD", description="Currency pair or Asset"), 
                   timeframe: str = Query("5m", description="Candle Timeframe (1m, 5m, 15m)")):
    """
    Get Real-time Trade Signal with Probability Calibration & Explainable AI Reason.
    """
    result = predictor.predict_signal(symbol, timeframe)
    return result

@app.get("/api/backtest")
def run_backtest_endpoint(symbol: str = "EURUSD", timeframe: str = "5m", balance: float = 1000.0):
    """
    Runs historical backtest for a symbol and returns performance metrics.
    """
    df = DataFeed.fetch_data(symbol, timeframe)
    if df is None:
        return {"error": f"Unable to fetch data for {symbol}"}
    
    results = Backtester.run_backtest(df, initial_balance=balance)
    return results

@app.get("/api/walk-forward")
def run_walk_forward_endpoint(symbol: str = "EURUSD", timeframe: str = "5m"):
    """
    Runs Walk-Forward Validation to prevent overfitting.
    """
    df = DataFeed.fetch_data(symbol, timeframe)
    if df is None:
        return {"error": f"Unable to fetch data for {symbol}"}
    
    results = WalkForwardValidator.run_validation(df)
    return results

@app.post("/api/train")
def trigger_training(background_tasks: BackgroundTasks, symbol: str = "EURUSD", timeframe: str = "5m"):
    """
    Triggers model training in background to avoid Render timeout.
    """
    background_tasks.add_task(ModelTrainer.train_and_save, symbol, timeframe)
    return {"status": "Model training initiated in background."}
