import os
import joblib
import pandas as pd
import numpy as np
from core.data_feed import DataFeed
from core.feature_engine import FeatureEngine
from config import Config

class SignalPredictor:
    def __init__(self):
        self.model = None
        self.calibrator = None
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(Config.MODEL_PATH):
                self.model = joblib.load(Config.MODEL_PATH)
            if os.path.exists(Config.CALIBRATOR_PATH):
                self.calibrator = joblib.load(Config.CALIBRATOR_PATH)
        except Exception as e:
            print(f"Model loading notice: {e}")

    def predict_signal(self, symbol: str = "EURUSD", timeframe: str = "5m"):
        # Fetch Data
        df = DataFeed.fetch_data(symbol, timeframe)
        
        if df is None or len(df) < 30:
            return {
                "status": "error",
                "message": f"Could not fetch enough candle data for {symbol} ({timeframe}). Yahoo Finance might be rate-limiting.",
                "suggestion": "Try another symbol like 'BTC-USD' or timeframe '1d'"
            }

        # Generate SMC & Technical Features
        df_featured = FeatureEngine.generate_features(df)
        latest_row = df_featured.iloc[-1]

        # Basic SMC Rules as Baseline Strategy
        bullish_signal = (latest_row.get('bullish_ob', 0) == 1) or (latest_row.get('rsi', 50) < 35)
        bearish_signal = (latest_row.get('bearish_ob', 0) == 1) or (latest_row.get('rsi', 50) > 65)

        # ML Model Inference if available
        if self.model is not None:
            feature_cols = [c for c in df_featured.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
            X_latest = df_featured[feature_cols].iloc[[-1]]
            
            raw_prob = self.model.predict_proba(X_latest)[0][1]
            if self.calibrator:
                prob_call = float(self.calibrator.predict([raw_prob])[0])
            else:
                prob_call = float(raw_prob)
            prob_put = 1.0 - prob_call
        else:
            # Fallback Logic when model is not trained yet
            prob_call = 0.75 if bullish_signal else (0.25 if bearish_signal else 0.50)
            prob_put = 1.0 - prob_call

        # Signal Direction
        if prob_call > 0.60:
            direction = "CALL"
            reason = f"Bullish bias detected. RSI at {latest_row.get('rsi', 50):.1f} with SMC Support Alignment."
        elif prob_put > 0.60:
            direction = "PUT"
            reason = f"Bearish bias detected. RSI at {latest_row.get('rsi', 50):.1f} with SMC Resistance Alignment."
        else:
            direction = "WAIT"
            reason = f"Market in Neutral Zone. RSI at {latest_row.get('rsi', 50):.1f}. Confidence below threshold."

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "direction": direction,
            "confidence": round(max(prob_call, prob_put) * 100, 2),
            "prob_call": round(prob_call * 100, 2),
            "prob_put": round(prob_put * 100, 2),
            "explanation": reason,
            "last_price": float(latest_row['Close'])
        }
