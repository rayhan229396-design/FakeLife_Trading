import os
import joblib
import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
from data_feed import DataFeed
from config import Config

class SignalPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads trained model if exists."""
        if os.path.exists(Config.MODEL_PATH):
            try:
                self.model = joblib.load(Config.MODEL_PATH)
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None

    def predict_signal(self, symbol: str, timeframe: str = "5m") -> dict:
        """
        Analyzes live market candles, runs ML prediction & produces Explainable AI reasoning.
        """
        df = DataFeed.fetch_data(symbol, timeframe)
        if df is None or len(df) < 30:
            return {"error": "Failed to retrieve market candles."}

        # Feature extraction
        df_featured = FeatureEngine.generate_features(df)
        feature_cols = FeatureEngine.get_feature_columns()
        latest_candle = df_featured.iloc[[-1]][feature_cols]

        # Extract Technical Values for XAI (Explanation)
        rsi = float(df_featured['rsi'].iloc[-1])
        bos = int(df_featured['bos'].iloc[-1])
        bull_ob = int(df_featured['bullish_ob'].iloc[-1])
        bear_ob = int(df_featured['bearish_ob'].iloc[-1])
        sweep_high = int(df_featured['liquidity_sweep_high'].iloc[-1])
        sweep_low = int(df_featured['liquidity_sweep_low'].iloc[-1])

        # Model Inference
        if self.model is not None:
            probs = self.model.predict_proba(latest_candle)[0]
            prob_put = float(probs[0])
            prob_call = float(probs[1])
        else:
            # Fallback Rule-based probabilities if model is not trained yet
            prob_call = 0.60 if (bull_ob or sweep_low or bos == 1) else 0.40
            prob_put = 0.60 if (bear_ob or sweep_high or bos == -1) else 0.40

        confidence = round(max(prob_call, prob_put) * 100, 1)

        # XAI (Explainable AI) Reason Builder
        reasons = []
        if bos == 1: reasons.append("Bullish Structure Break (BOS)")
        elif bos == -1: reasons.append("Bearish Structure Break (BOS)")
        
        if bull_ob: reasons.append("Bullish Order Block Zone Detected")
        if bear_ob: reasons.append("Bearish Order Block Zone Detected")
        
        if sweep_low: reasons.append("Liquidity Sweep at Support Level")
        if sweep_high: reasons.append("Liquidity Sweep at Resistance Level")
        
        if rsi < 35: reasons.append(f"RSI Oversold ({round(rsi, 1)})")
        elif rsi > 65: reasons.append(f"RSI Overbought ({round(rsi, 1)})")

        xai_explanation = ", ".join(reasons) if reasons else "Standard ML Technical Alignment"

        # Signal Output Logic (Threshold >= 65%)
        threshold = Config.ML_CONFIDENCE_THRESHOLD
        if prob_call >= threshold:
            direction = "CALL"
            reason = f"BUY Signal triggered: {xai_explanation} (Confidence: {confidence}%)"
        elif prob_put >= threshold:
            direction = "PUT"
            reason = f"SELL Signal triggered: {xai_explanation} (Confidence: {confidence}%)"
        else:
            direction = "WAIT"
            reason = f"NO TRADE: Probability ({confidence}%) below 65% threshold. Reason: {xai_explanation}"

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "confidence": confidence,
            "prob_call": round(prob_call * 100, 1),
            "prob_put": round(prob_put * 100, 1),
            "explanation": reason,
            "rsi": round(rsi, 2)
        }
