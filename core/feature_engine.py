import pandas as pd
import numpy as np

# 'from core.market_structure import ...' এর বদলে সরাসরি import:
from market_structure import MarketStructure
from liquidity import LiquidityEngine
from order_block import OrderBlockDetector
from fair_value_gap import FairValueGap

class FeatureEngine:
    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        SMC Features + Classic Technical Indicators
        """
        df = df.copy()

        # 1. Apply SMC Engines
        df = MarketStructure.analyze(df)
        df = LiquidityEngine.detect_liquidity(df)
        df = OrderBlockDetector.find_order_blocks(df)
        df = FairValueGap.detect_fvg(df)

        # 2. Candlestick Features
        open_p, high_p, low_p, close_p = df['Open'], df['High'], df['Low'], df['Close']
        df['body_size'] = abs(close_p - open_p) / (open_p + 1e-8)
        df['upper_wick'] = (high_p - df[['Open', 'Close']].max(axis=1)) / (open_p + 1e-8)
        df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - low_p) / (open_p + 1e-8)

        # 3. Technical Indicators (RSI, EMA, Momentum)
        # RSI
        delta = close_p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-5)
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMAs
        df['ema_20'] = close_p.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close_p.ewm(span=50, adjust=False).mean()
        df['dist_ema_50'] = (close_p - df['ema_50']) / (df['ema_50'] + 1e-8)
        df['ema_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

        # Volatility
        df['volatility'] = close_p.pct_change().rolling(window=10).std()

        # Clean NaN values
        df = df.fillna(0)
        return df

    @staticmethod
    def get_feature_columns():
        return [
            'body_size', 'upper_wick', 'lower_wick', 'rsi', 
            'dist_ema_50', 'ema_cross', 'volatility', 'bos', 
            'structure_trend', 'is_eqh', 'is_eql', 'liquidity_sweep_high', 
            'liquidity_sweep_low', 'bullish_ob', 'bearish_ob', 
            'has_bullish_fvg', 'has_bearish_fvg'
        ]
