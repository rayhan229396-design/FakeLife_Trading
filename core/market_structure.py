import pandas as pd
import numpy as np

class MarketStructure:
    @staticmethod
    def analyze(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Market Structure Analysis: Swing Highs/Lows, BOS & CHoCH
        """
        df = df.copy()
        
        # 1. Identify Swing Highs & Swing Lows
        df['swing_high'] = df['High'][(df['High'] == df['High'].rolling(window*2+1, center=True).max())]
        df['swing_low'] = df['Low'][(df['Low'] == df['Low'].rolling(window*2+1, center=True).min())]

        # Forward fill last swing points
        df['last_swing_high'] = df['swing_high'].ffill()
        df['last_swing_low'] = df['swing_low'].ffill()

        # 2. Break of Structure (BOS) & Change of Character (CHoCH)
        bos_bull = (df['Close'] > df['last_swing_high'].shift(1)) & (df['Close'].shift(1) <= df['last_swing_high'].shift(1))
        bos_bear = (df['Close'] < df['last_swing_low'].shift(1)) & (df['Close'].shift(1) >= df['last_swing_low'].shift(1))

        df['bos'] = 0
        df.loc[bos_bull, 'bos'] = 1   # Bullish Structure Break
        df.loc[bos_bear, 'bos'] = -1  # Bearish Structure Break

        # Overall Trend Direction based on Market Structure
        df['structure_trend'] = np.where(df['Close'] > df['last_swing_high'], 1,
                                np.where(df['Close'] < df['last_swing_low'], -1, 0))
        df['structure_trend'] = df['structure_trend'].replace(0, np.nan).ffill().fillna(0)

        return df
