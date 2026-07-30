import pandas as pd
import numpy as np

class FairValueGap:
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies 3-Candle Fair Value Gaps (Bullish & Bearish Imbalances).
        """
        df = df.copy()

        # Bullish FVG: Candle 1 High < Candle 3 Low
        bullish_fvg = df['Low'] > df['High'].shift(2)
        df['bullish_fvg_top'] = np.where(bullish_fvg, df['Low'], np.nan)
        df['bullish_fvg_bottom'] = np.where(bullish_fvg, df['High'].shift(2), np.nan)
        df['has_bullish_fvg'] = np.where(bullish_fvg, 1, 0)

        # Bearish FVG: Candle 1 Low > Candle 3 High
        bearish_fvg = df['High'] < df['Low'].shift(2)
        df['bearish_fvg_top'] = np.where(bearish_fvg, df['Low'].shift(2), np.nan)
        df['bearish_fvg_bottom'] = np.where(bearish_fvg, df['High'], np.nan)
        df['has_bearish_fvg'] = np.where(bearish_fvg, 1, 0)

        return df
