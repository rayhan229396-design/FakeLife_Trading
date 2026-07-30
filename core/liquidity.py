import pandas as pd
import numpy as np

class LiquidityEngine:
    @staticmethod
    def detect_liquidity(df: pd.DataFrame, threshold_pct: float = 0.0005) -> pd.DataFrame:
        """
        Detects Equal Highs (EQH), Equal Lows (EQL) and Liquidity Sweeps.
        """
        df = df.copy()

        # 1. Equal Highs (Resistance Liquidity) & Equal Lows (Support Liquidity)
        high_diff = abs(df['High'] - df['High'].shift(1)) / df['Close']
        low_diff = abs(df['Low'] - df['Low'].shift(1)) / df['Close']

        df['is_eqh'] = np.where(high_diff < threshold_pct, 1, 0)
        df['is_eql'] = np.where(low_diff < threshold_pct, 1, 0)

        # 2. Liquidity Sweeps
        # High Sweep: Price spikes above last high but closes below it
        df['liquidity_sweep_high'] = np.where(
            (df['High'] > df['High'].shift(1)) & (df['Close'] < df['High'].shift(1)), 1, 0
        )

        # Low Sweep: Price spikes below last low but closes above it
        df['liquidity_sweep_low'] = np.where(
            (df['Low'] < df['Low'].shift(1)) & (df['Close'] > df['Low'].shift(1)), 1, 0
        )

        return df
