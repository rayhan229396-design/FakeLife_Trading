import pandas as pd

class OrderBlockDetector:
    @staticmethod
    def find_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies Bullish and Bearish Order Blocks (OB).
        """
        df = df.copy()

        # Bullish OB: The last bearish candle before a strong upward movement
        is_bearish_candle = df['Close'] < df['Open']
        strong_up_move = (df['Close'].shift(-1) > df['High']) | (df['Close'].shift(-2) > df['High'])
        
        df['bullish_ob'] = (is_bearish_candle & strong_up_move).astype(int)

        # Bearish OB: The last bullish candle before a strong downward movement
        is_bullish_candle = df['Close'] > df['Open']
        strong_down_move = (df['Close'].shift(-1) < df['Low']) | (df['Close'].shift(-2) < df['Low'])

        df['bearish_ob'] = (is_bullish_candle & strong_down_move).astype(int)

        return df
