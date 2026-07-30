import logging
import pandas as pd
import yfinance as yf
from typing import Optional, Dict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFeed")

class DataFeed:
    @staticmethod
    def fetch_data(symbol: str, timeframe: str = "5m") -> Optional[pd.DataFrame]:
        """
        Yahoo Finance থেকে ক্যান্ডেলস্টিক ডেটা লোড করার জন্য নিরাপদ ও লাইটওয়েট ফাংশন।
        """
        yf_symbol = Config.PAIRS.get(symbol, f"{symbol}=X")
        tf_info = Config.TIMEFRAMES.get(timeframe, Config.TIMEFRAMES["5m"])

        try:
            logger.info(f"Fetching {symbol} ({yf_symbol}) for timeframe {timeframe}...")
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=tf_info["period"], interval=tf_info["interval"])

            if df.empty or len(df) < 50:
                # Fallback API call
                df = yf.download(
                    tickers=yf_symbol, 
                    period=tf_info["period"], 
                    interval=tf_info["interval"], 
                    progress=False
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(yf_symbol, axis=1, level=1)

            if df.empty or len(df) < 50:
                logger.error(f"Insufficient data returned for {symbol}")
                return None

            # Clean and standard format
            df = df.reset_index()
            # Rename columns if needed
            df.columns = [c.capitalize() if isinstance(c, str) else c[0].capitalize() for c in df.columns]

            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
            elif 'Date' in df.columns:
                df.rename(columns={'Date': 'Timestamp'}, inplace=True)

            required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[c for c in required_cols if c in df.columns]]
            df = df.dropna().copy()
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    @classmethod
    def fetch_multi_timeframe(cls, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        হায়ার এবং লোয়ার একাধিক টাইমফ্রেমের ডেটা একসাথে এনে দেওয়ার ব্যবস্থা (MTF Analysis)।
        """
        mtf_data = {}
        for tf in ["1m", "5m", "15m"]:
            df = cls.fetch_data(symbol, tf)
            if df is not None:
                mtf_data[tf] = df
        return mtf_data
