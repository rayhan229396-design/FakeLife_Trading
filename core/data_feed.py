import yfinance as yf
import pandas as pd

class DataFeed:
    @staticmethod
    def fetch_data(symbol: str = "EURUSD=X", timeframe: str = "5m", period: str = "5d") -> pd.DataFrame:
        """
        Fetches historical candle data from Yahoo Finance safely.
        """
        try:
            # Clean symbol formatting
            clean_symbol = symbol.strip().upper()
            if not clean_symbol.endswith("=X") and len(clean_symbol) == 6:
                clean_symbol = f"{clean_symbol}=X"

            # Fetch ticker data
            ticker = yf.Ticker(clean_symbol)
            
            # Intraday intervals (1m, 5m, 15m) need enough historical period (e.g. 5d or 1mo)
            if timeframe in ["1m", "2m", "5m", "15m", "30m", "60m"]:
                df = ticker.history(period="5d", interval=timeframe)
            else:
                df = ticker.history(period="1mo", interval=timeframe)

            if df.empty:
                return None

            # Clean and prepare columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
