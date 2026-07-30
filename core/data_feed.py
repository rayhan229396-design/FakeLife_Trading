import yfinance as yf
import pandas as pd

class DataFeed:
    @staticmethod
    def fetch_data(symbol: str = "EURUSD", timeframe: str = "5m", period: str = "5d") -> pd.DataFrame:
        """
        Fetches historical candle data safely, auto-handling Forex/Crypto symbols.
        """
        try:
            clean_symbol = symbol.strip().upper()
            
            # Remove any unwanted URL encoding leftovers
            clean_symbol = clean_symbol.replace("%3D", "=").replace("=", "")
            
            # Auto-append =X for 6-letter currency pairs (e.g. EURUSD -> EURUSD=X)
            if len(clean_symbol) == 6 and not clean_symbol.endswith("=X"):
                clean_symbol = f"{clean_symbol}=X"

            ticker = yf.Ticker(clean_symbol)
            
            if timeframe in ["1m", "2m", "5m", "15m", "30m", "60m"]:
                df = ticker.history(period="5d", interval=timeframe)
            else:
                df = ticker.history(period="1mo", interval=timeframe)

            if df.empty:
                return None

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
