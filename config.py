import os

class Config:
    # App Settings
    APP_NAME: str = "AI Smart Trading Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # Timezone
    TIMEZONE: str = "Asia/Dhaka"

    # Supported Assets
    PAIRS = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "CAD=X",
        "USDCHF": "CHF=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X",
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "XAUUSD": "GC=F"
    }

    # Timeframes & Historical Data Limits (Optimized for Render Free RAM)
    TIMEFRAMES = {
        "1m": {"period": "5d", "interval": "1m"},
        "5m": {"period": "20d", "interval": "5m"},
        "15m": {"period": "60d", "interval": "15m"},
        "1h": {"period": "100d", "interval": "1h"}
    }

    # Machine Learning Thresholds
    ML_CONFIDENCE_THRESHOLD: float = 0.65  # 65%+ Probability
    CALIBRATION_METHOD: str = "isotonic"    # 'isotonic' or 'sigmoid'

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(os.path.join(Config.BASE_DIR, "models"), exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
