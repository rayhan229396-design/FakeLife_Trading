import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from prepare_dataset import DatasetPreparer
from calibration import ProbabilityCalibrator
from data_feed import DataFeed
from config import Config

class ModelTrainer:
    @staticmethod
    def train_and_save(symbol: str = "EURUSD", timeframe: str = "5m"):
        """
        Fetches historical data, trains Random Forest model with probability calibration, 
        and saves it to the models folder.
        """
        print(f"Fetching training data for {symbol} ({timeframe})...")
        df = DataFeed.fetch_data(symbol, timeframe)
        if df is None or len(df) < 100:
            print("Error: Not enough data to train model.")
            return False

        print("Preparing features and targets...")
        X, y, _ = DatasetPreparer.prepare_training_data(df)

        # Base Random Forest Engine
        base_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        print("Calibrating model probabilities...")
        calibrated_model = ProbabilityCalibrator.calibrate_model(
            base_clf, X, y, method=Config.CALIBRATION_METHOD
        )

        # Save model to disk
        model_dir = os.path.dirname(Config.MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(calibrated_model, Config.MODEL_PATH)
        print(f"Model successfully saved to: {Config.MODEL_PATH}")
        return True

if __name__ == "__main__":
    ModelTrainer.train_and_save()
