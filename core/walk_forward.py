import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prepare_dataset import DatasetPreparer
from calibration import ProbabilityCalibrator
from config import Config

class WalkForwardValidator:
    @staticmethod
    def run_validation(df: pd.DataFrame, train_size: int = 500, test_size: int = 100):
        """
        Walk-Forward Machine Learning Validation for Time Series.
        Prevents look-ahead bias by training on past data and testing on subsequent future data.
        """
        X, y, clean_df = DatasetPreparer.prepare_training_data(df)
        total_samples = len(X)
        
        if total_samples < (train_size + test_size):
            return {"error": "Not enough historical candles for Walk-Forward Validation."}

        predictions = []
        actuals = []
        
        # Sliding Window Loop
        for start in range(0, total_samples - train_size - test_size, test_size):
            train_end = start + train_size
            test_end = train_end + test_size
            
            X_train, y_train = X.iloc[start:train_end], y.iloc[start:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
            
            base_clf = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            calibrated_model = ProbabilityCalibrator.calibrate_model(
                base_clf, X_train, y_train, method=Config.CALIBRATION_METHOD
            )
            
            preds = calibrated_model.predict(X_test)
            predictions.extend(preds)
            actuals.extend(y_test)

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        accuracy = float(np.mean(predictions == actuals) * 100) if len(actuals) > 0 else 0.0
        
        return {
            "total_tested_candles": len(actuals),
            "walk_forward_accuracy": round(accuracy, 2),
            "status": "PASS" if accuracy >= 55.0 else "NEEDS_OPTIMIZATION"
        }
