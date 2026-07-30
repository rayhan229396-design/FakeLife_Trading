import numpy as np
from sklearn.calibration import CalibratedClassifierCV

class ProbabilityCalibrator:
    @staticmethod
    def calibrate_model(base_model, X_train, y_train, method: str = 'isotonic'):
        """
        Calibrates model probability outputs using Isotonic Regression or Sigmoid.
        """
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model, 
            method=method, 
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        return calibrated_model

    @staticmethod
    def get_calibrated_probabilities(model, X_sample):
        """
        Returns calibrated Bullish (CALL) and Bearish (PUT) probabilities.
        """
        probs = model.predict_proba(X_sample)[0]
        prob_bearish = float(probs[0])
        prob_bullish = float(probs[1])
        
        return prob_bullish, prob_bearish
