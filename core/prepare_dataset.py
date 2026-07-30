import pandas as pd
from feature_engine import FeatureEngine

class DatasetPreparer:
    @staticmethod
    def prepare_training_data(df: pd.DataFrame):
        """
        Extracts features and creates binary classification target labels.
        Target: 1 if Next Close > Current Close, else 0
        """
        df = FeatureEngine.generate_features(df)
        
        # Target creation (Next candle direction)
        df['next_close'] = df['Close'].shift(-1)
        df['target'] = (df['next_close'] > df['Close']).astype(int)

        # Drop the last row since its target is unknown
        clean_df = df.iloc[:-1].copy()
        
        feature_cols = FeatureEngine.get_feature_columns()
        X = clean_df[feature_cols]
        y = clean_df['target']

        return X, y, df
