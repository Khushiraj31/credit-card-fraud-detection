from src.feature_engineering import engineer_features
import pandas as pd

def test_feature_engineering():
    df = pd.read_csv("data/raw/creditcard.csv")
    df_new = engineer_features(df)
    assert 'Amount_log' in df_new.columns
