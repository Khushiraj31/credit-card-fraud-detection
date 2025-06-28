from src.model_training import train_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    df = pd.read_csv("data/processed/train_data.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    model = train_model(X, y, RandomForestClassifier())
    assert hasattr(model, "predict")
