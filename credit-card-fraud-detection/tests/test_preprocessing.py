from src.data_preprocessing import preprocess_data
import pandas as pd

def test_preprocess_output_shape():
    df = pd.read_csv("data/raw/creditcard.csv")
    X, y = preprocess_data(df)
    assert X.shape[0] == y.shape[0]
