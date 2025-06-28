import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from utils import create_directories
from config import *

def main():
    create_directories()
    
    print("Starting Credit Card Fraud Detection Model Training...")
    
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer()
    
    print("Loading and preprocessing data...")
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.preprocess_pipeline()
    
    print("Engineering features...")
    X_train_eng = feature_engineer.engineer_features(X_train)
    X_test_eng = feature_engineer.engineer_features(X_test)
    X_val_eng = feature_engineer.engineer_features(X_val)
    
    print("Selecting important features...")
    X_train_selected, selected_features = feature_engineer.select_features_univariate(X_train_eng, y_train, k=20)
    X_test_selected = feature_engineer.transform_features(X_test_eng[selected_features])
    X_val_selected = feature_engineer.transform_features(X_val_eng[selected_features])
    
    print(f"Selected features: {selected_features}")
    
    print("Training models...")
    trained_models = trainer.train_all_models(
        pd.DataFrame(X_train_selected, columns=selected_features), 
        y_train, 
        tune_hyperparameters=True
    )
    
    print("Saving feature importance...")
    for model_name, model in trained_models.items():
        importance_df = trainer.get_feature_importance(model_name, selected_features)
        if importance_df is not None:
            importance_path = RESULTS_PATHS['feature_importance'].replace('.csv', f'_{model_name}.csv')
            importance_df.to_csv(importance_path, index=False)
    
    print("Model training completed successfully!")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()