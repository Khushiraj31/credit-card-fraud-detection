import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from utils import load_model
from config import *

def predict_fraud(input_file, output_file=None, threshold=0.5):
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)
    
    print("Loading trained models...")
    models = {}
    model_names = ['random_forest', 'logistic_regression', 'xgboost']
    
    for model_name in model_names:
        try:
            models[model_name] = load_model(MODEL_PATHS[model_name])
            print(f"Loaded {model_name} model")
        except FileNotFoundError:
            print(f"Warning: {model_name} model not found")
    
    if not models:
        print("Error: No trained models found. Please train models first.")
        return
    
    try:
        scaler = load_model(MODEL_PATHS['scaler'])
        print("Loaded feature scaler")
    except FileNotFoundError:
        print("Warning: Scaler not found. Using raw features.")
        scaler = None
    
    X = data.copy()
    if 'Class' in X.columns:
        X = X.drop(columns=['Class'])
    
    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = X
    
    print("Making predictions...")
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        pred_proba = model.predict_proba(X_scaled)[:, 1]
        pred = (pred_proba >= threshold).astype(int)
        
        predictions[f'{model_name}_prediction'] = pred
        probabilities[f'{model_name}_probability'] = pred_proba
        
        fraud_count = np.sum(pred)
        fraud_percentage = (fraud_count / len(pred)) * 100
        
        print(f"{model_name}: {fraud_count} fraudulent transactions ({fraud_percentage:.2f}%)")
    
    if len(models) > 1:
        print("Creating ensemble predictions...")
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = (ensemble_prob >= threshold).astype(int)
        
        predictions['ensemble_prediction'] = ensemble_pred
        probabilities['ensemble_probability'] = ensemble_prob
        
        fraud_count = np.sum(ensemble_pred)
        fraud_percentage = (fraud_count / len(ensemble_pred)) * 100
        print(f"Ensemble: {fraud_count} fraudulent transactions ({fraud_percentage:.2f}%)")
    
    results_df = data.copy()
    for col, values in predictions.items():
        results_df[col] = values
    for col, values in probabilities.items():
        results_df[col] = values
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_predictions.csv')
    
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    high_risk_transactions = results_df[results_df['ensemble_probability'] > 0.8] if 'ensemble_probability' in results_df.columns else results_df[results_df[list(probabilities.keys())[0]] > 0.8]
    
    if len(high_risk_transactions) > 0:
        print(f"\nHigh-risk transactions (probability > 0.8): {len(high_risk_transactions)}")
        high_risk_file = output_file.replace('.csv', '_high_risk.csv')
        high_risk_transactions.to_csv(high_risk_file, index=False)
        print(f"High-risk transactions saved to {high_risk_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Predict fraudulent transactions')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    predict_fraud(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()