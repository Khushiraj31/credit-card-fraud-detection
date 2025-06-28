import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def create_directories():
    directories = [
        'data/raw', 'data/processed', 'data/results',
        'models', 'notebooks', 'tests', 'docs', 'scripts'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {}
    
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['precision'] = report['1']['precision']
    metrics['recall'] = report['1']['recall']
    metrics['f1_score'] = report['1']['f1-score']
    metrics['accuracy'] = report['accuracy']
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def print_model_performance(model_name, metrics):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

def detect_outliers_iqr(df, columns):
    outliers = pd.DataFrame()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers])
    return outliers.drop_duplicates()

def balance_dataset(X, y, method='smote'):
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
    else:
        X_balanced, y_balanced = X, y
    
    return X_balanced, y_balanced