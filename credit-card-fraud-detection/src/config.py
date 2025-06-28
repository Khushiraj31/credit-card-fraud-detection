import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'creditcard.csv')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'validation_data.csv')

MODEL_PATHS = {
    'random_forest': os.path.join(MODELS_DIR, 'random_forest_model.pkl'),
    'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression_model.pkl'),
    'xgboost': os.path.join(MODELS_DIR, 'xgboost_model.pkl'),
    'scaler': os.path.join(MODELS_DIR, 'scaler.pkl')
}

RESULTS_PATHS = {
    'predictions': os.path.join(RESULTS_DIR, 'model_predictions.csv'),
    'feature_importance': os.path.join(RESULTS_DIR, 'feature_importance.csv'),
    'metrics': os.path.join(RESULTS_DIR, 'performance_metrics.json')
}

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'random_state': 42
    }
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1