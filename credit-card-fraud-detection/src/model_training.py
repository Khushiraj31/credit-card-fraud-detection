import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from config import *
from utils import save_model, balance_dataset

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        
    def initialize_models(self):
        self.models = {
            'random_forest': RandomForestClassifier(**MODEL_PARAMS['random_forest']),
            'logistic_regression': LogisticRegression(**MODEL_PARAMS['logistic_regression']),
            'xgboost': XGBClassifier(**MODEL_PARAMS['xgboost'], eval_metric='logloss')
        }
    
    def train_model(self, model_name, X_train, y_train, balance_data=True):
        if balance_data:
            X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='smote')
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        model = self.models[model_name]
        model.fit(X_train_balanced, y_train_balanced)
        
        save_model(model, MODEL_PATHS[model_name])
        return model
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, cv=3):
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'xgboost': {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200]
            }
        }
        
        if model_name not in param_grids:
            return self.models[model_name]
        
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
        
        base_model = self.models[model_name]
        grid_search = GridSearchCV(
            base_model, param_grids[model_name], 
            cv=cv, scoring='f1', n_jobs=-1
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        self.best_params[model_name] = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        save_model(best_model, MODEL_PATHS[model_name])
        return best_model
    
    def cross_validate_model(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def train_all_models(self, X_train, y_train, tune_hyperparameters=False):
        self.initialize_models()
        trained_models = {}
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            
            if tune_hyperparameters:
                model = self.hyperparameter_tuning(model_name, X_train, y_train)
            else:
                model = self.train_model(model_name, X_train, y_train)
            
            trained_models[model_name] = model
            
            cv_results = self.cross_validate_model(model, X_train, y_train)
            print(f"Cross-validation F1 Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
        
        return trained_models
    
    def get_feature_importance(self, model_name, feature_names):
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def ensemble_predict(self, models, X):
        predictions = []
        probabilities = []
        
        for model in models.values():
            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]
            predictions.append(pred)
            probabilities.append(prob)
        
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_prob = np.mean(probabilities, axis=0)
        
        ensemble_pred_final = (ensemble_pred > 0.5).astype(int)
        
        return ensemble_pred_final, ensemble_prob