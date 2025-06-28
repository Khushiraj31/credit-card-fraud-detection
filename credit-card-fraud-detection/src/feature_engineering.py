import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from config import *

class FeatureEngineer:
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        
    def create_time_features(self, df, time_column='Time'):
        if time_column in df.columns:
            df['hour'] = (df[time_column] / 3600).astype(int) % 24
            df['day_of_week'] = ((df[time_column] / (3600 * 24)).astype(int)) % 7
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        return df
    
    def create_amount_features(self, df, amount_column='Amount'):
        if amount_column in df.columns:
            df['amount_log'] = np.log1p(df[amount_column])
            df['amount_sqrt'] = np.sqrt(df[amount_column])
            df['amount_squared'] = df[amount_column] ** 2
            
            df['amount_percentile'] = pd.qcut(df[amount_column], q=100, labels=False, duplicates='drop')
            
            amount_bins = [0, 1, 10, 50, 100, 500, 1000, float('inf')]
            df['amount_category'] = pd.cut(df[amount_column], bins=amount_bins, labels=False)
        
        return df
    
    def create_interaction_features(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if 'Amount' in df.columns and 'Time' in df.columns:
            df['amount_time_ratio'] = df['Amount'] / (df['Time'] + 1)
        
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns[:5]):
                for col2 in numeric_columns[i+1:6]:
                    if col1 != col2:
                        df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                        df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
        
        return df
    
    def create_statistical_features(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            df['features_mean'] = df[numeric_columns].mean(axis=1)
            df['features_std'] = df[numeric_columns].std(axis=1)
            df['features_max'] = df[numeric_columns].max(axis=1)
            df['features_min'] = df[numeric_columns].min(axis=1)
            df['features_range'] = df['features_max'] - df['features_min']
        
        return df
    
    def select_features_univariate(self, X, y, k=20):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        return X_selected, selected_features
    
    def select_features_rfe(self, X, y, n_features=20):
        estimator = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        return X_selected, selected_features
    
    def apply_pca(self, X, n_components=0.95):
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = self.pca.fit_transform(X)
        return X_pca
    
    def get_feature_importance(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def remove_correlated_features(self, df, threshold=0.95):
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        return df.drop(columns=to_drop)
    
    def engineer_features(self, df):
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        df = self.remove_correlated_features(df)
        
        return df
    
    def transform_features(self, X):
        if self.feature_selector:
            return self.feature_selector.transform(X)
        return X