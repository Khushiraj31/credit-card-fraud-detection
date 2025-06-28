import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from config import *
from utils import save_model

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
    def load_data(self, filepath=RAW_DATA_PATH):
        df = pd.read_csv(filepath)
        return df
    
    def basic_cleaning(self, df):
        df = df.drop_duplicates()
        df = df.dropna()
        return df
    
    def handle_missing_values(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def remove_outliers(self, df, columns, method='iqr'):
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def scale_features(self, X_train, X_test, X_val=None, method='standard'):
        if method == 'standard':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            save_model(self.scaler, MODEL_PATHS['scaler'])
        else:
            X_train_scaled = self.robust_scaler.fit_transform(X_train)
            X_test_scaled = self.robust_scaler.transform(X_test)
        
        if X_val is not None:
            if method == 'standard':
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_val_scaled = self.robust_scaler.transform(X_val)
            return X_train_scaled, X_test_scaled, X_val_scaled
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, target_column='Class'):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), 
            random_state=RANDOM_STATE, stratify=y_temp
        )
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def save_processed_data(self, X_train, X_test, X_val, y_train, y_test, y_val):
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        
        train_data.to_csv(TRAIN_DATA_PATH, index=False)
        test_data.to_csv(TEST_DATA_PATH, index=False)
        val_data.to_csv(VALIDATION_DATA_PATH, index=False)
    
    def preprocess_pipeline(self, filepath=RAW_DATA_PATH, target_column='Class'):
        df = self.load_data(filepath)
        df = self.basic_cleaning(df)
        df = self.handle_missing_values(df)
        
        X_train, X_test, X_val, y_train, y_test, y_val = self.split_data(df, target_column)
        
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_scaled, X_test_scaled, X_val_scaled = self.scale_features(
            X_train[numeric_columns], X_test[numeric_columns], X_val[numeric_columns]
        )
        
        X_train_final = pd.DataFrame(X_train_scaled, columns=numeric_columns, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_scaled, columns=numeric_columns, index=X_test.index)
        X_val_final = pd.DataFrame(X_val_scaled, columns=numeric_columns, index=X_val.index)
        
        self.save_processed_data(X_train_final, X_test_final, X_val_final, y_train, y_test, y_val)
        
        return X_train_final, X_test_final, X_val_final, y_train, y_test, y_val