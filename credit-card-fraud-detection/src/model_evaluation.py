import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           precision_recall_curve, roc_auc_score, average_precision_score)
from config import *
from utils import calculate_metrics, save_metrics, print_model_performance

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
        
        self.evaluation_results[model_name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        print_model_performance(model_name, metrics)
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.show()
        return fpr, tpr, auc_score
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.show()
        return precision, recall, avg_precision
    
    def compare_models(self, models_results):
        comparison_df = pd.DataFrame()
        
        for model_name, results in models_results.items():
            metrics = results['metrics']
            comparison_df[model_name] = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics.get('auc_roc', 0),
                metrics.get('average_precision', 0)
            ]
        
        comparison_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AP']
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        plt.figure(figsize=(12, 8))
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def analyze_false_positives(self, X_test, y_test, y_pred, top_features=10):
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_samples = X_test[fp_mask]
        
        if len(fp_samples) > 0:
            fp_analysis = pd.DataFrame({
                'feature': X_test.columns,
                'fp_mean': fp_samples.mean(),
                'overall_mean': X_test.mean(),
                'difference': fp_samples.mean() - X_test.mean()
            })
            
            fp_analysis['abs_difference'] = np.abs(fp_analysis['difference'])
            fp_analysis = fp_analysis.sort_values('abs_difference', ascending=False)
            
            return fp_analysis.head(top_features)
        
        return pd.DataFrame()
    
    def analyze_false_negatives(self, X_test, y_test, y_pred, top_features=10):
        fn_mask = (y_test == 1) & (y_pred == 0)
        fn_samples = X_test[fn_mask]
        
        if len(fn_samples) > 0:
            fn_analysis = pd.DataFrame({
                'feature': X_test.columns,
                'fn_mean': fn_samples.mean(),
                'overall_mean': X_test.mean(),
                'difference': fn_samples.mean() - X_test.mean()
            })
            
            fn_analysis['abs_difference'] = np.abs(fn_analysis['difference'])
            fn_analysis = fn_analysis.sort_values('abs_difference', ascending=False)
            
            return fn_analysis.head(top_features)
        
        return pd.DataFrame()
    
    def threshold_analysis(self, y_true, y_pred_proba, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            metrics = calculate_metrics(y_true, y_pred_thresh)
            
            results.append({
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            })
        
        return pd.DataFrame(results)
    
    def save_evaluation_results(self, model_name):
        if model_name in self.evaluation_results:
            results = self.evaluation_results[model_name]
            save_metrics(results['metrics'], RESULTS_PATHS['metrics'].replace('.json', f'_{model_name}.json'))
            
            predictions_df = pd.DataFrame({
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            })
            predictions_df.to_csv(RESULTS_PATHS['predictions'].replace('.csv', f'_{model_name}.csv'), index=False)
    
    def generate_classification_report(self, y_true, y_pred, model_name):
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        print(f"\nClassification Report - {model_name}")
        print(report_df)
        
        return report_df