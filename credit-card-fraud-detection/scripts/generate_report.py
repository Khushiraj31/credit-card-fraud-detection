import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from config import *

def generate_performance_report():
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'project': 'Credit Card Fraud Detection',
        'models': {}
    }
    
    model_names = ['random_forest', 'logistic_regression', 'xgboost']
    
    for model_name in model_names:
        metrics_file = RESULTS_PATHS['metrics'].replace('.json', f'_{model_name}.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            report['models'][model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc_roc': metrics.get('auc_roc', 0),
                'average_precision': metrics.get('average_precision', 0)
            }
    
    comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file, index_col=0)
        best_model = comparison_df.loc['F1-Score'].idxmax()
        report['best_model'] = best_model
        report['best_f1_score'] = comparison_df.loc['F1-Score', best_model]
    
    return report

def generate_feature_importance_summary():
    summary = {}
    model_names = ['random_forest', 'logistic_regression', 'xgboost']
    
    for model_name in model_names:
        importance_file = RESULTS_PATHS['feature_importance'].replace('.csv', f'_{model_name}.csv')
        
        if os.path.exists(importance_file):
            importance_df = pd.read_csv(importance_file)
            top_features = importance_df.head(10)
            
            summary[model_name] = {
                'top_features': top_features['feature'].tolist(),
                'importance_scores': top_features['importance'].tolist()
            }
    
    return summary

def create_markdown_report(performance_report, feature_summary):
    markdown_content = f"""# Credit Card Fraud Detection - Performance Report

**Generated on:** {performance_report['generated_at']}

## Model Performance Summary

"""
    
    if performance_report['models']:
        markdown_content += "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        markdown_content += "|-------|----------|-----------|--------|----------|----------|\n"
        
        for model_name, metrics in performance_report['models'].items():
            markdown_content += f"| {model_name.replace('_', ' ').title()} | "
            markdown_content += f"{metrics['accuracy']:.4f} | "
            markdown_content += f"{metrics['precision']:.4f} | "
            markdown_content += f"{metrics['recall']:.4f} | "
            markdown_content += f"{metrics['f1_score']:.4f} | "
            markdown_content += f"{metrics['auc_roc']:.4f} |\n"
    
    if 'best_model' in performance_report:
        markdown_content += f"""
## Best Performing Model

**{performance_report['best_model'].replace('_', ' ').title()}** achieved the highest F1-Score of **{performance_report['best_f1_score']:.4f}**

"""
    
    markdown_content += """
## Key Insights

### Model Comparison
- All models were trained using balanced datasets with SMOTE oversampling
- Performance metrics focus on minimizing false positives while maintaining high recall
- Cross-validation was used to ensure model stability

### Feature Engineering
- Time-based features were extracted from transaction timestamps
- Amount-based features including logarithmic and categorical transformations
- Statistical features computed across transaction patterns
- Feature selection was performed to reduce dimensionality

"""
    
    if feature_summary:
        markdown_content += "## Top Important Features\n\n"
        
        for model_name, features in feature_summary.items():
            markdown_content += f"### {model_name.replace('_', ' ').title()}\n\n"
            
            for i, (feature, importance) in enumerate(zip(features['top_features'], features['importance_scores']), 1):
                markdown_content += f"{i}. **{feature}**: {importance:.4f}\n"
            
            markdown_content += "\n"
    
    markdown_content += """
## Recommendations

1. **Production Deployment**: Use ensemble predictions for final fraud detection decisions
2. **Threshold Tuning**: Adjust prediction thresholds based on business requirements for false positive tolerance
3. **Model Monitoring**: Implement drift detection to monitor model performance over time
4. **Feature Updates**: Regularly update feature engineering pipeline with new transaction patterns
5. **Retraining**: Schedule periodic model retraining with fresh data

## Technical Notes

- **Class Imbalance**: Handled using SMOTE oversampling technique
- **Feature Scaling**: StandardScaler applied to numerical features
- **Model Selection**: Grid search used for hyperparameter optimization
- **Evaluation**: Stratified cross-validation with F1-score as primary metric
"""
    
    return markdown_content

def main():
    print("Generating performance report...")
    
    performance_report = generate_performance_report()
    feature_summary = generate_feature_importance_summary()
    
    report_file = os.path.join(RESULTS_DIR, 'performance_report.json')
    with open(report_file, 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    markdown_report = create_markdown_report(performance_report, feature_summary)
    
    markdown_file = os.path.join(RESULTS_DIR, 'performance_report.md')
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    print(f"Performance report saved to: {report_file}")
    print(f"Markdown report saved to: {markdown_file}")
    
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    if performance_report['models']:
        for model_name, metrics in performance_report['models'].items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    if 'best_model' in performance_report:
        print(f"\nBest Model: {performance_report['best_model'].replace('_', ' ').title()}")
        print(f"Best F1-Score: {performance_report['best_f1_score']:.4f}")

if __name__ == "__main__":
    main()