import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from model_evaluation import ModelEvaluator
from utils import load_model
from config import *

def main():
    print("Starting model evaluation...")
    
    evaluator = ModelEvaluator()
    
    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test = test_data.drop(columns=['Class'])
    y_test = test_data['Class']
    
    model_names = ['random_forest', 'logistic_regression', 'xgboost']
    evaluation_results = {}
    
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        
        try:
            model = load_model(MODEL_PATHS[model_name])
            metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = evaluator.evaluation_results[model_name]
            
            evaluator.save_evaluation_results(model_name)
            
            evaluator.plot_confusion_matrix(y_test, evaluation_results[model_name]['predictions'], model_name)
            evaluator.plot_roc_curve(y_test, evaluation_results[model_name]['probabilities'], model_name)
            evaluator.plot_precision_recall_curve(y_test, evaluation_results[model_name]['probabilities'], model_name)
            
            threshold_analysis = evaluator.threshold_analysis(y_test, evaluation_results[model_name]['probabilities'])
            print(f"\nThreshold Analysis for {model_name}:")
            print(threshold_analysis)
            
        except FileNotFoundError:
            print(f"Model {model_name} not found. Please train the model first.")
    
    if evaluation_results:
        print("\nGenerating model comparison...")
        comparison_df = evaluator.compare_models(evaluation_results)
        print("\nModel Comparison:")
        print(comparison_df)
        
        evaluator.plot_model_comparison(comparison_df)
        
        comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
        
        best_model = comparison_df.loc['F1-Score'].idxmax()
        print(f"\nBest performing model: {best_model}")
        
        print("\nAnalyzing prediction errors...")
        best_results = evaluation_results[best_model]
        fp_analysis = evaluator.analyze_false_positives(X_test, y_test, best_results['predictions'])
        fn_analysis = evaluator.analyze_false_negatives(X_test, y_test, best_results['predictions'])
        
        if not fp_analysis.empty:
            print("\nFalse Positives Analysis:")
            print(fp_analysis)
        
        if not fn_analysis.empty:
            print("\nFalse Negatives Analysis:")
            print(fn_analysis)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()