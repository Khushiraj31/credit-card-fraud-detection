# Credit Card Fraud Detection

Machine learning system for identifying fraudulent credit card transactions using supervised learning algorithms.

## Project Overview

This project implements various ML models to detect fraudulent transactions with focus on minimizing false positives while maintaining high detection accuracy.

## Dataset

Place your credit card dataset as `creditcard.csv` in the `data/raw/` directory. The dataset should contain transaction features and a binary target variable indicating fraud.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Models
```bash
python scripts/train_models.py
```

### Evaluating Models
```bash
python scripts/evaluate_models.py
```

### Making Predictions
```bash
python scripts/predict_fraud.py --input data/raw/new_transactions.csv
```

## Model Performance

The system implements multiple algorithms:
- Logistic Regression
- Random Forest
- XGBoost

Performance metrics focus on precision, recall, F1-score, and AUC-ROC to handle class imbalance effectively.

## Project Structure

- `src/`: Core implementation modules
- `notebooks/`: Jupyter notebooks for analysis
- `scripts/`: Executable scripts for training and prediction
- `models/`: Saved trained models
- `data/`: Dataset storage and processing results
- `tests/`: Unit tests for validation

## Results

Model performance and analysis results are stored in `data/results/` directory after execution.