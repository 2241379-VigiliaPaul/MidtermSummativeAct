# Midterm Summative Activity: Customer Churn Prediction

## Project Overview
This project implements customer churn prediction using machine learning models on the Telco Customer Churn dataset.

## Dataset
- **Source**: Telco Customer Churn dataset
- **Requirements**: ≥1000 rows, classification-friendly
- **Target Variable**: Churn (Yes/No)

## Models Implemented
1. Decision Tree (Rule-based)
2. Naive Bayes
3. Support Vector Machine (SVM)
4. [Chosen Algorithm - TBD]

## Data Splitting
- 90% for training/testing
  - 80% training
  - 20% testing
- 10% unseen validation

## Evaluation
- 10-fold cross-validation
- Metrics: Confusion Matrix, Accuracy, Precision, Recall, F1-score, Specificity (manual computation)

## Analysis
- Compare model performance on test set vs unseen validation set

## Repository Structure
```
.
├── data/
│   ├── raw/           # Original dataset
│   ├── processed/     # Cleaned/preprocessed data
│   └── splits/        # Train/test/validation splits
├── src/
│   ├── data_preprocessing.py
│   ├── models/
│   │   ├── decision_tree.py
│   │   ├── naive_bayes.py
│   │   ├── svm.py
│   │   └── chosen_algorithm.py
│   ├── evaluation.py
│   └── analysis.py
├── models/            # Saved trained models
├── results/           # Evaluation results and reports
├── notebooks/         # Jupyter notebooks for exploration
├── requirements.txt   # Python dependencies
├── README.md
└── .gitignore
```

## Setup Instructions
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place the Telco Customer Churn dataset in `data/raw/`
4. Run preprocessing: `python src/data_preprocessing.py`
5. Train models and evaluate

## Team Members
- [List team members and their assigned models/tasks]

## Requirements
- Python 3.8+
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn