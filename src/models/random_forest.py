# Import necessary libraries for Random Forest modeling and evaluation
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import joblib

# Define function to train Random Forest classifier
def train_chosen_algorithm(X_train, y_train, n_estimators=100, random_state=42):
    # Train Random Forest ensemble classifier with 100 trees for robust prediction averaging
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    return model

# Define function to compute classification metrics
def compute_metrics(y_true, y_pred, y_prob=None):
    # Calculate all classification metrics including specificity (true negative rate) and ROC-AUC
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "Confusion Matrix": cm,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["ROC-AUC"] = 0.0
    
    return metrics

# Define function to evaluate Random Forest using cross-validation
def evaluate_chosen_algorithm(model, X, y, cv=10):
    # Perform stratified cross-validation to maintain class distribution and return mean metrics
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\nEvaluating Random Forest with {cv}-fold CV...")
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    y_pred_cv = cross_val_predict(model, X, y, cv=splitter)
    y_prob_cv = cross_val_predict(model, X, y, cv=splitter, method='predict_proba')[:, 1]
    
    metrics = compute_metrics(y, y_pred_cv, y_prob_cv)
    
    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        elif isinstance(value, (int, float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    return metrics

if __name__ == "__main__":
    # Main execution pipeline for Random Forest model training and evaluation
    # Load dataset from CSV file
    df = pd.read_csv('data/processed/Cleaned_Telco_Customer.csv')
    # Split dataset into features (X) and target variable (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Strict Data Splitting: 90% main data, 10% unseen validation data
    X_main, X_unseen, y_main, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42)
    
    # From the 90%, split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.20, random_state=42)

    # Train Model
    print("Training Random Forest model...")
    rf_model = train_chosen_algorithm(X_train, y_train)

    # 10-Fold Cross-Validation on the training set
    print("Performing 10-fold cross-validation...")
    cv_metrics = evaluate_chosen_algorithm(rf_model, X_main, y_main)

    # Final Test Evaluation (on the 20% test set)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("\n--- 10-Fold CV Results (Mean) ---")
    for key, value in cv_metrics.items():
        print(f"{key.replace('test_', '').capitalize()}: {value:.4f}")

    print("\n--- Final Test Set Evaluation (20% Split) ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\nRandom Forest model training and evaluation completed.")