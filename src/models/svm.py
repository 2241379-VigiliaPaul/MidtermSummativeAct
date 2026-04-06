# Import necessary libraries for SVM modeling and evaluation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import pandas as pd

def load_cleaned_data(filepath="data/processed/Cleaned_Telco_Customer.csv"):
    # Load preprocessed dataset and validate that target column 'Churn' exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'Churn' not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in cleaned data.")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def train_svm(X_train, y_train, random_state=42):
    # Train Support Vector Machine classifier with class balancing for imbalanced churn dataset
    model = SVC(random_state=random_state, probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/svm.pkl')
    return model

def compute_metrics(y_true, y_pred, y_prob=None):
    # Compute classification metrics including specificity (TN/(TN+FP)) and ROC-AUC score
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

def evaluate_svm(model, X, y, cv=10):
    # Evaluate SVM using stratified cross-validation to ensure balanced class representation in each fold
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation import cross_validation_evaluation
    print(f"\nEvaluating SVM with {cv}-fold CV...")
    results = cross_validation_evaluation(model, X, y, cv=cv)
    for metric, value in results.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        elif isinstance(value, (int, float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    return results

if __name__ == "__main__":
    # Main execution pipeline for SVM model training and evaluation
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation import compute_all_metrics, specificity_score
    
    # Load and split data
    data_path = os.path.join('data', 'processed', 'Cleaned_Telco_Customer.csv')
    X, y = load_cleaned_data(data_path)
    X_main, X_unseen, y_main, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.20, random_state=42, stratify=y_main)

    # Preprocess and train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unseen_scaled = scaler.transform(X_unseen)
    print("Training SVM model...")
    model = train_svm(X_train_scaled, y_train)

    # Evaluate model performance
    print("Performing 10-fold cross-validation...")
    cv_results = evaluate_svm(model, X_train_scaled, y_train, cv=10)
    
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    test_results = compute_all_metrics(y_test, y_test_pred, y_test_prob)
    
    y_unseen_pred = model.predict(X_unseen_scaled)
    y_unseen_prob = model.predict_proba(X_unseen_scaled)[:, 1]
    unseen_results = compute_all_metrics(y_unseen, y_unseen_pred, y_unseen_prob)

    # Display results
    print("\n--- 10-Fold CV Results ---")
    for metric, value in cv_results.items():
        if metric not in ["Confusion Matrix", "CV Folds", "Fold Accuracies"]:
            print(f"{metric}: {value:.4f}")
    
    print("\n--- Test Set Evaluation (20%) ---")
    for metric, value in test_results.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\n--- Unseen Data Evaluation (10%) ---")
    for metric, value in unseen_results.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nSVM model training and evaluation completed")