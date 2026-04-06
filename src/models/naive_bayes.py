# Import necessary libraries for Naive Bayes modeling and data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# Define function to load and prepare training data
def load_and_prepare_data(file_path='data/processed/Cleaned_Telco_Customer.csv'):
    # Load cleaned dataset and split into train (80%), test (20%), and unseen validation (10%) sets
    df = pd.read_csv(file_path)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_dev, X_unseen, y_dev, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.20, random_state=42)
    return X_train, y_train, X_test, y_test, X_unseen, y_unseen


# Define function to train Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    # Train Gaussian Naive Bayes classifier - fast probabilistic model assuming feature independence
    model = GaussianNB()
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/naive_bayes.pkl')
    return model


# Define function to compute classification metrics
def compute_metrics(y_true, y_pred, y_prob=None):
    # Calculate comprehensive classification metrics including confusion matrix components and ROC-AUC
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


# Define function to evaluate Naive Bayes model
def evaluate_naive_bayes(model, X, y, cv=10):
    # Evaluate Naive Bayes model and return all performance metrics for given dataset
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return compute_metrics(y, y_pred, y_prob)


if __name__ == "__main__":
    # Main execution pipeline for Naive Bayes model training and evaluation
    # Load and prepare data
    X_train, y_train, X_test, y_test, X_unseen, y_unseen = load_and_prepare_data()
    # Train Naive Bayes model
    model = train_naive_bayes(X_train, y_train)
    test_results = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    unseen_results = compute_metrics(y_unseen, model.predict(X_unseen), model.predict_proba(X_unseen)[:, 1])
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=10).mean()

    # Display Report
    print(f"{'Metric':<15} | {'Testing Set (20%)':<18} | {'Unseen Data (10%)':<18}")
    print("-" * 55)
    for metric in test_results.keys():
        if metric == "Confusion Matrix":
            print(f"{metric:<15} | See below{' ' * 9} | See below")
            print(f"  Test CM:\n{test_results[metric]}")
            print(f"  Unseen CM:\n{unseen_results[metric]}")
        else:
            print(f"{metric:<15} | {test_results[metric]:<18.4f} | {unseen_results[metric]:<18.4f}")

    print(f"\nMean Cross-Validation Accuracy (Dev Set): {cv_accuracy:.4f}")

    # Plot Confusion Matrix for Unseen Data
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y_unseen, model.predict(X_unseen)), annot=True, fmt='d', cmap='Oranges')
    plt.title('Confusion Matrix: Performance on Unseen Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()