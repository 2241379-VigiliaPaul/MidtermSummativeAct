# Naive Bayes Model Implementation
# Probabilistic classifier for customer churn prediction

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

# Load and Preprocess the Dataset
file_path = '/content/Telco_Customer - Telco-Customer-Churn-Preprocessed.csv'
df = pd.read_csv(file_path)

# Do Further Data Cleaning
df = df.drop(columns=['customerID'])
pm_col = 'PaymentMethod (eCheck = 0, mCheck = 1, bTransfer =3, card = 4)'
df[pm_col] = df[pm_col].astype(str).str[0].astype(int)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 10% unnseen data
X_dev, X_unseen, y_dev, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42)

# 80% and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.20, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Helper function to compute all requested metrics
def compute_performance(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

# Evaluate on Testing Set
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]
test_results = compute_performance(y_test, y_test_pred, y_test_prob)

# Evaluate Unseen Data
y_unseen_pred = model.predict(X_unseen)
y_unseen_prob = model.predict_proba(X_unseen)[:, 1]
unseen_results = compute_performance(y_unseen, y_unseen_pred, y_unseen_prob)

cv_accuracy = cross_val_score(model, X_dev, y_dev, cv=5).mean()

# Display Report
print(f"{'Metric':<15} | {'Testing Set (20%)':<18} | {'Unseen Data (10%)':<18}")
print("-" * 55)
for metric in test_results.keys():
    print(f"{metric:<15} | {test_results[metric]:<18.4f} | {unseen_results[metric]:<18.4f}")

print(f"\nMean Cross-Validation Accuracy (Dev Set): {cv_accuracy:.4f}")

# Plot Confusion Matrix for Unseen Data
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_unseen, y_unseen_pred), annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix: Performance on Unseen Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

def specificity_score(y_true, y_pred):
    """
    Manually compute specificity
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def train_naive_bayes(X_train, y_train):
    """
    Train Naive Bayes model
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_naive_bayes(model, X, y, cv=10):
    """
    Evaluate Naive Bayes using 10-fold cross-validation
    Returns all required metrics
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return compute_performance(y, y_pred, y_prob)

if __name__ == "__main__":
    # Main execution for Naive Bayes
    # Load data splits
    # Train model
    model = train_naive_bayes(X_train, y_train)
    # Evaluate and save results
    results = evaluate_naive_bayes(model, X_test, y_test)
    print("Naive Bayes model training and evaluation completed")