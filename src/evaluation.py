# evaluation.py
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
import os

def specificity_score(y_true, y_pred):
    """
    Manually compute specificity: TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Handle edge case where the denominator is 0
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all required evaluation metrics
    """
    metrics = {
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity_score(y_true, y_pred)
    }
    
    # ROC-AUC requires probability scores instead of discrete predictions
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['ROC-AUC'] = "N/A (No probabilities provided)"
        
    return metrics

def cross_validation_evaluation(model, X, y, cv=10):
    """
    Perform 10-fold cross-validation and compute metrics
    """
    # Get discrete predictions for standard metrics
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    # Get probability predictions for ROC-AUC
    try:
        y_prob_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    except AttributeError:
        # Fallback if a model doesn't support predict_proba
        y_prob_cv = None

    metrics = compute_all_metrics(y, y_pred_cv, y_prob_cv)
    return metrics

def save_evaluation_results(results, model_name, filepath):
    """
    Save evaluation results to file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"--- Evaluation Results: {model_name} ---\n")
        for metric_name, value in results.items():
            if metric_name == 'Confusion Matrix':
                f.write(f"{metric_name}:\n{value}\n")
            else:
                f.write(f"{metric_name}: {value:.4f}\n")

if __name__ == "__main__":
    print("Evaluation utilities loaded")