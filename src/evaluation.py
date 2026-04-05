# evaluation.py
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    Perform cross-validation and compute metrics.
    Also reports fold-level accuracy so CV application is explicit.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    print(f"Applying StratifiedKFold with {splitter.get_n_splits()} folds...")

    y_pred_cv = cross_val_predict(model, X, y, cv=splitter)

    try:
        y_prob_cv = cross_val_predict(model, X, y, cv=splitter, method="predict_proba")[:, 1]
    except AttributeError:
        y_prob_cv = None

    fold_scores = cross_validate(
        model,
        X,
        y,
        cv=splitter,
        scoring="accuracy",
        return_train_score=False,
    )["test_score"]

    metrics = compute_all_metrics(y, y_pred_cv, y_prob_cv)
    metrics["CV Folds"] = cv
    metrics["Fold Accuracies"] = [round(x, 4) for x in fold_scores.tolist()]
    metrics["Fold Accuracy Mean"] = float(np.mean(fold_scores))
    metrics["Fold Accuracy Std"] = float(np.std(fold_scores))
    return metrics

def save_evaluation_results(results, model_name, filepath):
    """
    Save evaluation results to file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"--- Evaluation Results: {model_name} ---\n")
        for metric_name, value in results.items():
            if metric_name == "Confusion Matrix":
                f.write(f"{metric_name}:\n{value}\n")
            elif isinstance(value, (float, np.floating)):
                f.write(f"{metric_name}: {value:.4f}\n")
            else:
                f.write(f"{metric_name}: {value}\n")

def evaluate_on_dataset(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except AttributeError:
        y_prob = None
    return compute_all_metrics(y, y_pred, y_prob)


def save_holdout_comparison(metrics_20, metrics_10, filepath):
    """
    Save a side by side metrics table:
    Metric | Training Set 20% | Unseen Data (10%)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    metric_order = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Specificity",
        "ROC-AUC",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Metric,Training Set 20%,Unseen Data (10%)\n")
        for m in metric_order:
            left = metrics_20.get(m, "N/A")
            right = metrics_10.get(m, "N/A")

            if isinstance(left, (float, np.floating)):
                left = f"{left:.4f}"
            if isinstance(right, (float, np.floating)):
                right = f"{right:.4f}"

            f.write(f"{m},{left},{right}\n")

def save_test_vs_unseen_results(test_metrics, unseen_metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    metric_order = ["Accuracy", "Precision", "Recall", "F1-score", "Specificity", "ROC-AUC"]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Metric,Testing Set (20%),Unseen Data (10%)\n")
        for metric in metric_order:
            left = test_metrics.get(metric, "N/A")
            right = unseen_metrics.get(metric, "N/A")

            if isinstance(left, (float, np.floating)):
                left = f"{left:.4f}"
            if isinstance(right, (float, np.floating)):
                right = f"{right:.4f}"

            f.write(f"{metric},{left},{right}\n")

def save_confusion_matrix_pair_heatmap(
    y_test_true,
    y_test_pred,
    y_unseen_true,
    y_unseen_pred,
    filepath="results/confusion_matrix_pair.png",
    labels=("No Churn", "Churn"),
):
    """
    Save side-by-side confusion matrix heatmaps for:
    - Testing Set (20%)
    - Unseen Data (10%)
    """
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    cm_unseen = confusion_matrix(y_unseen_true, y_unseen_pred)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Testing Set (20%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_unseen,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("Unseen Data (10%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("Decision Tree Confusion Matrices", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Confusion matrix pair heatmap saved to: {filepath}")

def save_confusion_matrix_heatmap(
    y_true,
    y_pred,
    filepath="results/confusion_matrix_whole.png",
    title="Confusion Matrix: Testing (20%) + Unseen (10%)",
    labels=("No Churn", "Churn"),
    cmap="Oranges",
):
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,  # color measure on the side
        cbar_kws={"label": "Number of Samples"},
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Whole confusion matrix heatmap saved to: {filepath}")

if __name__ == "__main__":
    print("Evaluation utilities loaded")