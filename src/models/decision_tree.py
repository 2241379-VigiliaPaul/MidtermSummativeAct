from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np


def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/decision_tree.pkl')
    
    return model


def compute_metrics(y_true, y_pred, y_prob=None):
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


def extract_decision_rules(model, feature_names, output_path=None):
    tree_rules = export_text(model, feature_names=list(feature_names))
    print("\n--- Extracted Decision Tree Rules ---")
    print(tree_rules)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- Extracted Decision Tree Rules ---\n")
            f.write(tree_rules)
        print(f"Rules saved to: {output_path}")
    return tree_rules

def evaluate_decision_tree(model, X, y, cv=10):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation import cross_validation_evaluation
    
    print(f"\nEvaluating Decision Tree with {cv}-fold CV...")
    results = cross_validation_evaluation(model, X, y, cv=cv)
    
    for metric, value in results.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        elif isinstance(value, (int, float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    return results

def visualize_decision_tree(model, feature_names, class_names=("No Churn", "Churn"), output_path="results/decision_tree_simple.png", display_depth=3):
    # Create and save simplified visual diagram of decision tree
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    short_names = [name.split("(")[0].strip() for name in feature_names]
    plt.figure(figsize=(24, 12))
    plot_tree(model, feature_names=short_names, class_names=list(class_names), filled=True, rounded=True, max_depth=display_depth, impurity=False, proportion=True, precision=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Simplified decision tree image saved to: {output_path}")

if __name__ == "__main__":
    # Main execution pipeline for Decision Tree model
    import sys, pandas as pd
    from sklearn.model_selection import train_test_split
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation import compute_all_metrics, specificity_score
    
    # Load and prepare data
    data_path = os.path.join('data', 'processed', 'Cleaned_Telco_Customer.csv')
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    feature_names = X.columns.tolist()

    # Data splitting: 90/10, then 80/20
    X_main, X_unseen, y_main, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.20, random_state=42, stratify=y_main)

    # Train model and extract rules
    print("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    extract_decision_rules(dt_model, feature_names, 'results/decision_tree_rules.txt')
    visualize_decision_tree(dt_model, feature_names, 'results/decision_tree_visualization.png')

    # Evaluate model performance
    print("Performing 10-fold cross-validation...")
    cv_results = evaluate_decision_tree(dt_model, X_train, y_train, cv=10)
    
    y_test_pred = dt_model.predict(X_test)
    y_test_prob = dt_model.predict_proba(X_test)[:, 1]
    test_results = compute_all_metrics(y_test, y_test_pred, y_test_prob)
    
    y_unseen_pred = dt_model.predict(X_unseen)
    y_unseen_prob = dt_model.predict_proba(X_unseen)[:, 1]
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
    
    print("\nDecision Tree model training and evaluation completed")