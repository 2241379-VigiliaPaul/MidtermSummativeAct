# Main Script
# Orchestrates the entire customer churn prediction pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_data
from src.evaluation import (
    save_evaluation_results,
    evaluate_on_dataset,
    save_test_vs_unseen_results,
    save_confusion_matrix_heatmap,
)
from src.models.decision_tree import (
    train_decision_tree,
    extract_decision_rules,
    evaluate_decision_tree,
    visualize_decision_tree,
)


def main():
    print("Starting Customer Churn Prediction Pipeline")

    print("Loading data...")
    try:
        df = pd.read_excel("data/splits/Telco_Customer.xlsx")
    except FileNotFoundError:
        print("Error: Could not find dataset at data/splits/Telco_Customer.xlsx")
        return
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return

    try:
        df = preprocess_data(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    if "Churn" not in df.columns:
        print("Error: Target column 'Churn' not found after preprocessing.")
        return

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Splitting data according to requirements...")
    X_temp, X_unseen, y_temp, y_unseen = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
    )

    print(
        f"Data ready. Training samples: {X_train.shape}, "
        f"Testing samples: {X_test.shape}, Unseen samples: {X_unseen.shape}"
    )

    print("\nTraining Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train, max_depth=4)

    print("\nSaving decision tree images...")
    visualize_decision_tree(
        dt_model,
        feature_names=X_train.columns,
        class_names=("No Churn", "Churn"),
        output_path="results/decision_tree_simple.png",
        display_depth=3,
    )

    visualize_decision_tree(
        dt_model,
        feature_names=X_train.columns,
        class_names=("No Churn", "Churn"),
        output_path="results/decision_tree_full.png",
        display_depth=10,
    )

    print("\nExtracting decision rules...")
    extract_decision_rules(
        dt_model,
        feature_names=X_train.columns,
        output_path="results/decision_tree_rules.txt",
    )

    print("\nEvaluating Decision Tree with 10-fold CV...")
    dt_metrics = evaluate_decision_tree(dt_model, X_train, y_train, cv=10)

    save_evaluation_results(
        dt_metrics,
        model_name="Decision Tree",
        filepath="results/decision_tree_evaluation.txt",
    )

    print("\nEvaluating Testing Set (20%) and Unseen Data (10%)...")
    test_metrics = evaluate_on_dataset(dt_model, X_test, y_test)
    unseen_metrics = evaluate_on_dataset(dt_model, X_unseen, y_unseen)

    save_test_vs_unseen_results(
        test_metrics=test_metrics,
        unseen_metrics=unseen_metrics,
        filepath="results/decision_tree_test20_vs_unseen10.csv",
    )

    y_test_pred = dt_model.predict(X_test)
    y_unseen_pred = dt_model.predict(X_unseen)

    # Combine 20% test + 10% unseen into one overall confusion matrix
    y_all_true = np.concatenate([y_test.to_numpy(), y_unseen.to_numpy()])
    y_all_pred = np.concatenate([y_test_pred, y_unseen_pred])

    save_confusion_matrix_heatmap(
        y_true=y_all_true,
        y_pred=y_all_pred,
        filepath="results/confusion_matrix_whole.png",
        title="Confusion Matrix: Overall Holdout Data (20% Test + 10% Unseen)",
        labels=("No Churn", "Churn"),
        cmap="Oranges",
    )

    print("\nPipeline completed successfully")


if __name__ == "__main__":
    main()