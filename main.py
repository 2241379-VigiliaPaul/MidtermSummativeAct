# Main Script
# Orchestrates the entire customer churn prediction pipeline

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_data
from src.evaluation import save_evaluation_results
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
        # Use read_excel for .xlsx files
        df = pd.read_excel("data/splits/Telco_Customer.xlsx")
    except FileNotFoundError:
        print("Error: Could not find dataset at data/splits/Telco_Customer.xlsx")
        return
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return

    # Preprocess so model gets numeric inputs
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
    dt_model = train_decision_tree(X_train, y_train)

    print("\nSaving decision tree image...")
    visualize_decision_tree(
        dt_model,
        feature_names=X_train.columns,
        class_names=("No Churn", "Churn"),
        output_path="results/decision_tree.png",
    )

    print("\nExtracting decision rules...")
    extract_decision_rules(
        dt_model,
        feature_names=X_train.columns,
        output_path="results/decision_tree_rules.txt",
    )

    print("\nEvaluating Decision Tree with 10-fold CV...")
    dt_metrics = evaluate_decision_tree(dt_model, X_train, y_train, cv=10)

    # Save evaluation output
    save_evaluation_results(
        dt_metrics,
        model_name="Decision Tree",
        filepath="results/decision_tree_evaluation.txt",
    )

    print("\nPipeline completed successfully")


if __name__ == "__main__":
    main()