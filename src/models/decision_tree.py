# decision_tree.py
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from src.evaluation import cross_validation_evaluation
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np

def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train Decision Tree model
    Setting max_depth keeps the extracted rules readable and prevents overfitting.
    """
    # Initialize DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    
    # Fit on training data
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/decision_tree.pkl')
    
    return model

def extract_decision_rules(model, feature_names, output_path=None):
    """
    Extracts and prints the text representation of the decision tree rules.
    Optionally saves rules to a text file.
    """
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
    """
    Evaluate Decision Tree using 10-fold cross-validation
    """
    print(f"\nEvaluating Decision Tree with {cv}-fold CV...")
    results = cross_validation_evaluation(model, X, y, cv=cv)
    
    # Print results to console with safe formatting by value type
    for metric, value in results.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        elif isinstance(value, (int, float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
            
    return results

def visualize_decision_tree(
    model,
    feature_names,
    class_names=("No Churn", "Churn"),
    output_path="results/decision_tree_simple.png",
    display_depth=3,
):
    """
    Save a simplified visual diagram of the trained decision tree as a PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Shorten very long engineered column names for readability in the graph
    short_names = [name.split("(")[0].strip() for name in feature_names]

    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=short_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        max_depth=display_depth,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Simplified decision tree image saved to: {output_path}")

if __name__ == "__main__":
    # Placeholder for local testing. 
    # In practice, this logic will be triggered by your main.py pipeline.
    print("Decision Tree module ready.")