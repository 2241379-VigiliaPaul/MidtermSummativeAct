# decision_tree.py
from sklearn.tree import DecisionTreeClassifier, export_text
from src.evaluation import cross_validation_evaluation, save_evaluation_results
import numpy as np
import joblib
import os

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
    
    # Print results to console
    for metric, value in results.items():
        if metric == 'Confusion Matrix':
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
            
    return results

if __name__ == "__main__":
    # Placeholder for local testing. 
    # In practice, this logic will be triggered by your main.py pipeline.
    print("Decision Tree module ready.")