import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def specificity_score(y_true, y_pred):
    """
    Manually compute specificity: TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def train_chosen_algorithm(X_train, y_train):
    """
    Initialize and train the Random Forest model
    """
    # Using 100 trees as a standard robust baseline
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model for future validation
    joblib.dump(model, 'random_forest_model.pkl')
    return model

def evaluate_chosen_algorithm(model, X, y, cv=10):
    """
    Evaluate using 10-fold cross-validation
    Returns mean scores for all required metrics
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'specificity': make_scorer(specificity_score)
    }
    
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Extract the mean of each metric
    metrics = {metric: np.mean(scores) for metric, scores in results.items() if 'test_' in metric}
    return metrics

if __name__ == "__main__":
    # 1. Load the preprocessed dataset
    # Make sure this filename matches your cleaned file in VS Code
    df = pd.read_csv('Cleaned_Telco_Customer.csv')
    
    # Define features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. Strict Data Splitting
    # Split A: 90% main data, 10% unseen validation data
    X_main, X_unseen, y_main, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42)
    
    # Split B: From the 90%, split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.20, random_state=42)

    # 3. Train the Model
    print("Training Random Forest model...")
    rf_model = train_chosen_algorithm(X_train, y_train)

    # 4. 10-Fold Cross-Validation on the training set
    print("Performing 10-fold cross-validation...")
    cv_metrics = evaluate_chosen_algorithm(rf_model, X_main, y_main)

    # 5. Final Test Evaluation (on the 20% test set)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("\n--- 10-Fold CV Results (Mean) ---")
    for key, value in cv_metrics.items():
        print(f"{key.replace('test_', '').capitalize()}: {value:.4f}")

    print("\n--- Final Test Set Evaluation (20% Split) ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\nRandom Forest model training and evaluation completed.")