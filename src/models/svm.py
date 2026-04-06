# Support Vector Machine Model Implementation
# SVM classifier for customer churn prediction

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import pandas as pd

def load_cleaned_data(filepath="data/processed/Cleaned_Telco_Customer.csv"):
    """
    Load the cleaned Telco customer dataset and return features/target.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data file not found: {filepath}")

    df = pd.read_csv(filepath)
    if 'Churn' not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in cleaned data.")

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


def train_svm(X_train, y_train):
    """
    Train SVM model
    """
    # Initialize SVC with probability=True for ROC-AUC
    model = SVC(random_state=42, probability=True)
    
    # Fit on training data
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/svm.pkl')
    
    return model

def evaluate_svm(model, X, y, cv=10):
    """
    Evaluate SVM using 10-fold cross-validation
    Returns all required metrics
    """
    from src.evaluation import cross_validation_evaluation
    
    print(f"\nEvaluating SVM with {cv}-fold CV...")
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

if __name__ == "__main__":
    from src.evaluation import save_evaluation_results

    data_path = os.path.join('data', 'processed', 'Cleaned_Telco_Customer.csv')
    X, y = load_cleaned_data(data_path)

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess data: apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_svm(X_train_scaled, y_train)
    
    # Evaluate and save results
    results = evaluate_svm(model, X_train_scaled, y_train, cv=10)
    save_evaluation_results(results, 'SVM', 'results/svm_evaluation.txt')
    
    print("SVM model training and evaluation completed")