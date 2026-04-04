# Data Preprocessing Script
# Handles loading, cleaning, and splitting the Telco Customer Churn dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """
    Load the Telco Customer Churn dataset
    """
    # TODO: Implement data loading
    # df = pd.read_csv(filepath)
    # return df
    pass

def preprocess_data(df):
    """
    Clean and preprocess the data
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features if needed
    """
    # TODO: Implement preprocessing steps
    # - Convert 'TotalCharges' to numeric
    # - Handle missing values
    # - Encode categorical variables
    # - etc.
    pass

def split_data(df, target_column='Churn'):
    """
    Split data according to requirements:
    - 90% for training/testing (80% train, 20% test within this)
    - 10% unseen validation
    """
    # TODO: Implement data splitting
    # First split: 90% train_test, 10% validation
    # Second split: 80% train, 20% test from train_test
    pass

if __name__ == "__main__":
    # TODO: Main execution
    # raw_data_path = "data/raw/Telco-Customer-Churn.csv"
    # processed_data_path = "data/processed/"
    # splits_path = "data/splits/"
    
    # df = load_data(raw_data_path)
    # df_processed = preprocess_data(df)
    # split_data(df_processed)
    
    print("Data preprocessing completed")