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
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Clean and preprocess the data
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features if needed
    """
    # Drop customerID
    df = df.drop(columns=['customerID'])
    
    # Convert TotalCharges to numeric, fill NaN with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Encode categorical variables
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    
    # Encode service columns based on InternetService
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df.apply(lambda row: 2 if row['InternetService'] == 0 else (1 if row[col] == 'Yes' else 0), axis=1)
    
    df['Contract'] = df['Contract'].map({'One year': 1, 'Two year': 2, 'Month-to-month': 3})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Rename columns to match cleaned dataset
    df = df.rename(columns={
        'gender': 'gender( m = 1, f = 0)',
        'SeniorCitizen': 'SeniorCitizen (Y=1, N=0)',
        'Partner': 'Partner (Y=1, N=0)',
        'Dependents': 'Dependents (Y=1, N=0)',
        'PhoneService': 'PhoneService (Y=1, N=0)',
        'MultipleLines': 'MultipleLines (Y=1, N=0 , NS = 2)',
        'InternetService': 'InternetService ( N = 0, DSL = 1, FBR OPT. = 2',
        'OnlineSecurity': 'OnlineSecurity ( 0 internet serv. =2)',
        'OnlineBackup': 'OnlineBackup ( 0 internet serv. =2)',
        'DeviceProtection': 'DeviceProtection ( 0 internet serv. =2)',
        'TechSupport': 'TechSupport ( 0 internet serv. =2)',
        'StreamingTV': 'StreamingTV ( 0 internet serv. =2)',
        'StreamingMovies': 'StreamingMovies ( 0 internet serv. =2)',
        'Contract': 'Contract ( 1yr = 1, 2 yr = 2 , month to month = 3',
        'PaperlessBilling': 'PaperlessBilling ( Y = 1, N = 0)',
        'PaymentMethod': 'PaymentMethod (eCheck = 0, mCheck = 1, bTransfer =2, card = 3)'
    })
    
    return df

def split_data(df, target_column='Churn'):
    """
    Split data according to requirements:
    - 90% for training/testing (80% train, 20% test within this)
    - 10% unseen validation
    """
    # First split: 6339 train_test, 704 validation
    train_test, validation = train_test_split(
        df, test_size=704, random_state=42, stratify=df[target_column]
    )
    
    # Second split: 5071 train, 1268 test from train_test
    train, test = train_test_split(
        train_test, test_size=1268, random_state=42, stratify=train_test[target_column]
    )
    
    # Save splits to data/splits/
    os.makedirs('data/splits', exist_ok=True)
    train.to_csv('data/splits/train.csv', index=False)
    test.to_csv('data/splits/test.csv', index=False)
    validation.to_csv('data/splits/validation.csv', index=False)
    
    print("Data splitting completed:")
    print(f"Train: {len(train)} samples")
    print(f"Test: {len(test)} samples")
    print(f"Validation: {len(validation)} samples")
    
    return train, test, validation

if __name__ == "__main__":
    # Load raw data, preprocess, save cleaned, and split
    raw_data_path = "data/raw/Raw_Telco_Customer.csv"
    cleaned_data_path = "data/processed/Cleaned_Telco_Customer.csv"
    
    df = load_data(raw_data_path)
    df_processed = preprocess_data(df)
    df_processed.to_csv(cleaned_data_path, index=False)
    print("Preprocessing completed and saved to", cleaned_data_path)
    
    split_data(df_processed)
    
    print("Data preprocessing completed")