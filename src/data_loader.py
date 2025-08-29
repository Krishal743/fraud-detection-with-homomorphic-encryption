# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_bank_data(bank_id, test_size=0.2, random_state=42):
    file_path = f"data/processed/bank{bank_id}.csv"
    df = pd.read_csv(file_path)

    # Drop only ID columns
    df = df.drop(columns=["nameOrig", "nameDest"], errors='ignore')

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Separate features and target
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
