# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_bank_data(bank_id, dataset_path="data/raw/PaySim.csv"):
    """
    Load and preprocess PaySim data shard for a given bank.
    Splits the dataset into train/test and returns X_train, X_test, y_train, y_test.
    """
    # Load full dataset
    df = pd.read_csv(dataset_path)

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define features and target
    feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']
    X = df[feature_cols]
    y = df['isFraud']

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # Split into train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split train data into 3 shards for 3 banks
    shard_size = len(X_train_full) // 3
    start_idx = (bank_id - 1) * shard_size
    end_idx = start_idx + shard_size if bank_id < 3 else len(X_train_full)

    X_train = X_train_full.iloc[start_idx:end_idx]
    y_train = y_train_full.iloc[start_idx:end_idx]

    return X_train, X_test, y_train, y_test
