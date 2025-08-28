import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_bank_data

def federated_training():
    num_banks = 3
    # Load first bank's data to initialize global model
    X_train, X_test, y_train, y_test = load_bank_data(1)

    # Initialize global model
    global_model = LogisticRegression(max_iter=1000)
    global_model.fit(X_train, y_train.values.ravel())  # dummy fit to set classes_

    # Placeholder: store aggregated coefficients and intercepts
    aggregated_coef = np.zeros_like(global_model.coef_)
    aggregated_intercept = np.zeros_like(global_model.intercept_)

    # Federated loop (simplified)
    for bank_id in range(1, num_banks + 1):
        X_bank, _, y_bank, _ = load_bank_data(bank_id)
        local_model = LogisticRegression(max_iter=1000)
        local_model.fit(X_bank, y_bank.values.ravel())
        # Aggregate weights (simple averaging)
        aggregated_coef += local_model.coef_ / num_banks
        aggregated_intercept += local_model.intercept_ / num_banks

    # Overwrite global model weights
    global_model.coef_ = aggregated_coef
    global_model.intercept_ = aggregated_intercept

    # Evaluate
    score = global_model.score(X_test, y_test.values.ravel())
    print(f"Global model accuracy: {score:.4f}")

if __name__ == "__main__":
    federated_training()
