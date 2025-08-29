# federated_train_simple.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import load_bank_data
from src.metrics_logger import MetricsLogger
import pandas as pd

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def federated_training(epochs=5, lr=0.01):
    datasets = []
    input_dim = None

    # Load datasets for 3 banks
    for bank_id in range(1, 4):
        X_train, X_test, y_train, y_test = load_bank_data(bank_id)

        # Convert all to numeric and handle NaNs
        X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        if input_dim is None:
            input_dim = X_train.shape[1]

        datasets.append({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        })

    # Initialize global model
    global_model = SimpleModel(input_dim)
    criterion = nn.BCELoss()
    logger = MetricsLogger()

    for epoch in range(1, epochs+1):
        local_losses = []
        local_models = []

        # Train each bank locally
        for data in datasets:
            model = SimpleModel(input_dim)
            model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            optimizer.zero_grad()
            y_pred = model(data["X_train"])
            loss = criterion(y_pred, data["y_train"])
            loss.backward()
            optimizer.step()

            local_losses.append(loss.item())
            local_models.append(model.state_dict())

        # Federated averaging
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = sum(local_model[key] for local_model in local_models) / len(local_models)
        global_model.load_state_dict(global_dict)

        # Log metrics
        logger.log(epoch, local_losses, global_model, datasets)

    # Save model and metrics
    torch.save(global_model.state_dict(), "models/federated_simple_model.pt")
    logger.save("metrics/federated_simple_metrics.csv")
    print("✅ Non-encrypted federated training complete!")

if __name__ == "__main__":
    federated_training(epochs=5, lr=0.01)
