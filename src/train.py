# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import load_bank_data
import pandas as pd
from src.metrics_logger import MetricsLogger

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def train_baseline(bank_id, epochs=5, lr=0.01):
    # Load bank data
    X_train, X_test, y_train, y_test = load_bank_data(bank_id)
    input_dim = X_train.shape[1]

    # Convert categorical/string columns to numeric if any
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

    # Initialize model, optimizer, loss
    model = SimpleModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Initialize logger
    logger = MetricsLogger()

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Log metrics
        logger.log(epoch, [loss.item()], model, [{"X_test": X_test, "y_test": y_test}])

        print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    # Save model and metrics after training
    model_path = f"models/baseline_bank{bank_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.save(f"metrics/baseline_bank{bank_id}_metrics.csv")
    print(f"✅ Baseline model and metrics for Bank {bank_id} saved!")

    return model

if __name__ == "__main__":
    # Train Bank 1
    train_baseline(bank_id=1, epochs=5, lr=0.01)
    
    # Train Bank 2
    train_baseline(bank_id=2, epochs=5, lr=0.01)
    
    # Train Bank 3
    train_baseline(bank_id=3, epochs=5, lr=0.01)

