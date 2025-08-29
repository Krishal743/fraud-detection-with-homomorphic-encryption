# src/evaluate.py
import torch
import pandas as pd
from src.data_loader import load_bank_data

class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def evaluate(bank_id, model_path):
    X_train, X_test, y_train, y_test = load_bank_data(bank_id)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

    model = SimpleModel(X_test.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_labels = (y_pred >= 0.5).float()
        acc = (y_pred_labels == y_test).float().mean().item()
        print(f"Bank {bank_id} Accuracy: {acc:.4f}")
        return acc
