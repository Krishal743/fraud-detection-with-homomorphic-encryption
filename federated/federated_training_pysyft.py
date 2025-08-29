# federated/federated_train_encrypted.py
import torch
import torch.nn as nn
import torch.optim as optim
import syft as sy
from src.data_loader import load_bank_data
from src.metrics_logger import MetricsLogger
from src.train import SimpleModel

hook = sy.TorchHook(torch)  # required for PySyft
banks = [sy.VirtualWorker(hook, id=f"bank{i}") for i in range(1, 4)]

def train_encrypted_federated(epochs=5, lr=0.01):
    datasets = []
    for i, bank in enumerate(banks, start=1):
        X_train, X_test, y_train, y_test = load_bank_data(i)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

        # Send data to virtual worker
        datasets.append({
            "X_train": X_train.send(bank),
            "y_train": y_train.send(bank),
            "X_test": X_test,  # keep locally for evaluation
            "y_test": y_test
        })

    input_dim = datasets[0]["X_train"].shape[1]
    global_model = SimpleModel(input_dim)
    logger = MetricsLogger()

    for epoch in range(1, epochs+1):
        local_losses = []
        local_params = []

        # Local training
        for data in datasets:
            model = SimpleModel(input_dim)
            model.load_state_dict(global_model.state_dict())
            model.send(data["X_train"].location)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()

            model.train()
            optimizer.zero_grad()
            y_pred = model(data["X_train"])
            loss = criterion(y_pred, data["y_train"])
            loss.backward()
            optimizer.step()

            local_losses.append(loss.get().item())  # get back the loss
            local_params.append({k: v.get() for k, v in model.state_dict().items()})

        # FedAvg
        new_state_dict = {}
        for key in global_model.state_dict().keys():
            new_state_dict[key] = sum([p[key] for p in local_params]) / len(local_params)
        global_model.load_state_dict(new_state_dict)

        # Log metrics using local test sets
        logger.log(epoch, local_losses, global_model, [{"X_test": d["X_test"], "y_test": d["y_test"]} for d in datasets])

    # Save final global model
    torch.save(global_model.state_dict(), "models/fed_encrypted_global.pt")
    logger.save("metrics/fed_encrypted_metrics.csv")
    print("✅ Encrypted Federated Training Completed")

if __name__ == "__main__":
    train_encrypted_federated()
