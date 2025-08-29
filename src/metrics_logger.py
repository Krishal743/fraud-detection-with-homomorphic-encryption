# src/metrics_logger.py
import pandas as pd
import torch
import torch.nn as nn

class MetricsLogger:
    def __init__(self):
        self.records = []

    def log(self, epoch, local_losses, global_model, datasets):
        test_accuracies = []
        loss_fn = nn.BCELoss()
        for data in datasets:
            X_test = data["X_test"]
            y_test = data["y_test"]

            with torch.no_grad():
                y_pred = global_model(X_test)
                y_pred_labels = (y_pred >= 0.5).float()
                acc = (y_pred_labels == y_test).float().mean().item()
                test_accuracies.append(acc)

        self.records.append({
            "epoch": epoch,
            **{f"bank{i+1}_local_loss": local_losses[i] for i in range(len(local_losses))},
            **{f"bank{i+1}_test_acc": test_accuracies[i] for i in range(len(test_accuracies))}
        })

        # Print summary per epoch
        loss_summary = ", ".join([f"{l:.4f}" for l in local_losses])
        acc_summary = ", ".join([f"{a:.4f}" for a in test_accuracies])
        print(f"[Epoch {epoch}] Local Losses: {loss_summary} | Test Accuracies: {acc_summary}")

    def save(self, filename="metrics.csv"):
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f"✅ Metrics saved to {filename}")
