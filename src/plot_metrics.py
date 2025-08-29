# src/plot_metrics.py
import json
import matplotlib.pyplot as plt

LOG_FILE = "metrics_log.json"

def plot_metrics(log_file=LOG_FILE):
    with open(log_file, "r") as f:
        data = json.load(f)

    # Plot baseline
    baseline = data["baseline"]
    plt.figure(figsize=(8, 5))
    plt.bar(["Train", "Test"], [baseline["train_acc"], baseline["test_acc"]], color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.title("Baseline Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot local metrics for federated
    fed = data["federated"]
    local_train = [m["train_acc"] for m in fed["local_metrics"]]
    local_test = [m["test_acc"] for m in fed["local_metrics"]]
    x = range(1, len(local_train)+1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, local_train, marker='o', label="Local Train Acc")
    plt.plot(x, local_test, marker='x', label="Local Test Acc")
    plt.hlines(fed["global_acc"], 1, len(local_train), colors='r', linestyles='dashed', label="Global Acc")
    plt.title("Federated Training Accuracy")
    plt.xlabel("Bank / Client")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    # Plot local metrics for encrypted federated
    enc = data["encrypted_federated"]
    local_train = [m["train_acc"] for m in enc["local_metrics"]]
    local_test = [m["test_acc"] for m in enc["local_metrics"]]
    x = range(1, len(local_train)+1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, local_train, marker='o', label="Local Train Acc")
    plt.plot(x, local_test, marker='x', label="Local Test Acc")
    plt.hlines(enc["global_acc"], 1, len(local_train), colors='r', linestyles='dashed', label="Global Acc")
    plt.title("Encrypted Federated Training Accuracy")
    plt.xlabel("Bank / Client")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_metrics()
