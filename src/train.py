import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_bank_data

def train_baseline(bank_id):
    print(f"\n=== Training baseline for Bank {bank_id} ===")
    
    # Load preprocessed data for this bank
    X_train, X_test, y_train, y_test = load_bank_data(bank_id)
    
    # Train logistic regression baseline
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model
    model_path = f"models/baseline_bank{bank_id}.pkl"
    joblib.dump(clf, model_path)
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    # Train baseline for all 3 bank shards
    for bank_id in range(1, 4):
        train_baseline(bank_id)
