import boto3
import joblib
import json
from sklearn.metrics import accuracy_score
from src.data_loader import load_bank_data

# ==========================================================
# ✅ AWS Setup
# ----------------------------------------------------------
# No need to put access/secret keys here if you've run `aws configure`.
# boto3 will automatically read from ~/.aws/credentials
# ==========================================================
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def explain_with_titan(transaction, prediction):
    """Send fraud prediction to Titan for explanation."""
    prompt = f"""
    A fraud detection model flagged this transaction as {prediction}.
    Features (scaled): {transaction}

    Please explain in simple terms why the model may have considered this suspicious.
    """

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0.7,
            "maxTokenCount": 256,
            "topP": 0.9
        }
    })

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        body=body.encode("utf-8")
    )

    # Titan returns a streaming body → decode JSON
    result = json.loads(response["body"].read())
    explanation = result["results"][0]["outputText"]

    print("\n[Titan Explanation]")
    print(explanation)

def evaluate(bank_id=1):
    print(f"\n=== Evaluating Bank {bank_id} model ===")
    
    # Load test data
    X_train, X_test, y_train, y_test = load_bank_data(bank_id)
    
    # Load trained model
    model_path = f"models/baseline_bank{bank_id}.pkl"
    clf = joblib.load(model_path)
    
    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Go through test set and call Titan for flagged transactions
    for i, pred in enumerate(y_pred):
        if pred == 1:  # Fraudulent
            transaction = X_test.iloc[i].to_dict()
            print(f"\n🚨 Fraudulent transaction detected: {transaction}")
            explain_with_titan(transaction, "fraudulent")

if __name__ == "__main__":
    evaluate(bank_id=1)
