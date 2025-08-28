import boto3
import joblib
import json
from sklearn.metrics import accuracy_score
from src.data_loader import load_bank_data

# ==========================================================
# ✅ AWS Setup
# ----------------------------------------------------------
# Uses credentials from `aws configure`
# ==========================================================
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def explain_with_titan(transaction, prediction):
    """Send fraud prediction to Titan for explanation, return JSON."""
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

    result = json.loads(response["body"].read())
    return {
        "transaction": transaction,
        "prediction": prediction,
        "explanation": result["results"][0]["outputText"]
    }

def evaluate(bank_id=1, save_json="fraud_explanations.json"):
    print(f"\n=== Evaluating Bank {bank_id} model ===")
    
    # Load test data
    _, X_test, _, y_test = load_bank_data(bank_id)
    
    # Load trained model
    model_path = f"models/baseline_bank{bank_id}.pkl"
    clf = joblib.load(model_path)
    
    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Collect fraud explanations
    fraud_results = []
    for i, pred in enumerate(y_pred):
        if pred == 1:  # Fraudulent
            transaction = X_test.iloc[i].to_dict()
            fraud_results.append(explain_with_titan(transaction, "fraudulent"))
    
    # Save to JSON file (optional)
    if fraud_results:
        with open(save_json, "w") as f:
            json.dump(fraud_results, f, indent=4)
        print(f"\n💾 Saved explanations to {save_json}")
    
    return fraud_results

if __name__ == "__main__":
    results = evaluate(bank_id=1)
    print("\n=== JSON Output ===")
    print(json.dumps(results, indent=4))