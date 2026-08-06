"""Load the champion churn model and score a few customers."""
import os
import mlflow
from src.data import load_data

MODEL_NAME = "churn-classifier"


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    # load the version promote.py marked as champion
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    # same split + encoding the model was trained on
    X_train, X_test, y_train, y_test = load_data()

    # score the first 5 customers from the test set
    customers = X_test.head(5)
    actual = y_test.head(5)

    predictions = model.predict(customers)
    probabilities = model.predict_proba(customers)[:, 1]

    for i in range(5):
        print(f"customer {i}: predicted={predictions[i]}  churn_prob={probabilities[i]:.3f}  actual={actual.iloc[i]}")


if __name__ == "__main__":
    main()