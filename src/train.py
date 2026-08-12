"""Train a churn classifier and register it to MLflow."""
import os

import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from src.data import load_data


def train_and_register(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("churn")

    with mlflow.start_run():
        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        roc_auc = roc_auc_score(y_test, proba)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"ROC-AUC: {roc_auc:.4f}  Recall: {recall:.4f}  Precision: {precision:.4f}  F1: {f1:.4f}")

        mlflow.log_metrics({"roc_auc": roc_auc, "recall": recall,
                            "precision": precision, "f1": f1})
        mlflow.sklearn.log_model(model, name="model",
                                 registered_model_name="churn-classifier",
                                 input_example=X_test[:5])
        return roc_auc

def main():
    X_train, X_test, y_train, y_test = load_data()
    train_and_register(X_train, X_test, y_train, y_test)
    


if __name__ == "__main__":
    main()