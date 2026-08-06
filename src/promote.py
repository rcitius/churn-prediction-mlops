"""Promotion gate: make the newest churn model champion if it's good enough and beats the current one."""
import os
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "churn-classifier"
MIN_ROC_AUC = 0.75

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
client = MlflowClient()


def score_of(version):
    run = client.get_run(version.run_id)
    return run.data.metrics["roc_auc"]


def main():
    # 1. newest version = the candidate
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    candidate = max(versions, key=lambda v: int(v.version))
    cand_score = score_of(candidate)
    print(f"Candidate v{candidate.version}: roc_auc={cand_score:.4f}")

    # 2. current champion (or None if there isn't one yet)
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_score = score_of(champion)
    except mlflow.exceptions.MlflowException:
        champion = None

    # 3. decide
    if cand_score < MIN_ROC_AUC:
        print("Below quality bar. No change.")
    elif champion is None or cand_score > champ_score:
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        print(f"PROMOTED v{candidate.version} -> champion")
    else:
        print(f"Champion v{champion.version} stays.")


if __name__ == "__main__":
    main()