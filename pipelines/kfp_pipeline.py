"""KFP pipeline: load the data, then train and register the model."""
from pathlib import Path

from kfp import dsl, local

# absolute, because components run in a temp directory — a relative
# sqlite path would create a throwaway db there
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"


@dsl.component
def load_data_step(dataset: dsl.Output[dsl.Dataset]):
    import joblib
    from src.data import load_data

    splits = load_data()
    joblib.dump(splits, dataset.path)


@dsl.component
def train_step(dataset: dsl.Input[dsl.Dataset], tracking_uri: str):
    import os
    import joblib
    from src.train import train_and_register

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    X_train, X_test, y_train, y_test = joblib.load(dataset.path)
    train_and_register(X_train, X_test, y_train, y_test)


@dsl.pipeline(name="churn-pipeline")
def churn_pipeline(tracking_uri: str = TRACKING_URI):
    load_task = load_data_step()
    train_step(dataset=load_task.outputs["dataset"], tracking_uri=tracking_uri)


if __name__ == "__main__":
    local.init(runner=local.SubprocessRunner(use_venv=False))
    churn_pipeline()