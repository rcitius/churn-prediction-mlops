"""Load and split the churn dataset."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
TARGET = "Churn"
RANDOM_STATE = 42


def load_data(test_size=0.2):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)