"""Basic checks for the churn pipeline."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data import load_data
from src.monitor import find_drifted


def test_data_split_is_sane():
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "Churn" not in X_train.columns       # target must not leak into features
    assert set(y_train.unique()) == {0, 1}      # target is 0/1


def test_model_beats_auc_floor():
    X_train, X_test, y_train, y_test = load_data()

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    assert auc > 0.75


def test_monitor_flags_only_the_drifted_column():
    reference = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "MonthlyCharges": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    })
    current = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "MonthlyCharges": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    })

    drifted = find_drifted(reference, current, ["tenure", "MonthlyCharges"])

    assert drifted == ["MonthlyCharges"]