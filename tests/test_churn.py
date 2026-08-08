"""Basic checks for the churn pipeline."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data import load_data


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