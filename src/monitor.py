"""Check if new data has drifted away from the training data."""
from scipy.stats import ks_2samp

from src.data import load_data

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    X_train, X_test, y_train, y_test = load_data()

    for feature in FEATURES:
        old_values = X_train[feature]
        new_values = X_test[feature]

        result = ks_2samp(old_values, new_values)
        p_value = result.pvalue

        if p_value < 0.05:
            print(feature, "DRIFTED", p_value)
        else:
            print(feature, "ok", p_value)


if __name__ == "__main__":
    main()