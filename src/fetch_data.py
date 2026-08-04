"""Fetch the Telco Customer Churn dataset -> data/churn.csv (cleaned)."""
from pathlib import Path
import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "churn.csv"


def main() -> None:
    df = pd.read_csv(URL)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()