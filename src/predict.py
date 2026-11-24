"""
Batch-score a customer file with the trained model.

Usage (from project root):
    python src/predict.py data/bank-additional-full.csv reports/scored_customers.csv

Input: CSV in the original bank-additional format (';' or ',' separated).
Output: same rows + purchase_probability, will_subscribe, priority_rank
        sorted from most to least likely buyer -> a ready-made call list.
"""
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bank_features import add_features  # noqa: E402


def main(in_path: str, out_path: str) -> None:
    art = joblib.load(os.path.join(ROOT, "models", "best_model.pkl"))
    df = pd.read_csv(in_path, sep=None, engine="python")
    proba = art["pipeline"].predict_proba(add_features(df))[:, 1]
    df["purchase_probability"] = proba.round(4)
    df["will_subscribe"] = pd.Series(proba >= art["threshold"]).map({True: "yes", False: "no"}).values
    df = df.sort_values("purchase_probability", ascending=False).reset_index(drop=True)
    df["priority_rank"] = range(1, len(df) + 1)
    df.to_csv(out_path, index=False)
    n_yes = (df["will_subscribe"] == "yes").sum()
    print(f"Scored {len(df):,} customers -> {out_path}")
    print(f"Predicted buyers: {n_yes:,} ({n_yes / len(df):.1%}) at threshold {art['threshold']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
