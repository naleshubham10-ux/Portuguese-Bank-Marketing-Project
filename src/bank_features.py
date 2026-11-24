"""
Shared feature engineering for the Portuguese Bank term-deposit model.

The same function is used in the notebook (training) and in the Flask app /
predict script (inference), so raw customer records go through identical steps.
"""
import pandas as pd

TARGET = "y"

# 'duration' is only known AFTER the call ends -> leaks the answer, so it is dropped
LEAKY_COLS = ["duration"]

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous", "prev_contacted",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "age_group",
]

AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + engineer features on a raw bank-additional dataframe."""
    out = df.copy()

    # pdays = 999 means "never contacted before" -> explicit flag + sentinel -1
    out["prev_contacted"] = (out["pdays"] != 999).astype(int)
    out["pdays"] = out["pdays"].replace(999, -1)

    # campaign is heavily right-skewed (max 56) -> cap extreme repeat calls
    out["campaign"] = out["campaign"].clip(upper=15)

    # life-stage bucket (young & retired customers behave very differently)
    out["age_group"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    out = out.drop(columns=[c for c in LEAKY_COLS + [TARGET] if c in out.columns])
    return out[NUMERIC_COLS + CATEGORICAL_COLS]
