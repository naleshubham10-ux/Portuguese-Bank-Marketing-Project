"""
Flask web app — Portuguese Bank Term Deposit Predictor.

Run from the project root:
    python app/app.py
then open http://127.0.0.1:5000
"""
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bank_features import add_features  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
artifact = joblib.load(MODEL_PATH)
PIPELINE = artifact["pipeline"]
THRESHOLD = artifact["threshold"]

app = Flask(__name__)

OPTIONS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["university.degree", "high.school", "professional.course", "basic.9y",
                  "basic.6y", "basic.4y", "illiterate", "unknown"],
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}
NUMERIC_FIELDS = ["age", "campaign", "pdays", "previous", "emp.var.rate",
                  "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
DEFAULTS = {"age": 40, "campaign": 1, "pdays": 999, "previous": 0, "emp.var.rate": 1.1,
            "cons.price.idx": 93.994, "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0}


def predict_record(record: dict) -> dict:
    df = pd.DataFrame([record])
    for col in NUMERIC_FIELDS:
        df[col] = pd.to_numeric(df[col])
    proba = float(PIPELINE.predict_proba(add_features(df))[0, 1])
    will_buy = proba >= THRESHOLD
    if proba >= 0.85:
        band = "Very high"
    elif will_buy:
        band = "High"
    elif proba >= 0.4:
        band = "Medium"
    else:
        band = "Low"
    return {"probability": round(proba, 4), "will_subscribe": bool(will_buy),
            "priority": band, "threshold": round(THRESHOLD, 3)}


@app.route("/", methods=["GET", "POST"])
def index():
    result, form = None, {**DEFAULTS, **{k: v[0] for k, v in OPTIONS.items()}}
    if request.method == "POST":
        form = request.form.to_dict()
        result = predict_record(form)
    return render_template("index.html", options=OPTIONS, form=form, result=result,
                           metrics=artifact["test_metrics"], model_name=artifact["model_name"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    records = data if isinstance(data, list) else [data]
    try:
        return jsonify([predict_record({**DEFAULTS, **r}) for r in records])
    except Exception as exc:  # bad input
        return jsonify({"error": str(exc)}), 400


@app.route("/health")
def health():
    return {"status": "ok", "model": artifact["model_name"], "threshold": THRESHOLD}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
