# PRCP-1000 — Portuguese Bank Marketing: Term Deposit Prediction

Predict which existing customers will subscribe to a **term deposit** after a direct phone-marketing call, and give the marketing team data-driven recommendations.

**Domain:** Finance / Banking · **Data:** `bank-additional-full.csv` (41,188 calls, May 2008 – Nov 2010, 20 features + target `y`)

## Results (hold-out test set, 8,236 customers)

| Metric | Value |
|---|---|
| Best model | **Tuned LightGBM** (decision threshold 0.69) |
| ROC-AUC | **0.814** |
| PR-AUC | 0.489 (baseline 0.113) |
| Recall / Precision / F1 | 0.598 / 0.483 / 0.535 |
| Accuracy | 88.3% |
| Top-10% call list conversion | **52.1%** vs 11.3% random (4.6× lift) |
| Top-30% call list | captures **75%** of all buyers |

`duration` is excluded (it is only known after the call — leakage). With it, a benchmark model would reach 0.95 AUC, which is not achievable in practice.

## Project structure

```
PRCP-1000-PortugueseBank/
├── data/bank-additional-full.csv          # dataset (as instructed in the brief)
├── notebook/PRCP-1000-PortugueseBank.ipynb  # ★ single notebook: Task 1, 2, 3 + reports (executed, with outputs)
├── src/
│   ├── bank_features.py                   # shared feature engineering
│   └── predict.py                         # batch scoring -> ranked call list
├── app/
│   ├── app.py                             # Flask web app + JSON API
│   └── templates/index.html
├── models/best_model.pkl                  # trained pipeline + threshold + metrics
├── reports/
│   ├── figures/                           # 15 EDA & evaluation charts
│   ├── model_comparison_cv.csv            # 5-fold CV results, 8 models
│   ├── model_comparison_test.csv          # test-set results, all models
│   └── metrics.json
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1) Re-run the full analysis (≈4 min) — regenerates figures, reports and model
cd notebook && jupyter notebook PRCP-1000-PortugueseBank.ipynb

# 2) Web app (from project root)
python app/app.py              # open http://127.0.0.1:5000

# 3) JSON API
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
     -d '{"age":67,"job":"retired","marital":"married","education":"basic.4y","default":"no","housing":"no","loan":"no","contact":"cellular","month":"oct","day_of_week":"thu","campaign":1,"pdays":6,"previous":1,"poutcome":"success","emp.var.rate":-3.4,"cons.price.idx":92.431,"cons.conf.idx":-26.9,"euribor3m":0.73,"nr.employed":5017.5}'

# 4) Score a whole customer file -> sorted call list
python src/predict.py data/bank-additional-full.csv reports/scored_customers.csv
```

> The model file was saved with scikit-learn 1.9.1. If you install a different version and loading fails, just re-run the notebook to retrain (≈4 min).

## Approach

1. **Data quality:** 12 duplicates removed; `unknown` kept as a category; `pdays=999` → `prev_contacted` flag; `campaign` capped at 15; `age_group` added; `duration` dropped.
2. **EDA (Task 1):** target imbalance (11.3%), segment conversion rates (job, age, poutcome, contact, month), campaign fatigue, macro-economic effect, correlation & leakage analysis.
3. **Modelling (Task 2):** 8 models with 5-fold stratified CV — Logistic Regression (± SMOTE), KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM. RandomizedSearchCV on XGBoost & LightGBM, F1-optimal threshold from out-of-fold predictions.
4. **Suggestions (Task 3):** model-ranked calling, re-target past responders, segment products for students/retired, ≤3 calls per customer, mobile-first, re-balanced campaign calendar, economy-aware timing, agent training, better data capture, A/B testing & quarterly retraining.

## Top-5 model drivers
`euribor3m`, `age`, `campaign`, `cons.price.idx`, `cons.conf.idx`

**Author:** Shubham Nale — B.Tech Computer Science Engineering
