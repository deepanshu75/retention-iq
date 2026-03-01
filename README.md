# Retention-IQ

Churn prediction API for subscription products. Send a user id (UUID); get back a risk score and the main drivers. No Docker, no SHAP—Python, FastAPI, and scikit-learn only.

---

## Overview

Retention-IQ reads a CSV of subscription-style features (tenure, sessions in the last 30 days, payment failures, support tickets, feature usage), trains a logistic regression model, and serves it over FastAPI. A single endpoint, `GET /churn-score/{user_id}`, returns:

- **Churn probability** (0–1)
- **Risk level** (low / medium / high)
- **Top 2 drivers** (coefficient × scaled feature value) so you can see why a user is scored that way

Use it for dashboards, lifecycle automation, or as a small service that plugs into existing subscription tooling.

---

## Why churn prediction matters

Retention drives lifetime value. Catching at-risk users before they churn lets you act—dunning, win-back, support, or product nudges. Payment failures and drops in engagement are leading indicators; this service turns them into one score you can use in rules or pipelines.

---

## Example use case

A platform handles many renewals and users with mixed signals (failed payments, fewer logins, low feature use). They need a small service that scores users in real time, flags the risky ones, and feeds that into internal tooling or automation. Retention-IQ is built for that: minimal, interpretable, and easy to drop in.

---

## Model

- **Algorithm:** Logistic regression with scaled features (StandardScaler + LogisticRegression).
- **Interpretability:** Coefficients give direction and relative importance. We expose “top drivers” as coefficient × scaled value so you can explain scores without bringing in SHAP or other tooling.
- **Why logistic regression:** Chosen for interpretability and production simplicity. Tree-based models can give slightly better accuracy, but coefficient transparency is often more useful in subscription decision systems—debugging, compliance, and product decisions stay straightforward.

---

## API

**Health check**

```bash
curl http://localhost:8000/health
```

**Churn score** (use a `user_id` that exists in your dataset, e.g. from `data/subscription_data.csv`)

```bash
curl http://localhost:8000/churn-score/<uuid>
```

Example response:

```json
{
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "churn_probability": 0.73,
  "risk_level": "high",
  "top_drivers": [
    {"feature": "payment_failures", "impact": 1.42},
    {"feature": "sessions_last_30d", "impact": -1.01}
  ]
}
```

Interactive docs: `http://localhost:8000/docs`.

---

## Running the project

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data** (synthetic subscription dataset with UUIDs)

   ```bash
   python generate_data.py
   ```

   Writes `data/subscription_data.csv`.

3. **Train the model**

   ```bash
   python train.py
   ```

   Saves the pipeline to `app/model.joblib`.

4. **Start the server**

   ```bash
   uvicorn app.main:app --reload
   ```

   Or use `./run.sh` if you have it: it generates data and trains when needed, then starts uvicorn with `--reload` so you can edit and save without restarting.

---

## Development

Changes are merged via pull requests; we use **squash and merge** to keep main history clean. Open a branch, push, then open a PR against `main`.
