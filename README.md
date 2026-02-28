# Retention-IQ

Churn prediction API for subscription products. One endpoint: pass a user id, get back a risk score and the main drivers (from a logistic regression model). No Docker, no SHAP.

**What it does**

Takes a CSV of subscription-style features (tenure, sessions, payment failures, support tickets, feature usage), trains a small logistic regression model, and serves it via FastAPI. Hit `/churn-score/{user_id}` to get probability, risk level (low/medium/high), and the top 2 drivers by coefficient impact. Handy for dashboards or piping into lifecycle flows.

**Why it’s useful**

Retention drives LTV; catching at-risk users before they churn lets you intervene (dunning, win-back, support). Payment failures and engagement drops are leading indicators—this service turns them into a single score you can act on.

**Example use**

A platform has lots of renewals and users with mixed signals (failed payments, fewer logins, low feature use). They need a small service that scores users in real time, flags the risky ones, and feeds that into tooling or automation. Retention-IQ is that kind of service: minimal, interpretable, drop-in.

**Model**

Logistic regression, scaled features. Coefficients give you direction and relative importance; we surface “top drivers” as coefficient × scaled value so you can see why someone is high or low risk without pulling in SHAP.

**API**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/churn-score/42
```

Response includes `churn_probability`, `risk_level`, and `top_drivers` (e.g. `payment_failures` +1.42, `sessions_last_30d` -1.01). User id must exist in the dataset.

**Run it**

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` generates data and trains the model if needed, then starts uvicorn with `--reload`. No need to click Run again—edit and save.

Or step by step: `python generate_data.py` → `python train.py` → `uvicorn app.main:app --reload`. Docs at `http://localhost:8000/docs`.

**Resume line**

Built Retention-IQ: FastAPI + logistic regression churn API with real-time scores and interpretable top drivers for subscription platforms.
