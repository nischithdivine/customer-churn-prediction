# Customer Churn Prediction: Telco

Predicting which telecom customers are likely to leave, choosing who to contact based on the cost of each mistake, and serving the model as an API in a Docker container.

**Start with [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb)** for the full story.

## Result

Logistic regression ranks churners well: **test Average Precision 0.634** (random = 0.265).

With a cost-based threshold of **0.21** instead of the default 0.5, on 1,409 held-out customers:

| Threshold | Churners caught | Wasted offers | Precision | Recall | Total cost |
|---|---|---|---|---|---|
| **0.21 (cost-based)** | **317 of 374** | 352 | 47% | **85%** | **$95k** |
| 0.5 (default) | 209 of 374 | 109 | 66% | 56% | $114k |

About **17% cheaper**, under assumed costs of $100 per retention offer and $500 per lost customer.

## Approach

1. **Data fix:** 11 blank `TotalCharges` values all belonged to customers with tenure 0 (not billed yet), so they were set to 0, not imputed.
2. **Metric:** only 26.5% of customers churn, so accuracy is misleading (always predicting "stays" scores 73.5%). Models are compared with **Average Precision**.
3. **Split:** stratified 80/20, so both sets keep the 26.5% churn rate.
4. **Pipeline:** numbers scaled, text columns one-hot encoded, all inside a scikit-learn pipeline.
5. **Model comparison:** 8 models, same pipeline, same 5 stratified folds.
6. **Threshold:** chosen by total cost on out-of-fold predictions from the training set, so the test set played no part in the choice.
7. **Serving:** the fitted pipeline and threshold are saved together and served with FastAPI inside a Docker container.

## Model comparison (5-fold CV, Average Precision)

| Model | CV AP |
|---|---|
| XGBoost | 0.667 |
| Random Forest | 0.663 |
| **Logistic Regression (chosen)** | **0.662** |
| MLP (32, 16) | 0.651 |
| SVM (RBF) | 0.644 |
| KNN (k=50) | 0.624 |
| Decision Tree (depth 5) | 0.600 |
| Baseline (same score for everyone) | 0.265 |

The top five sit within the ~0.03 fold-to-fold noise, so they are tied. I chose **logistic regression**: same accuracy, fastest, gives real probabilities, and its coefficients show why a customer is flagged.

## Key findings

- **Tenure** is the strongest signal: 55% of churners leave within their first 12 months.
- **Month-to-month** customers churn at ~43% vs ~3% for two-year contracts.
- **Fiber optic**, **electronic check** payment and **no tech support** all go with churn above 40%.
- **The threshold choice saves more than the model choice:** five models tied, but moving from 0.5 to 0.21 cut cost by ~17%.
- **Watch the coefficients:** MonthlyCharges gets a negative weight even though churners pay more, because fiber (the expensive plan) already carries that effect.

## Prediction API

`POST /predict` takes one customer's 19 fields as JSON and returns:

```json
{ "churn_probability": 0.614, "flag_for_offer": true, "threshold": 0.21 }
```

- Every request is validated with Pydantic: a missing field or a wrong type returns a 422 error before it reaches the model.
- `GET /health` returns `{"status": "ok"}` for uptime checks.
- Interactive docs at `/docs`.

The Docker image uses Python 3.10 and pins scikit-learn 1.7.2, the version that trained the model, and installs only the 5 libraries the API needs (`requirements-api.txt`).

## Limitations and next steps

- Costs are placeholders. The threshold depends heavily on the offer's real success rate: if only half of offers work, the best threshold rises to ~0.40.
- Five different models hit the same ceiling, so new data (support calls, usage trends) would help more than a new algorithm.
- SHAP values would explain each customer's score individually.
- The API has no authentication or monitoring yet; production use would need both, plus drift checks on incoming data.

## How to run

**Analysis**

1. Download the IBM Telco Customer Churn dataset (e.g. [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) and save it as `data/Telco-Customer-Churn.csv`.
2. Install the libraries: `pip install -r requirements.txt`
3. Open `notebooks/01_analysis.ipynb` and run all cells. The last cell saves the model to `models/churn_model.joblib` (not stored in the repo).

**API with Docker** (after step 3)

```
docker build -t churn-api .
docker run -d -p 8000:8000 --name churn churn-api
```

Then open http://127.0.0.1:8000/docs.

## Project structure

```
customer-churn-prediction/
├── app/
│   └── main.py               # FastAPI app: /health and /predict
├── data/                     # Telco-Customer-Churn.csv goes here (not in the repo)
├── models/                   # saved pipeline + threshold (not in the repo)
├── notebooks/
│   ├── 00_exploration.ipynb  # working notes and experiments
│   └── 01_analysis.ipynb     # the clean story
├── Dockerfile
├── requirements.txt          # analysis environment
├── requirements-api.txt      # API container only
└── README.md
```

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, FastAPI, Pydantic, Docker
