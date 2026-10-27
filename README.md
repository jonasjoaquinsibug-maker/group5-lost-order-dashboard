# Lost Orders Dashboard (Streamlit)

Group 5: Ram Kelly Jersen Leliza and Jonas Joaquin Sibug, BSIT 4-402i
Foundations of Business Analytics, Final Project

**Audience:** the online store's operations manager.
**Business question:** 18.4% of orders are cancelled or returned. Where should the team act, can the
order details we record predict which orders will be lost, and should the discount policy change?

## What is inside

| File | Purpose |
|---|---|
| `app.py` | The dashboard (4 pages: Overview, Can we predict lost orders?, What should we do?, Data and method) |
| `pipeline.py` | The same cleaning steps as the Colab notebook, run on the raw CSV when the app starts |
| `analytics_results.json` | Model and test results exported from the notebook's advanced analytics section |
| `data/ecommerce_retail_transactions_raw.csv` | The raw Kaggle file (12,180 rows) |
| `requirements.txt` | Exact library versions |
| `.streamlit/config.toml` | Colours and theme |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

## Put it online (free live link)

1. Create a **public** GitHub repository and upload everything in this folder, keeping the `data/` and `.streamlit/` folders.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. After a few minutes you get a link like `https://<your-app>.streamlit.app` to submit.

## Dashboard checklist (brief section B)

- **5 KPIs:** net order value, orders, lost-order rate (against a 15% target), lost order value, discount given.
- **6 overview visuals**, each with a business question and an insight note that updates with the filters:
  monthly lost-order rate, monthly kept vs lost value, lost rate by category, lost rate by discount level,
  country × payment heat map, kept vs lost value by category.
- **4 filters:** order date range, country, product category, payment method (plus a reset button).
- **Advanced analytics connected:** model comparison, ROC curves, overfitting check, feature importance,
  a discount-cap simulator and a lost-rate target calculator with recommended actions.
