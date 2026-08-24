# 📊 Telco Customer Churn & Revenue Risk Analytics Platform

An end-to-end data analytics and predictive modeling platform built to analyze customer retention drivers, quantify revenue at risk, and predict individual churn probability using SQL, Python, scikit-learn, and Streamlit.

**🔗 Live App:** [Add your Streamlit Cloud link here]

## 🚀 Key Business Problem

Customer churn directly impacts monthly recurring revenue (MRR). This interactive analytics platform helps stakeholders identify high-risk customer cohorts, evaluate contract and service drivers of churn, and run real-time, per-customer churn risk predictions — with a clear explanation of *why* each prediction was made, before customers cancel.

## 🛠️ Tech Stack & Skills

- **Database & Querying:** SQLite (in-memory), Complex SQL — CTEs, Window Functions (`NTILE`), `CASE` logic
- **Data Manipulation & Analytics:** Python, pandas, NumPy
- **Machine Learning & Modeling:** scikit-learn — Random Forest (primary model, class-balanced) benchmarked against a Logistic Regression baseline (StandardScaler + balanced weighting)
- **Explainable AI:** SHAP (TreeExplainer) for per-customer, feature-level churn explanations
- **Data Visualization:** Plotly, Streamlit
- **Version Control & Deployment:** Git, GitHub, Streamlit Community Cloud

## 📊 Analytical Highlights & Features

1. **SQL Pipeline & Aggregations** — Transformed 7,000+ raw records into tenure cohorts and spending quartiles using in-memory SQLite CTEs and window functions.
2. **Model Benchmarking** — Compared a class-balanced Random Forest (75.9% accuracy, 0.837 ROC-AUC) against a Logistic Regression baseline (73.5% accuracy, 0.832 ROC-AUC) on the same held-out test split. With churn at ~26.5% of customers, ROC-AUC and F1 are reported alongside accuracy since accuracy alone is misleading on imbalanced data.
3. **Decision Threshold Tuning** — An interactive slider recomputes precision/recall/F1 and the confusion matrix live, so the precision–recall trade-off for flagging at-risk customers is explicit and adjustable rather than fixed at a default 50% cutoff.
4. **Explainable Predictions (SHAP)** — Every simulated customer gets a live, per-prediction breakdown of which factors pushed their churn risk up or down, not just a single probability score.
5. **Click-to-Filter Cross-Analysis** — Clicking a bar or pie slice cross-filters the entire dashboard (KPIs, density chart, at-risk list) to that segment, without a page reload.
6. **Revenue Impact & ROI Estimator** — Quantifies revenue at risk (~$139K/month across the full customer base) and models the net ROI of a hypothetical retention campaign under adjustable cost and success-rate assumptions.
7. **At-Risk Customer Export** — Scores every customer currently in view and exports a ranked, downloadable CSV for a retention outreach list.
8. **Interactive Dashboard** — A live Streamlit app with cohort visual analytics, a real-time risk simulator, and a custom dark-themed UI.

## ⚙️ How to Run Locally

1. Clone the repository:
   ```
   git clone https://github.com/rahul-cloud970/churn-analytics.git
   cd churn-analytics
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app (make sure `telco_churn.csv` is in the same folder as `app.py`):
   ```
   streamlit run app.py
   ```
