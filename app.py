import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Telco Churn Analytics", layout="wide")

st.title("📊 Telco Customer Churn & Revenue Risk Analytics")
st.markdown("An end-to-end analytics platform to analyze customer churn and predict high-risk accounts.")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("telco_churn.csv")
    # Clean TotalCharges column (convert spaces to NaN and handle numeric conversion)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

df = load_data()

# 2. Key Performance Indicators (KPIs)
st.subheader("1. Business Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churned_customers = len(df[df['Churn'] == 'Yes'])
churn_rate = (churned_customers / total_customers) * 100
total_at_risk_revenue = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Monthly Revenue At-Risk", f"${total_at_risk_revenue:,.2f}")

st.divider()

# 3. Exploratory Data Analysis (Visual Charts)
st.subheader("2. Exploratory Data Analysis")
c1, c2 = st.columns(2)

with c1:
    fig_contract = px.histogram(
        df, x="Contract", color="Churn", 
        barmode="group", title="Churn by Contract Type",
        color_discrete_map={'Yes': '#E63946', 'No': '#1D3557'}
    )
    st.plotly_chart(fig_contract, use_container_width=True)

with c2:
    fig_pay = px.histogram(
        df, x="PaymentMethod", color="Churn", 
        barmode="group", title="Churn by Payment Method",
        color_discrete_map={'Yes': '#E63946', 'No': '#1D3557'}
    )
    st.plotly_chart(fig_pay, use_container_width=True)

st.divider()

# 4. Machine Learning Model Training
st.subheader("3. Predictive Churn Model (Random Forest)")

# Preprocessing features for model
@st.cache_resource
def train_model(data):
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = data[features].copy()
    y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, accuracy

model, feature_cols, model_accuracy = train_model(df)
st.success(f"Random Forest Model Trained Successfully! Test Accuracy: **{model_accuracy * 100:.2f}%**")

# 5. Live Prediction Tool for Individual Customers
st.subheader("4. Predict Churn Risk for a Customer")

col_a, col_b, col_c = st.columns(3)
with col_a:
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", value=65.0)
with col_b:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
with col_c:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    total_charges = tenure * monthly_charges

# Predict logic
if st.button("Calculate Churn Probability"):
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method
    }])
    
    input_encoded = pd.get_dummies(input_data).reindex(columns=feature_cols, fill_value=0)
    churn_prob = model.predict_proba(input_encoded)[0][1] * 100
    
    st.write("---")
    if churn_prob >= 50:
        st.error(f"⚠️ **High Risk of Churn!** Estimated Probability: **{churn_prob:.1f}%**")
    else:
        st.success(f"✅ **Low Risk Customer!** Estimated Probability: **{churn_prob:.1f}%**")