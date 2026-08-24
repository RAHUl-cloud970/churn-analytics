import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Telco Churn Executive Platform",
    layout="wide",
    page_icon="📊",
)


def render_html(content: str) -> None:
    """st.markdown treats <div> blocks as closed by the first blank line
    (not the matching closing tag), so an indented multi-line block with a
    blank line in it gets dumped as literal text. Strip indentation and
    blank lines before rendering to avoid that."""
    lines = [line.strip() for line in content.strip("\n").splitlines()]
    lines = [line for line in lines if line]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ---------------------------------------------------------------- styling
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --border: rgba(71, 85, 105, 0.34);
        --text: #F8FAFC;
        --gold: #FACC15;
        --radius-xl: 22px;
        --radius-lg: 18px;
        --radius-md: 14px;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(14, 165, 233, 0.10), transparent 25%),
            radial-gradient(circle at 92% 4%, rgba(250, 204, 21, 0.06), transparent 22%),
            radial-gradient(circle at 50% 100%, rgba(56, 189, 248, 0.045), transparent 28%),
            linear-gradient(135deg, #050816 0%, #080D1C 48%, #060A14 100%);
        color: var(--text);
    }
    .main .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #070B16; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #155E75, #0EA5E9, #CA8A04); border-radius: 999px; border: 2px solid #070B16; }
    .hero-banner {
        position: relative; overflow: hidden; isolation: isolate;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(12, 25, 46, 0.94)), radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.15), transparent 32%);
        border: 1px solid rgba(71, 85, 105, 0.45);
        border-radius: var(--radius-xl);
        padding: 30px 34px; margin-bottom: 24px;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
    }
    .hero-wave { position: absolute; right: -20px; top: -20px; width: 55%; height: 160%; opacity: 0.8; pointer-events: none; }
    .hero-title { font-size: clamp(25px, 2.2vw, 34px); font-weight: 800; color: #F8FAFC; margin-bottom: 7px; letter-spacing: -0.8px; position: relative; z-index: 1; }
    .hero-subtitle { font-size: 13.5px; line-height: 1.6; color: #94A3B8; font-weight: 500; position: relative; z-index: 1; }
    .section-title { position: relative; font-size: 19px; font-weight: 800; color: #F1F5F9; margin: 10px 0 5px 0; letter-spacing: -0.25px; }
    .section-title::before { content: ""; display: inline-block; width: 4px; height: 18px; margin-right: 9px; vertical-align: -3px; border-radius: 999px; background: linear-gradient(180deg, #38BDF8, #FACC15); }
    .section-caption { color: #64748B; font-size: 13px; margin-bottom: 14px; }
    div[data-testid="stMetric"] {
        position: relative; overflow: hidden; min-height: 112px;
        background: linear-gradient(145deg, rgba(19, 28, 46, 0.96), rgba(10, 17, 32, 0.96));
        border: 1px solid rgba(71, 85, 105, 0.40);
        padding: 20px 20px 16px 20px; border-radius: var(--radius-lg);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.24);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]::before { content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 2px; background: linear-gradient(90deg, rgba(56, 189, 248, 0), #38BDF8, #FACC15, rgba(250, 204, 21, 0)); opacity: 0.8; }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(56, 189, 248, 0.35); }
    div[data-testid="stMetric"] label { color: #7C8AA5 !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.75px; }
    div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-size: clamp(22px, 2vw, 28px) !important; font-weight: 800 !important; letter-spacing: -0.5px; }
    .chart-card {
        position: relative; overflow: hidden;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.94), rgba(8, 14, 28, 0.96));
        border: 1px solid rgba(71, 85, 105, 0.34);
        border-radius: var(--radius-lg);
        padding: 17px 18px 7px 18px; margin-bottom: 7px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
    }
    .chart-card-title { font-size: 15px; font-weight: 800; color: #F1F5F9; margin: 2px 0 4px 4px; }
    .chart-card-hint { font-size: 11.5px; color: #64748B; margin: 0 0 10px 4px; }
    .js-plotly-plot, .plotly, .plot-container { border-radius: 14px; overflow: hidden; }
    .js-plotly-plot .xtick text, .js-plotly-plot .ytick text, .js-plotly-plot .gtitle, .js-plotly-plot .legendtext { font-family: 'Inter', sans-serif !important; }
    .js-plotly-plot .gtitle { font-weight: 800 !important; }
    .stButton > button {
        background: rgba(250, 204, 21, 0.06) !important; color: #FACC15 !important; font-weight: 800;
        border-radius: 12px; border: 1.5px solid rgba(250, 204, 21, 0.55) !important;
        padding: 11px 16px; width: 100%; min-height: 45px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { background: rgba(250, 204, 21, 0.14) !important; transform: translateY(-1px); }
    div[data-testid="stDownloadButton"] > button {
        background: rgba(56, 189, 248, 0.08) !important; color: #38BDF8 !important; font-weight: 800;
        border-radius: 12px; border: 1.5px solid rgba(56, 189, 248, 0.5) !important;
        padding: 11px 16px; width: 100%; min-height: 45px;
    }
    div[data-testid="stDownloadButton"] > button:hover { background: rgba(56, 189, 248, 0.16) !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[data-testid="stNumberInput"] input {
        background: linear-gradient(145deg, rgba(19, 28, 46, 0.96), rgba(12, 19, 36, 0.96)) !important;
        border: 1px solid rgba(71, 85, 105, 0.42) !important; border-radius: 12px !important; color: #F8FAFC !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover, div[data-testid="stNumberInput"] input:hover { border-color: rgba(56, 189, 248, 0.45) !important; }
    div[data-baseweb="select"] > div:focus-within, div[data-testid="stNumberInput"] input:focus { border-color: #38BDF8 !important; box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.10) !important; outline: none !important; }
    div[data-testid="stNumberInput"] input { min-height: 42px; padding-left: 13px; font-weight: 600; }
    ul[data-baseweb="menu"] { background: linear-gradient(145deg, #111B2F, #0B1324) !important; border: 1px solid rgba(71, 85, 105, 0.46) !important; border-radius: 12px !important; padding: 6px !important; }
    li[data-baseweb="option"]:hover { background: rgba(56, 189, 248, 0.10) !important; color: #F8FAFC !important; }
    .filter-box { background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(8, 14, 28, 0.94)); border: 1px solid rgba(71, 85, 105, 0.38); border-radius: var(--radius-md); padding: 14px 16px 4px 16px; }
    .filter-label { font-size: 13px; font-weight: 700; color: #CBD5E1; margin-bottom: 6px; }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: rgba(51, 65, 85, 0.72) !important; border-radius: 999px !important; }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div { background: linear-gradient(90deg, #0284C7 0%, #0EA5E9 45%, #FACC15 100%) !important; border-radius: 999px !important; }
    div[data-testid="stSlider"] [role="slider"] { background: radial-gradient(circle at 35% 30%, #FFF7AE 0%, #FACC15 38%, #CA8A04 100%) !important; border: 3px solid #08101F !important; width: 19px !important; height: 19px !important; box-shadow: 0 0 0 4px rgba(250,204,21,0.12) !important; }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: #FACC15 !important; font-weight: 800 !important; }
    div[data-testid="stTickBar"] { display: none; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(71, 85, 105, 0.34); }
    .sim-input-card { background: linear-gradient(145deg, rgba(19, 28, 46, 0.96), rgba(9, 16, 30, 0.98)); border: 1px solid rgba(71, 85, 105, 0.40); border-radius: var(--radius-md); padding: 14px 16px 6px 16px; margin-bottom: 12px; }
    .sim-input-label { font-size: 12.5px; font-weight: 700; color: #94A3B8; margin-bottom: 2px; }
    .cm-grid { display: grid; grid-template-columns: auto 1fr 1fr; grid-template-rows: auto 1fr 1fr auto; gap: 8px; align-items: center; }
    .cm-cell { border-radius: 14px; padding: 18px 10px; text-align: center; color: #F8FAFC; box-shadow: 0 8px 18px rgba(0,0,0,0.25); }
    .cm-cell .cm-num { font-size: 30px; font-weight: 800; line-height: 1; }
    .cm-cell .cm-label { font-size: 12.5px; font-weight: 600; opacity: 0.92; margin-top: 4px; }
    .cm-tn { background: linear-gradient(135deg, #0F766E, #14B8A6); }
    .cm-fp { background: linear-gradient(135deg, #B45309, #F59E0B); }
    .cm-fn { background: linear-gradient(135deg, #6D28D9, #9333EA); }
    .cm-tp { background: linear-gradient(135deg, #1D4ED8, #3B82F6); }
    .cm-row-label { color: #94A3B8; font-size: 12.5px; font-weight: 700; text-align: center; padding: 0 4px; }
    .cm-col-label { color: #94A3B8; font-size: 12.5px; font-weight: 700; text-align: center; }
    .cm-axis-label { color: #64748B; font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; text-align: center; }
    .factor-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
    .factor-name { flex: 0 0 150px; font-size: 13px; font-weight: 600; color: #CBD5E1; }
    .factor-bar-track { flex: 1; height: 9px; background: rgba(51,65,85,0.55); border-radius: 999px; overflow: hidden; }
    .factor-bar-fill { height: 100%; border-radius: 999px; }
    .factor-tag { flex: 0 0 100px; font-size: 11px; font-weight: 700; text-align: right; }
    .tag-high { color: #FB7185; }
    .tag-medium { color: #FACC15; }
    .tag-low { color: #38BDF8; }
    .result-card { border-radius: 14px; padding: 16px 18px; margin-top: 10px; border: 1px solid rgba(255,255,255,0.08); }
    .result-critical { background: linear-gradient(135deg, rgba(127,29,58,0.55), rgba(127,29,58,0.25)); }
    .result-moderate { background: linear-gradient(135deg, rgba(146,64,14,0.55), rgba(146,64,14,0.25)); }
    .result-safe { background: linear-gradient(135deg, rgba(6,95,70,0.55), rgba(6,95,70,0.25)); }
    .result-title { font-size: 16px; font-weight: 800; margin-bottom: 3px; }
    .result-desc { font-size: 12.5px; color: #CBD5E1; }
    .result-action { background: linear-gradient(135deg, rgba(6,95,70,0.5), rgba(6,95,70,0.22)); }
    .xfilter-chip { display: inline-flex; align-items: center; gap: 6px; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.4); color: #38BDF8; font-size: 12.5px; font-weight: 700; padding: 6px 12px; border-radius: 999px; margin-right: 8px; }
    .xfilter-empty { color: #64748B; font-size: 12.5px; font-style: italic; }
    .compare-table { width: 100%; }
    .compare-row { display: grid; grid-template-columns: 1.3fr 1fr 1fr; padding: 11px 8px; border-bottom: 1px solid rgba(71,85,105,0.22); align-items: center; }
    .compare-row:last-child { border-bottom: none; }
    .compare-header .compare-value, .compare-header .compare-metric { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #7C8AA5; font-weight: 700; }
    .compare-metric { font-size: 13.5px; font-weight: 600; color: #CBD5E1; }
    .compare-value { font-size: 14.5px; font-weight: 700; color: #CBD5E1; text-align: right; padding-right: 6px; }
    .compare-winner { color: #38BDF8; }
    div[data-testid="stAlert"] { border-radius: 13px !important; border: 1px solid rgba(71, 85, 105, 0.30) !important; }
    label, [data-testid="stWidgetLabel"] p { color: #A8B5C9 !important; font-weight: 600 !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #71809A !important; }
    hr { border: 0 !important; height: 1px !important; margin: 24px 0 !important; background: linear-gradient(90deg, transparent, rgba(71,85,105,0.55), rgba(56,189,248,0.20), rgba(71,85,105,0.55), transparent) !important; }
    </style>
    """
)

PLOT_TEMPLATE = "plotly_dark"
COLOR_MAP = {"No": "#38BDF8", "Yes": "#FB7185"}
COHORT_ORDER = ["1. 0-1 Year (High Risk)", "2. 1-2 Years", "3. 2-4 Years", "4. 4+ Years (Loyal)"]
DENSITY_SCALE = [[0, "#0B1120"], [0.35, "#0E4C6E"], [0.65, "#0EA5E9"], [1, "#FACC15"]]
FEATURES = ["tenure", "MonthlyCharges", "Contract", "PaymentMethod", "InternetService"]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_churn.csv")


def style_fig(fig, title=None):
    # title_font only gets attached when a title is actually set — attaching
    # it unconditionally makes Plotly initialize an empty title object and
    # render the literal word "undefined" for charts that don't pass one.
    layout_kwargs = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if title:
        layout_kwargs["title"] = dict(text=f"<b>{title}</b>", font=dict(size=15, color="#F1F5F9"))
    fig.update_layout(**layout_kwargs)
    return fig


def encode_features(data: pd.DataFrame, columns) -> pd.DataFrame:
    return pd.get_dummies(data[FEATURES].copy()).reindex(columns=columns, fill_value=0)


def compute_metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, cm


@st.cache_data(show_spinner="Loading and transforming customer data...")
def get_data_from_sql(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{os.path.basename(path)}'. Place it next to app.py.")

    df_raw = pd.read_csv(path)
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df_raw["TotalCharges"] = df_raw["TotalCharges"].fillna(df_raw["TotalCharges"].median())

    conn = sqlite3.connect(":memory:")
    df_raw.to_sql("telco_customers", conn, index=False, if_exists="replace")

    sql_query = """
    WITH CustomerMetrics AS (
        SELECT
            customerID, gender, SeniorCitizen, tenure, Contract, PaymentMethod,
            InternetService, MonthlyCharges, TotalCharges, Churn,
            CASE
                WHEN tenure <= 12 THEN '1. 0-1 Year (High Risk)'
                WHEN tenure <= 24 THEN '2. 1-2 Years'
                WHEN tenure <= 48 THEN '3. 2-4 Years'
                ELSE '4. 4+ Years (Loyal)'
            END AS tenure_cohort,
            NTILE(4) OVER (ORDER BY MonthlyCharges DESC) AS spending_quartile
        FROM telco_customers
    )
    SELECT * FROM CustomerMetrics;
    """
    df_sql = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df_sql


try:
    df = get_data_from_sql(DATA_PATH)
except FileNotFoundError as e:
    st.error(f"🚫 {e}")
    st.stop()

# ---------------------------------------------------------------- header
HERO_WAVE_SVG = (
    '<svg class="hero-wave" viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="waveGrad1" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#38BDF8" stop-opacity="0"/>'
    '<stop offset="55%" stop-color="#38BDF8" stop-opacity="0.5"/>'
    '<stop offset="100%" stop-color="#FACC15" stop-opacity="0.3"/>'
    '</linearGradient>'
    '<linearGradient id="waveGrad2" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#0EA5E9" stop-opacity="0"/>'
    '<stop offset="60%" stop-color="#0EA5E9" stop-opacity="0.3"/>'
    '<stop offset="100%" stop-color="#38BDF8" stop-opacity="0"/>'
    '</linearGradient>'
    '</defs>'
    '<path d="M50,250 C220,180 320,60 660,30" stroke="url(#waveGrad1)" stroke-width="2.2" fill="none"/>'
    '<path d="M100,270 C260,210 360,110 680,70" stroke="url(#waveGrad2)" stroke-width="1.4" fill="none"/>'
    '<circle cx="640" cy="40" r="2.4" fill="#FACC15" opacity="0.85"/>'
    '<circle cx="600" cy="75" r="1.6" fill="#38BDF8" opacity="0.75"/>'
    '</svg>'
)

render_html(
    f"""
    <div class="hero-banner">
    {HERO_WAVE_SVG}
    <div class="hero-title">📊 Telco Executive Revenue & Retention Platform</div>
    <div class="hero-subtitle">Strategy Layer — Model Benchmarking, Density Analytics & Click-to-Filter Cross-Analysis</div>
    </div>
    """
)

# ---------------------------------------------------------------- filters
all_contracts = sorted(df["Contract"].unique().tolist())
all_payments = sorted(df["PaymentMethod"].unique().tolist())
all_internet = sorted(df["InternetService"].unique().tolist())


def reset_filters():
    st.session_state["contract_filter"] = all_contracts
    st.session_state["payment_filter"] = all_payments
    st.session_state["xf_cohort"] = None
    st.session_state["xf_contract"] = None


st.session_state.setdefault("xf_cohort", None)
st.session_state.setdefault("xf_contract", None)

render_html('<div class="section-title">🎛️ Strategic Segmentation</div>')
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

with ctrl_col1:
    render_html('<div class="filter-box"><div class="filter-label">📄 Contract Type:</div>')
    selected_contract = st.multiselect(
        "Contract Type", options=all_contracts, default=all_contracts, key="contract_filter",
        label_visibility="collapsed",
    )
    render_html("</div>")
with ctrl_col2:
    render_html('<div class="filter-box"><div class="filter-label">💳 Payment Method:</div>')
    selected_payment = st.multiselect(
        "Payment Method", options=all_payments, default=all_payments, key="payment_filter",
        label_visibility="collapsed",
    )
    render_html("</div>")
with ctrl_col3:
    st.write("")
    st.write("")
    st.button("🔄 Reset View", on_click=reset_filters)

# Injected right after the multiselect widgets mount, so it sits later in
# the DOM than Streamlit's own tag styling and reliably wins the cascade
# (a rule declared earlier in the page loses to a same-specificity rule
# that Streamlit injects for the widget when it mounts).
render_html(
    """
    <style>
    span[data-baseweb="tag"],
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #0369A1 !important;
        background-image: linear-gradient(135deg, #0EA5E9 0%, #0369A1 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 9px !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    span[data-baseweb="tag"] svg,
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
        fill: #F8FAFC !important;
    }
    </style>
    """
)

filtered_df = df[(df["Contract"].isin(selected_contract)) & (df["PaymentMethod"].isin(selected_payment))]

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

chips = []
if st.session_state["xf_cohort"]:
    chips.append(f'<span class="xfilter-chip">🎯 Tenure Cohort: {st.session_state["xf_cohort"][3:]}</span>')
if st.session_state["xf_contract"]:
    chips.append(f'<span class="xfilter-chip">🎯 Contract: {st.session_state["xf_contract"]}</span>')

chip_col, clear_col = st.columns([5, 1])
with chip_col:
    if chips:
        render_html("".join(chips))
    else:
        render_html('<span class="xfilter-empty">No active chart filters — click a bar or slice below to cross-filter the whole dashboard.</span>')
with clear_col:
    if chips:
        if st.button("✖ Clear chart filters"):
            st.session_state["xf_cohort"] = None
            st.session_state["xf_contract"] = None
            st.rerun()

view_df = filtered_df.copy()
if st.session_state["xf_cohort"]:
    view_df = view_df[view_df["tenure_cohort"] == st.session_state["xf_cohort"]]
if st.session_state["xf_contract"]:
    view_df = view_df[view_df["Contract"] == st.session_state["xf_contract"]]

if view_df.empty:
    st.warning("No customers match the active chart filters. Clear a filter above to see data again.")
    st.stop()

st.divider()

# ---------------------------------------------------------------- KPIs
st.markdown('<div class="section-title">1. Financial Health & Exposure</div>', unsafe_allow_html=True)

total_cust = len(view_df)
churn_cust = int((view_df["Churn"] == "Yes").sum())
churn_rate = (churn_cust / total_cust) * 100 if total_cust else 0
revenue_at_risk = view_df.loc[view_df["Churn"] == "Yes", "MonthlyCharges"].sum()

recovery_pct = st.slider(
    "Assumed churn-recovery rate for 'Potential Recovery' estimate (%)",
    min_value=5, max_value=50, value=15, step=5,
    help="This is a scenario assumption, not a measured figure — adjust to model different retention-campaign outcomes.",
)
potential_savings = revenue_at_risk * (recovery_pct / 100)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Customers", f"{total_cust:,}")
kpi2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
kpi3.metric("Revenue At-Risk", f"${revenue_at_risk:,.0f}")
kpi4.metric(f"Potential Recovery ({recovery_pct}%)", f"${potential_savings:,.0f}")

st.divider()

# ---------------------------------------------------------------- live insights
with st.expander("📝 Analyst Insights (auto-generated from the current filter view)", expanded=True):
    m2m_data = view_df[view_df["Contract"] == "Month-to-month"]
    other_data = view_df[view_df["Contract"] != "Month-to-month"]
    m2m_churn_rate = (m2m_data["Churn"] == "Yes").mean() if not m2m_data.empty else 0
    other_churn_rate = (other_data["Churn"] == "Yes").mean() if not other_data.empty else 0
    risk_ratio = (m2m_churn_rate / other_churn_rate) if other_churn_rate > 0 else 0

    new_cust = view_df[view_df["tenure"] <= 12]
    tenure_risk_pct = (new_cust["Churn"] == "Yes").mean() * 100 if not new_cust.empty else 0

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown(f"**Data findings ({len(view_df):,} customers in view)**")
        st.write(f"- Month-to-month customers churn **{risk_ratio:.1f}x** more often than customers on longer contracts, in this segment.")
        st.write(f"- **{tenure_risk_pct:.1f}%** of customers with 12 months' tenure or less have churned.")
    with col_ins2:
        st.markdown("**Suggested focus**")
        if risk_ratio > 2:
            st.info("Contract-type risk is elevated — consider term-based incentives for month-to-month customers.")
        elif tenure_risk_pct > 25:
            st.warning("Early-tenure churn is elevated — consider more onboarding touchpoints in the first 6 months.")
        else:
            st.success("This segment looks comparatively stable — a good candidate for upsell rather than retention spend.")

st.divider()

# ---------------------------------------------------------------- charts
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    render_html('<div class="chart-card-hint">👆 Click a bar to cross-filter the whole dashboard</div>')
    fig_cohort = px.histogram(
        filtered_df, x="tenure_cohort", color="Churn", barmode="group",
        category_orders={"tenure_cohort": COHORT_ORDER},
        color_discrete_map=COLOR_MAP, template=PLOT_TEMPLATE,
    )
    fig_cohort.update_layout(bargap=0.25, bargroupgap=0.08)
    fig_cohort.update_traces(marker_line_width=0)
    cohort_event = st.plotly_chart(
        style_fig(fig_cohort, "Churn Concentration by Tenure"),
        use_container_width=True, on_select="rerun", selection_mode="points", key="cohort_chart",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_density = go.Figure(
        go.Histogram2dContour(
            x=view_df["tenure"],
            y=view_df["MonthlyCharges"],
            colorscale=DENSITY_SCALE,
            contours=dict(showlabels=False, coloring="fill"),
            line=dict(width=0.5, color="rgba(255,255,255,0.08)"),
            colorbar=dict(title="Density", thickness=12, len=0.8, tickfont=dict(color="#94A3B8")),
        )
    )
    fig_density.update_layout(xaxis_title="tenure", yaxis_title="MonthlyCharges")
    st.plotly_chart(style_fig(fig_density, "Risk Density Zone (Tenure vs. Charges)"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

cohort_points = (cohort_event.get("selection", {}) or {}).get("points", []) if cohort_event else []
new_cohort = cohort_points[0].get("x") if cohort_points else None
if new_cohort != st.session_state["xf_cohort"]:
    st.session_state["xf_cohort"] = new_cohort
    st.rerun()

st.divider()

# ---------------------------------------------------------------- models
st.markdown('<div class="section-title">2. Churn Drivers & Predictive Insights</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_random_forest(data: pd.DataFrame):
    X = pd.get_dummies(data[FEATURES].copy())
    y = data["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)

    return model, X.columns, y_test.to_numpy(), y_proba_test, (fpr, tpr), roc_auc


@st.cache_resource(show_spinner="Training logistic regression baseline...")
def train_logistic_regression(data: pd.DataFrame):
    X = pd.get_dummies(data[FEATURES].copy())
    y = data["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = pipe.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_test),
    }
    return metrics, (fpr, tpr)


model, feature_cols, rf_y_test, rf_y_proba, rf_roc_data, rf_roc_auc = train_random_forest(df)
lr_metrics, lr_roc_data = train_logistic_regression(df)

rf_default_metrics, _ = compute_metrics_at_threshold(rf_y_test, rf_y_proba, 0.5)
rf_default_metrics["roc_auc"] = rf_roc_auc

st.caption(
    "Both models are trained once on the full dataset (independent of the filters above), so scores stay "
    "consistent while you explore the charts. Churn is ~26% of customers here, so accuracy alone doesn't "
    "tell you much — precision/recall/F1/ROC-AUC show how well each model actually catches churners."
)

# --- model comparison (plain table, no chart) ---
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
render_html('<div class="chart-card-title">Model Comparison — Random Forest vs. Logistic Regression</div><div class="chart-card-hint">Both evaluated on the same held-out test split, default 0.5 threshold. Better score in each row shown in blue.</div>')

metric_rows = [
    ("Accuracy", rf_default_metrics["accuracy"], lr_metrics["accuracy"]),
    ("Precision", rf_default_metrics["precision"], lr_metrics["precision"]),
    ("Recall", rf_default_metrics["recall"], lr_metrics["recall"]),
    ("F1-Score", rf_default_metrics["f1"], lr_metrics["f1"]),
    ("ROC-AUC", rf_roc_auc, lr_metrics["roc_auc"]),
]
rows_html = ['<div class="compare-row compare-header"><div class="compare-metric">Metric</div><div class="compare-value">Random Forest</div><div class="compare-value">Logistic Regression</div></div>']
for name, rf_val, lr_val in metric_rows:
    rf_cls = "compare-winner" if rf_val >= lr_val else ""
    lr_cls = "compare-winner" if lr_val > rf_val else ""
    rows_html.append(
        f'<div class="compare-row"><div class="compare-metric">{name}</div>'
        f'<div class="compare-value {rf_cls}">{rf_val*100:.1f}%</div>'
        f'<div class="compare-value {lr_cls}">{lr_val*100:.1f}%</div></div>'
    )
render_html(f'<div class="compare-table">{"".join(rows_html)}</div>')
st.markdown("</div>", unsafe_allow_html=True)

# --- threshold tuning ---
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
render_html('<div class="chart-card-title">Decision Threshold Tuning (Random Forest)</div>')
threshold = st.slider(
    "Probability threshold for flagging a customer as churn risk",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Lower threshold catches more churners (higher recall) but flags more false alarms (lower precision).",
)
thresh_metrics, thresh_cm = compute_metrics_at_threshold(rf_y_test, rf_y_proba, threshold)

tm1, tm2, tm3, tm4, tm5 = st.columns(5)
tm1.metric("Accuracy", f"{thresh_metrics['accuracy']*100:.1f}%")
tm2.metric("Precision (Churn)", f"{thresh_metrics['precision']*100:.1f}%")
tm3.metric("Recall (Churn)", f"{thresh_metrics['recall']*100:.1f}%")
tm4.metric("F1-Score", f"{thresh_metrics['f1']*100:.1f}%")
tm5.metric("ROC-AUC", f"{rf_roc_auc:.3f}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
c3, c4 = st.columns([1, 2])

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    render_html('<div class="chart-card-hint">👆 Click a slice to cross-filter the whole dashboard</div>')
    churned_only = filtered_df[filtered_df["Churn"] == "Yes"]
    if churned_only.empty:
        st.info("No churned customers in this filter selection.")
        contract_event = None
    else:
        fig_pie = px.pie(
            churned_only, names="Contract", hole=0.55,
            color_discrete_sequence=["#0EA5E9", "#FACC15", "#FB7185"], template=PLOT_TEMPLATE,
        )
        fig_pie.update_traces(marker_line_width=0, textfont_size=13)
        contract_event = st.plotly_chart(
            style_fig(fig_pie, "High-Risk Contracts"),
            use_container_width=True, on_select="rerun", selection_mode="points", key="contract_chart",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fi_df = pd.DataFrame({"Feature": feature_cols, "Impact": model.feature_importances_}).sort_values("Impact", ascending=True)
    fig_fi = px.bar(
        fi_df.tail(8), x="Impact", y="Feature", orientation="h",
        color="Impact", color_continuous_scale=["#78350F", "#FACC15"],
        template=PLOT_TEMPLATE, text="Impact",
    )
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig_fi.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig_fi, "Top Drivers of Customer Churn"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

contract_points = (contract_event.get("selection", {}) or {}).get("points", []) if contract_event else []
new_contract = contract_points[0].get("label") if contract_points else None
if new_contract != st.session_state["xf_contract"]:
    st.session_state["xf_contract"] = new_contract
    st.rerun()

c5, c6 = st.columns(2)
with c5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    render_html('<div class="chart-card-title">🔲 Confusion Matrix</div><div class="chart-card-hint">At your selected threshold above</div>')

    tn, fp, fn, tp = int(thresh_cm[0][0]), int(thresh_cm[0][1]), int(thresh_cm[1][0]), int(thresh_cm[1][1])

    render_html(
        f"""
        <div class="cm-grid">
        <div></div>
        <div class="cm-col-label">No Churn</div>
        <div class="cm-col-label">Churn</div>
        <div class="cm-row-label">No Churn</div>
        <div class="cm-cell cm-tn"><div class="cm-num">{tn}</div><div class="cm-label">True Negative</div></div>
        <div class="cm-cell cm-fp"><div class="cm-num">{fp}</div><div class="cm-label">False Positive</div></div>
        <div class="cm-row-label">Churn</div>
        <div class="cm-cell cm-fn"><div class="cm-num">{fn}</div><div class="cm-label">False Negative</div></div>
        <div class="cm-cell cm-tp"><div class="cm-num">{tp}</div><div class="cm-label">True Positive</div></div>
        <div></div>
        <div class="cm-axis-label" style="grid-column: 2 / span 2;">Predicted</div>
        </div>
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c6:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    rf_fpr, rf_tpr = rf_roc_data
    lr_fpr, lr_tpr = lr_roc_data
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines", name=f"Random Forest (AUC={rf_roc_auc:.3f})", line=dict(color="#38BDF8", width=3)))
    fig_roc.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode="lines", name=f"Logistic Regression (AUC={lr_metrics['roc_auc']:.3f})", line=dict(color="#FACC15", width=2, dash="dot")))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#64748B", dash="dash")))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(style_fig(fig_roc, "ROC Curve"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------- at-risk list
st.markdown('<div class="section-title">3. Top At-Risk Customers</div>', unsafe_allow_html=True)
render_html('<div class="section-caption">Random Forest risk scores for customers currently in view — export for a retention outreach list.</div>')

scored_encoded = encode_features(view_df, feature_cols)
risk_scores = model.predict_proba(scored_encoded)[:, 1]

at_risk_df = view_df[["customerID", "tenure", "Contract", "PaymentMethod", "InternetService", "MonthlyCharges", "Churn"]].copy()
at_risk_df["Churn Probability (%)"] = (risk_scores * 100).round(1)
at_risk_df = at_risk_df.sort_values("Churn Probability (%)", ascending=False).reset_index(drop=True)

top_n = st.slider("Number of customers to show", min_value=10, max_value=min(200, len(at_risk_df)), value=min(25, len(at_risk_df)), step=5)
top_at_risk = at_risk_df.head(top_n)

st.dataframe(
    top_at_risk,
    use_container_width=True,
    height=360,
    hide_index=True,
    column_config={
        "customerID": st.column_config.TextColumn("Customer ID"),
        "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
        "Churn": st.column_config.TextColumn("Actual Churn"),
        "Churn Probability (%)": st.column_config.NumberColumn("Churn Probability", format="%.1f%%"),
    },
)

csv_bytes = top_at_risk.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download at-risk customer list (CSV)",
    data=csv_bytes,
    file_name="at_risk_customers.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------- ROI estimator
st.markdown('<div class="section-title">💰 Retention Campaign ROI Estimator</div>', unsafe_allow_html=True)
render_html('<div class="section-caption">A what-if scenario using your own cost and success-rate assumptions — not a measured outcome.</div>')

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
c_roi1, c_roi2 = st.columns(2)
with c_roi1:
    offer_cost = st.number_input("Cost of retention offer per customer ($)", value=20.0, min_value=0.0)
    success_rate = st.slider("Expected success rate of the offer (%)", 5, 100, 20)

customers_targeted = int((at_risk_df["Churn Probability (%)"] > 60).sum())
customers_saved = int(customers_targeted * (success_rate / 100))
total_cost = customers_targeted * offer_cost
revenue_saved = customers_saved * view_df["MonthlyCharges"].mean()
roi = ((revenue_saved - total_cost) / total_cost) * 100 if total_cost > 0 else 0

with c_roi2:
    st.write(f"Targeting **{customers_targeted}** customers currently above 60% predicted risk.")
    st.metric("Estimated Net Monthly Value", f"${revenue_saved - total_cost:,.0f}", f"{roi:.1f}% ROI")
    st.caption("Estimated monthly revenue retained minus campaign cost, based on the assumptions above.")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------- simulator
st.markdown('<div class="section-title">4. Real-Time Customer Risk Simulator</div>', unsafe_allow_html=True)

sim_left, sim_right = st.columns([1.15, 1])

with sim_right:
    render_html('<div class="sim-input-card"><div class="sim-input-label">⏱️ Tenure (Months)</div>')
    tenure = st.slider("Tenure (Months)", 1, 72, 12, label_visibility="collapsed")
    render_html("</div>")

    in_a, in_b = st.columns(2)
    with in_a:
        render_html('<div class="sim-input-card"><div class="sim-input-label">💲 Monthly Charges ($)</div>')
        monthly = st.number_input("Monthly Charges", 18.0, 120.0, 75.0, label_visibility="collapsed")
        render_html("</div>")
    with in_b:
        render_html('<div class="sim-input-card"><div class="sim-input-label">📄 Contract Type</div>')
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], label_visibility="collapsed")
        render_html("</div>")

    in_c, in_d = st.columns(2)
    with in_c:
        render_html('<div class="sim-input-card"><div class="sim-input-label">💳 Payment Method</div>')
        payment = st.selectbox("Payment Method", all_payments, label_visibility="collapsed")
        render_html("</div>")
    with in_d:
        render_html('<div class="sim-input-card"><div class="sim-input-label">🌐 Internet Service</div>')
        internet = st.selectbox("Internet Service", all_internet, label_visibility="collapsed")
        render_html("</div>")

input_data = pd.DataFrame([{
    "tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
    "PaymentMethod": payment, "InternetService": internet,
}])
input_encoded = encode_features(input_data, feature_cols)
prob = model.predict_proba(input_encoded)[0][1] * 100

if prob >= 70:
    risk_color, risk_class, risk_title = "#FB7185", "result-critical", f"🚨 Critical Risk: {prob:.1f}%"
    risk_desc = "This customer is very likely to churn."
elif prob >= 40:
    risk_color, risk_class, risk_title = "#FACC15", "result-moderate", f"⚠️ Moderate Risk: {prob:.1f}%"
    risk_desc = "This customer shows meaningful churn signals."
else:
    risk_color, risk_class, risk_title = "#34D399", "result-safe", f"✅ Safe: {prob:.1f}% Risk"
    risk_desc = "This customer is unlikely to churn soon."

with sim_right:
    render_html(
        f"""
        <div class="result-card {risk_class}">
        <div class="result-title" style="color:{risk_color};">{risk_title}</div>
        <div class="result-desc">{risk_desc}</div>
        </div>
        """
    )
    if prob >= 40:
        render_html(
            """
            <div class="result-card result-action">
            <div class="result-title" style="color:#34D399; font-size:14px;">🛡️ Action: Retention Offer Recommended</div>
            <div class="result-desc">Proactive intervention can help reduce churn risk for this profile.</div>
            </div>
            """
        )

with sim_left:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    render_html('<div class="chart-card-title">Risk Probability Gauge</div><div class="chart-card-hint">Estimated likelihood of customer churn</div>')
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%", "font": {"size": 44, "color": risk_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#7C8AA5"}},
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#0EA5E9"},
                    {"range": [25, 50], "color": "#34D399"},
                    {"range": [50, 75], "color": "#FACC15"},
                    {"range": [75, 100], "color": "#FB7185"},
                ],
                "threshold": {"line": {"color": "#F8FAFC", "width": 4}, "thickness": 0.82, "value": prob},
            },
        )
    )
    fig_gauge.update_layout(height=280, margin=dict(t=10, b=10, l=30, r=30))
    st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    render_html('<div class="chart-card-title">Why this score?</div><div class="chart-card-hint">How each factor pushed this specific prediction</div>')

    @st.cache_resource
    def get_shap_explainer(_model):
        return shap.TreeExplainer(_model)

    def grouped_from(feature_map):
        return {
            "Tenure (Months)": feature_map.get("tenure", 0.0),
            "Monthly Charges ($)": feature_map.get("MonthlyCharges", 0.0),
            "Contract Type": sum(v for k, v in feature_map.items() if k.startswith("Contract_")),
            "Payment Method": sum(v for k, v in feature_map.items() if k.startswith("PaymentMethod_")),
            "Internet Service": sum(v for k, v in feature_map.items() if k.startswith("InternetService_")),
        }

    shap_rendered = False
    if SHAP_AVAILABLE:
        try:
            explainer = get_shap_explainer(model)
            raw_shap = explainer.shap_values(input_encoded)
            if isinstance(raw_shap, list):
                churn_shap = raw_shap[1][0]
            else:
                arr = np.asarray(raw_shap)
                churn_shap = arr[0, :, 1] if arr.ndim == 3 else arr[0]

            shap_map = dict(zip(feature_cols, churn_shap))
            grouped_shap = grouped_from(shap_map)
            max_abs = max(abs(v) for v in grouped_shap.values()) or 1.0

            for name, value in sorted(grouped_shap.items(), key=lambda kv: abs(kv[1]), reverse=True):
                pct = abs(value) / max_abs * 100
                bar_color = "#FB7185" if value > 0 else "#38BDF8"
                direction = "pushes risk up" if value > 0 else "pushes risk down"
                render_html(
                    f"""
                    <div class="factor-row">
                    <div class="factor-name">{name}</div>
                    <div class="factor-bar-track"><div class="factor-bar-fill" style="width:{pct:.0f}%; background:{bar_color};"></div></div>
                    <div class="factor-tag" style="color:{bar_color};">{direction}</div>
                    </div>
                    """
                )
            shap_rendered = True
        except Exception:
            shap_rendered = False

    if not shap_rendered:
        st.caption("SHAP isn't available in this environment — showing overall model feature importance instead.")
        imp_map = dict(zip(feature_cols, model.feature_importances_))
        grouped_impact = grouped_from(imp_map)
        max_impact = max(grouped_impact.values()) if grouped_impact else 1.0
        for name, value in sorted(grouped_impact.items(), key=lambda kv: kv[1], reverse=True):
            pct_of_max = (value / max_impact * 100) if max_impact > 0 else 0
            if pct_of_max >= 60:
                tag_class, tag_text, bar_color = "tag-high", "High Impact", "#FB7185"
            elif pct_of_max >= 30:
                tag_class, tag_text, bar_color = "tag-medium", "Medium Impact", "#FACC15"
            else:
                tag_class, tag_text, bar_color = "tag-low", "Low Impact", "#38BDF8"
            render_html(
                f"""
                <div class="factor-row">
                <div class="factor-name">{name}</div>
                <div class="factor-bar-track"><div class="factor-bar-fill" style="width:{pct_of_max:.0f}%; background:{bar_color};"></div></div>
                <div class="factor-tag {tag_class}">{tag_text}</div>
                </div>
                """
            )
    st.markdown("</div>", unsafe_allow_html=True)