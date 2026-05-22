import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Customer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0e1a;
    --card: #111827;
    --card2: #161d2e;
    --border: #1e2d45;
    --accent: #00d4ff;
    --accent2: #ff4f7b;
    --accent3: #7c3aed;
    --text: #e2eaf4;
    --muted: #6b7fa3;
    --safe: #00e5a0;
    --danger: #ff4f7b;
    --radius: 16px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0d1a2e 0%, #111827 60%, #1a0d2e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 60%);
    border-radius: 50%;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 60%);
    border-radius: 50%;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent) 0%, #a78bfa 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.app-subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .accent-bar {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
}
.metric-card .m-label {
    font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem;
}
.metric-card .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 700;
}
.metric-card .m-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }

/* ── Section Heading ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.5rem 0 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Result Card ── */
.result-card {
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
}
.result-card.danger {
    background: linear-gradient(135deg, #2d0a14 0%, #1a0d1a 100%);
    border: 2px solid var(--danger);
    box-shadow: 0 0 40px rgba(255,79,123,0.2);
}
.result-card.safe {
    background: linear-gradient(135deg, #041a12 0%, #0a1a15 100%);
    border: 2px solid var(--safe);
    box-shadow: 0 0 40px rgba(0,229,160,0.2);
}
.result-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;
}
.result-prob {
    font-size: 3.5rem; font-weight: 800;
    font-family: 'Syne', sans-serif;
}
.result-desc { color: var(--muted); font-size: 0.9rem; margin-top: 0.75rem; }

/* ── Gauge ── */
.gauge-wrap { margin: 1.5rem 0; }

/* ── Info Pill ── */
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid;
    margin: 0.2rem;
}
.pill-blue  { color: var(--accent); border-color: var(--accent); background: rgba(0,212,255,0.1); }
.pill-red   { color: var(--danger); border-color: var(--danger); background: rgba(255,79,123,0.1); }
.pill-green { color: var(--safe);   border-color: var(--safe);   background: rgba(0,229,160,0.1); }
.pill-purple{ color: #a78bfa; border-color: #7c3aed; background: rgba(124,58,237,0.1); }

/* ── Input styling overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSelectbox [data-baseweb="select"] { background: var(--card2) !important; }
label { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ── Predict Button ── */
.stButton { width: 100% !important; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card2) !important;
    color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; }

/* ── DataFrame ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    """Search for a file in the script directory and common subdirectories."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, "model", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_model():
    path = find_file("Logistic.pkl")
    if path is None:
        st.error("❌ Could not find `Logistic.pkl`. Place it in the same folder as `churn_app.py`.")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_data():
    path = find_file("customer_churn_prediction_dataset.csv")
    if path is None:
        st.error("❌ Could not find `customer_churn_prediction_dataset.csv`. Place it in the same folder as `churn_app.py`.")
        st.stop()
    return pd.read_csv(path)

model = load_model()
df_raw = load_data()

FEATURES = list(model.feature_names_in_)

def build_input_df(inputs: dict) -> pd.DataFrame:
    """One-hot encode raw inputs to match model features."""
    row = {}
    # Numeric
    row["SeniorCitizen"]  = int(inputs["SeniorCitizen"])
    row["tenure"]         = float(inputs["tenure"])
    row["MonthlyCharges"] = float(inputs["MonthlyCharges"])
    row["TotalCharges"]   = float(inputs["TotalCharges"])

    cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
                "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies","Contract",
                "PaperlessBilling","PaymentMethod"]

    for col in cat_cols:
        val = inputs[col]
        key = f"{col}_{val}"
        for feat in FEATURES:
            if feat.startswith(f"{col}_"):
                row[feat] = 1 if feat == key else 0

    return pd.DataFrame([row])[FEATURES]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">⚡ ChurnSense</div>
    <div class="app-subtitle">AI-powered customer churn prediction · Logistic Regression · Telecom Analytics</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predict Churn", "📊  Dataset Explorer", "🧠  Model Insights"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("")

    with st.sidebar:
        st.markdown("""
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
             background:linear-gradient(90deg,#00d4ff,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin-bottom:0.25rem'>Customer Profile</div>
        <div style='color:#6b7fa3;font-size:0.8rem;margin-bottom:1.5rem'>
        Fill in the customer details below to predict churn probability.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
        gender         = st.selectbox("Gender",          ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen",  ["No", "Yes"])
        partner        = st.selectbox("Has Partner",      ["Yes", "No"])
        dependents     = st.selectbox("Has Dependents",   ["Yes", "No"])

        st.markdown('<div class="section-title">📶 Services</div>', unsafe_allow_html=True)
        phone_service  = st.selectbox("Phone Service",    ["Yes", "No"])
        multi_lines    = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec     = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_bkp     = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot    = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        streaming_mov  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown('<div class="section-title">💳 Account</div>', unsafe_allow_html=True)
        tenure         = st.number_input("Tenure (months)",       min_value=0,  max_value=72,  value=12)
        contract       = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])
        paperless      = st.selectbox("Paperless Billing",["Yes", "No"])
        payment        = st.selectbox("Payment Method",  ["Electronic check", "Mailed check",
                                                          "Bank transfer", "Credit card"])
        monthly_chg    = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total_chg      = st.number_input("Total Charges ($)",   min_value=0.0, max_value=10000.0,
                                          value=round(monthly_chg * tenure, 2), step=1.0)

        predict_btn = st.button("⚡  Predict Churn Risk")

    # ── Main result area
    col_res, col_tips = st.columns([1, 1], gap="large")

    with col_res:
        if predict_btn:
            inputs = {
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone_service, "MultipleLines": multi_lines,
                "InternetService": internet, "OnlineSecurity": online_sec,
                "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_mov, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly_chg, "TotalCharges": total_chg,
            }
            X = build_input_df(inputs)
            prob = model.predict_proba(X)[0][1]
            pred = int(prob >= 0.5)

            pct = int(prob * 100)
            is_churn = pred == 1
            card_cls  = "danger" if is_churn else "safe"
            icon      = "🚨" if is_churn else "✅"
            label     = "HIGH CHURN RISK" if is_churn else "LOW CHURN RISK"
            color     = "#ff4f7b" if is_churn else "#00e5a0"
            desc      = ("This customer shows strong churn signals. "
                         "Immediate retention action recommended."
                         if is_churn else
                         "Customer is likely to stay. Continue delivering value.")

            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-icon">{icon}</div>
                <div class="result-label" style="color:{color}">{label}</div>
                <div class="result-prob" style="color:{color}">{pct}%</div>
                <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.3rem">churn probability</div>
                <div class="result-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            fill_color = "#ff4f7b" if is_churn else "#00e5a0"
            st.markdown(f"""
            <div class="gauge-wrap">
                <div style="display:flex;justify-content:space-between;
                     font-size:0.75rem;color:#6b7fa3;margin-bottom:6px">
                    <span>0%</span><span>Risk Meter</span><span>100%</span>
                </div>
                <div style="background:#1e2d45;border-radius:99px;height:12px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;border-radius:99px;
                         background:linear-gradient(90deg,{'#00e5a0' if not is_churn else '#ff8c42'},{fill_color});
                         transition:width 0.6s ease"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                     font-size:0.75rem;margin-top:6px">
                    <span style="color:#00e5a0">✓ Retain</span>
                    <span style="color:#ff4f7b">⚠ Churn</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key risk tags
            risk_factors = []
            if contract == "Month-to-month":     risk_factors.append(("Month-to-month contract", "red"))
            if internet == "Fiber optic":         risk_factors.append(("Fiber optic user", "red"))
            if payment == "Electronic check":     risk_factors.append(("Electronic check payment", "red"))
            if tenure < 12:                       risk_factors.append(("Low tenure < 12m", "red"))
            if online_sec == "No":                risk_factors.append(("No Online Security", "red"))
            if tech_support == "No":              risk_factors.append(("No Tech Support", "red"))
            if tenure >= 36:                      risk_factors.append(("Long tenure", "green"))
            if contract == "Two year":            risk_factors.append(("2-year contract", "green"))
            if partner == "Yes":                  risk_factors.append(("Has partner", "green"))

            if risk_factors:
                st.markdown('<div class="section-title">🏷 Key Signals</div>', unsafe_allow_html=True)
                pills_html = ""
                for label_txt, cls in risk_factors:
                    c = "pill-red" if cls == "red" else "pill-green"
                    pills_html += f'<span class="pill {c}">{label_txt}</span>'
                st.markdown(f'<div style="margin-top:0.5rem">{pills_html}</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#111827;border:1px dashed #1e2d45;border-radius:16px;
                 padding:3rem;text-align:center;color:#6b7fa3;margin-top:1rem">
                <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
                <div style="font-family:Syne,sans-serif;font-size:1.1rem;
                     color:#e2eaf4;margin-bottom:0.5rem">Ready to Analyze</div>
                <div style="font-size:0.875rem">
                    Fill in the customer profile on the left sidebar<br>
                    and click <strong style="color:#00d4ff">Predict Churn Risk</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_tips:
        st.markdown('<div class="section-title">💡 Retention Playbook</div>', unsafe_allow_html=True)
        tips = [
            ("🤝", "Offer loyalty discounts", "Customers on Month-to-month contracts respond well to discounts for switching to annual plans."),
            ("🛡️", "Promote security add-ons", "Customers without Online Security or Tech Support have significantly higher churn rates."),
            ("📞", "Proactive outreach at 6-12m", "Tenure is the strongest retention predictor. Early engagement prevents early churn."),
            ("💳", "Migrate off electronic checks", "Customers paying by electronic check churn 2× more. Offer auto-pay incentives."),
            ("📦", "Bundle services", "Customers with more services (TV, backup, protection) show lower churn propensity."),
        ]
        for icon2, title2, body2 in tips:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;
                 padding:1.2rem;margin-bottom:0.75rem;display:flex;gap:1rem;align-items:flex-start">
                <div style="font-size:1.5rem;flex-shrink:0">{icon2}</div>
                <div>
                    <div style="font-family:Syne,sans-serif;font-weight:700;
                         color:#e2eaf4;font-size:0.9rem;margin-bottom:0.25rem">{title2}</div>
                    <div style="font-size:0.8rem;color:#6b7fa3;line-height:1.5">{body2}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════
with tab2:
    total    = len(df_raw)
    churned  = int((df_raw["Churn"] == "Yes").sum())
    retained = total - churned
    rate     = churned / total * 100

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#00d4ff,#7c3aed)"></div>
            <div class="m-label">Total Customers</div>
            <div class="m-value" style="color:#00d4ff">{total:,}</div>
            <div class="m-sub">in dataset</div>
        </div>
        <div class="metric-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#ff4f7b,#ff8c42)"></div>
            <div class="m-label">Churned</div>
            <div class="m-value" style="color:#ff4f7b">{churned:,}</div>
            <div class="m-sub">{rate:.1f}% of total</div>
        </div>
        <div class="metric-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#00e5a0,#00d4ff)"></div>
            <div class="m-label">Retained</div>
            <div class="m-value" style="color:#00e5a0">{retained:,}</div>
            <div class="m-sub">{100-rate:.1f}% of total</div>
        </div>
        <div class="metric-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#a78bfa,#7c3aed)"></div>
            <div class="m-label">Avg Monthly Charge</div>
            <div class="m-value" style="color:#a78bfa">${df_raw['MonthlyCharges'].mean():.2f}</div>
            <div class="m-sub">per customer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([1, 2], gap="large")
    with col_f1:
        st.markdown('<div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)
        filter_churn  = st.multiselect("Churn Status",       ["Yes", "No"],          default=["Yes", "No"])
        filter_gender = st.multiselect("Gender",             ["Male", "Female"],     default=["Male", "Female"])
        filter_cont   = st.multiselect("Contract Type",      df_raw["Contract"].unique().tolist(),
                                        default=df_raw["Contract"].unique().tolist())
        search_id = st.text_input("Search Customer ID", placeholder="e.g. 0001-XXXX")

    with col_f2:
        df_filt = df_raw[
            df_raw["Churn"].isin(filter_churn) &
            df_raw["gender"].isin(filter_gender) &
            df_raw["Contract"].isin(filter_cont)
        ]
        if search_id:
            df_filt = df_filt[df_filt["customerID"].str.contains(search_id, case=False, na=False)]

        st.markdown(f'<div class="section-title">📋 Records <span style="color:#6b7fa3;font-size:0.8rem">— {len(df_filt):,} rows</span></div>',
                    unsafe_allow_html=True)

        show_cols = ["customerID","gender","SeniorCitizen","tenure","Contract",
                     "MonthlyCharges","TotalCharges","InternetService","Churn"]
        st.dataframe(
            df_filt[show_cols].head(200).style.applymap(
                lambda v: "color: #ff4f7b; font-weight:600" if v == "Yes" and isinstance(v, str) else
                          "color: #00e5a0; font-weight:600" if v == "No"  and isinstance(v, str) else "",
                subset=["Churn"]
            ),
            width="stretch",
            height=380,
        )


# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔑 Top Feature Importances</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7fa3;font-size:0.85rem;margin-bottom:1rem">'
                'Logistic Regression coefficients — higher absolute value = stronger influence on churn prediction.</div>',
                unsafe_allow_html=True)

    coefs = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=False).head(20)

    col_c1, col_c2 = st.columns([3, 1], gap="large")
    with col_c1:
        max_abs = coefs["Coefficient"].abs().max()
        rows_html = ""
        for _, row in coefs.iterrows():
            c = row["Coefficient"]
            pct_bar = abs(c) / max_abs * 100
            color = "#ff4f7b" if c > 0 else "#00e5a0"
            dir_label = "↑ Churn" if c > 0 else "↓ Retain"
            rows_html += f"""
            <div style="display:flex;align-items:center;gap:1rem;
                 padding:0.7rem 1rem;margin-bottom:0.4rem;
                 background:#111827;border:1px solid #1e2d45;
                 border-radius:10px;transition:border-color 0.2s"
                 onmouseover="this.style.borderColor='{color}'"
                 onmouseout="this.style.borderColor='#1e2d45'">
                <div style="width:180px;font-size:0.78rem;color:#e2eaf4;
                     font-family:'DM Sans';flex-shrink:0">{row['Feature']}</div>
                <div style="flex:1;background:#1e2d45;border-radius:99px;height:8px;overflow:hidden">
                    <div style="width:{pct_bar}%;height:100%;background:{color};border-radius:99px"></div>
                </div>
                <div style="width:60px;text-align:right;font-family:'Syne',sans-serif;
                     font-weight:700;font-size:0.8rem;color:{color}">{c:+.3f}</div>
                <div style="width:70px;text-align:right;font-size:0.72rem;color:#6b7fa3">{dir_label}</div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:14px;padding:1.5rem">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#e2eaf4;
                 font-size:0.95rem;margin-bottom:1rem">How to Read</div>
            <div style="font-size:0.8rem;color:#6b7fa3;line-height:1.7">
                <div style="margin-bottom:0.75rem">
                    <span style="color:#ff4f7b;font-weight:600">Red bars →</span><br>
                    Positive coefficient: feature increases churn likelihood
                </div>
                <div style="margin-bottom:0.75rem">
                    <span style="color:#00e5a0;font-weight:600">Green bars →</span><br>
                    Negative coefficient: feature decreases churn likelihood
                </div>
                <div>
                    <span style="color:#a78bfa;font-weight:600">Bar length →</span><br>
                    Relative importance: longer = stronger signal
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:14px;
             padding:1.5rem;margin-top:1rem">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#e2eaf4;
                 font-size:0.95rem;margin-bottom:1rem">Model Info</div>
            <div style="font-size:0.8rem;color:#6b7fa3;line-height:2">
                <div>Algorithm <span style="float:right;color:#00d4ff">Logistic Regression</span></div>
                <div>Features <span style="float:right;color:#00d4ff">45</span></div>
                <div>Classes <span style="float:right;color:#00d4ff">Binary (0/1)</span></div>
                <div>Encoding <span style="float:right;color:#00d4ff">One-Hot</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)