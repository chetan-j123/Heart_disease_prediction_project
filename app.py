# =================================================
# app.py — Streamlit App
# =================================================

import streamlit as st
import pandas as pd
import joblib
import shap
import requests
from pathlib import Path

# =================================================
# Streamlit config
# =================================================
st.set_page_config(
    page_title="CVD Risk Predictor",
    layout="centered"
)

st.title("❤️ 10-Year Cardiovascular Disease Risk Predictor")
st.caption("ML decision-support with SHAP explainability")

# =================================================
# EXACT GitHub RAW LINKS (from predict.ipynb)
# =================================================
URLS = {
    "model_A": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_A.pkl",
    "model_B": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_B.pkl",
    "threshold_A": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_A_threshold.pkl",
    "threshold_B": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_B_threshold.pkl",
    "features_A": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_a_features.pkl",
    "features_B": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/model_b_features.pkl",
    "hackathon_csv": "https://github.com/chetan-j123/Heart_decision_prediction_project/raw/master/cardiac_failure_processed.csv"
}

# =================================================
# Local cache (Streamlit-safe)
# =================================================
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_from_github(url):
    file_path = CACHE_DIR / url.split("/")[-1]
    if not file_path.exists():
        r = requests.get(url)
        r.raise_for_status()
        file_path.write_bytes(r.content)
    return file_path

# =================================================
# Load models & artifacts
# =================================================
@st.cache_resource
def load_models():
    model_A = joblib.load(load_from_github(URLS["model_A"]))
    model_B = joblib.load(load_from_github(URLS["model_B"]))

    FEATURES_A = joblib.load(load_from_github(URLS["features_A"]))
    FEATURES_B = joblib.load(load_from_github(URLS["features_B"]))

    THRESHOLD_A = joblib.load(load_from_github(URLS["threshold_A"]))
    THRESHOLD_B = joblib.load(load_from_github(URLS["threshold_B"]))

    explainer_A = shap.TreeExplainer(model_A.estimator)
    explainer_B = shap.TreeExplainer(model_B.estimator)

    return (
        model_A, model_B,
        FEATURES_A, FEATURES_B,
        THRESHOLD_A, THRESHOLD_B,
        explainer_A, explainer_B
    )

(
    model_A, model_B,
    FEATURES_A, FEATURES_B,
    THRESHOLD_A, THRESHOLD_B,
    explainer_A, explainer_B
) = load_models()

# =================================================
# Feature bounds and default values
# =================================================
FEATURE_CONFIG = {
    # Model-A features (Full medical model)
    "model_A": {
        "age": {"min": 20, "max": 80, "default": 45, "step": 1},
        "male": {"options": [0, 1], "default": 0},
        "education": {"min": 1, "max": 4, "default": 2, "step": 1},
        "currentSmoker": {"options": [0, 1], "default": 0},
        "cigsPerDay": {"min": 0, "max": 60, "default": 0, "step": 1},
        "BPMeds": {"options": [0, 1], "default": 0},
        "prevalentStroke": {"options": [0, 1], "default": 0},
        "prevalentHyp": {"options": [0, 1], "default": 0},
        "diabetes": {"options": [0, 1], "default": 0},
        "totChol": {"min": 100, "max": 400, "default": 200, "step": 1},
        "sysBP": {"min": 80, "max": 250, "default": 120, "step": 1},
        "diaBP": {"min": 50, "max": 150, "default": 80, "step": 1},
        "BMI": {"min": 15, "max": 50, "default": 25, "step": 0.1},
        "heartRate": {"min": 40, "max": 120, "default": 72, "step": 1},
        "glucose": {"min": 50, "max": 300, "default": 100, "step": 1}
    },
    # Model-B features (Fast, 8 features)
    "model_B": {
        "age": {"min": 20, "max": 80, "default": 45, "step": 1},
        "male": {"options": [0, 1], "default": 0},
        "currentSmoker": {"options": [0, 1], "default": 0},
        "totChol": {"min": 100, "max": 400, "default": 200, "step": 1},
        "sysBP": {"min": 80, "max": 250, "default": 120, "step": 1},
        "diaBP": {"min": 50, "max": 150, "default": 80, "step": 1},
        "BMI": {"min": 15, "max": 50, "default": 25, "step": 0.1},
        "glucose": {"min": 50, "max": 300, "default": 100, "step": 1}
    }
}

# Healthy baseline values for what-if analysis
HEALTHY_BASELINE = {
    "BMI": 22,
    "sysBP": 120,
    "diaBP": 80,
    "totChol": 180,
    "glucose": 90,
    "currentSmoker": 0,
    "cigsPerDay": 0
}

# =================================================
# Prediction + SHAP
# =================================================
def predict_with_explain(model, explainer, X_user, threshold, top_k=5):
    # Predict probability
    risk = model.predict_proba(X_user)[0, 1]
    label = int(risk >= threshold)

    # SHAP values
    shap_vals = explainer.shap_values(X_user)[0]
    contrib = dict(zip(X_user.columns, shap_vals))
    
    # Sort by absolute contribution
    contrib = dict(
        sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return risk, label, dict(list(contrib.items())[:top_k])

# =================================================
# What-if analysis
# =================================================
def what_if_analysis(model, X_user, feature):
    if feature not in HEALTHY_BASELINE or feature in ["age", "male", "education"]:
        return None

    X_new = X_user.copy()
    X_new[feature] = HEALTHY_BASELINE[feature]

    old = model.predict_proba(X_user)[0, 1]
    new = model.predict_proba(X_new)[0, 1]

    return round((old - new) * 100, 2)

# =================================================
# Model selection with session state
# =================================================
st.subheader("Select Model")

# Initialize session state for model selection and input data
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Model-B (Fast, 8 features)"
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}

choice = st.radio(
    "Choose model:",
    ["Model-B (Fast, 8 features)", "Model-A (Full medical model)"],
    key="model_choice"
)

# Update model selection and reset defaults when model changes
if st.session_state.model_choice != st.session_state.selected_model:
    st.session_state.selected_model = st.session_state.model_choice
    st.session_state.input_values = {}  # Clear input values on model change

if choice.startswith("Model-B"):
    model, explainer = model_B, explainer_B
    FEATURES, threshold = FEATURES_B, THRESHOLD_B
    model_key = "model_B"
    st.info("⚡ **Model-B**: Fast prediction with 8 key features (Age, Sex, Smoking, Cholesterol, Blood Pressure, BMI, Glucose)")
else:
    model, explainer = model_A, explainer_A
    FEATURES, threshold = FEATURES_A, THRESHOLD_A
    model_key = "model_A"
    st.info("🏥 **Model-A**: Comprehensive medical model with 13 features including medical history, smoking details, and education level")

# =================================================
# User input with proper ranges
# =================================================
st.subheader("Patient Details")

user_data = {}

with st.form("input_form"):
    for f in FEATURES:
        # Check if feature exists in config, use default config if not
        if f in FEATURE_CONFIG[model_key]:
            config = FEATURE_CONFIG[model_key][f]
        else:
            # Default configuration for any unexpected features
            if f in ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]:
                config = {"options": [0, 1], "default": 0}
            elif f == "education":
                config = {"min": 1, "max": 4, "default": 2, "step": 1}
            elif f == "cigsPerDay":
                config = {"min": 0, "max": 60, "default": 0, "step": 1}
            elif f == "heartRate":
                config = {"min": 40, "max": 120, "default": 72, "step": 1}
            else:
                # Generic numeric config
                config = {"min": 0, "max": 100, "default": 50, "step": 1}
        
        # Get current value from session state or use default
        current_val = st.session_state.input_values.get(f, config.get("default", 0))
        
        if "options" in config:  # Binary/categorical features
            # Make sure current value is in options
            if current_val not in config["options"]:
                current_val = config["default"]
            
            user_data[f] = st.selectbox(
                f.replace("_", " ").title(),
                options=config["options"],
                index=config["options"].index(current_val),
                format_func=lambda x: "Yes" if x == 1 else "No",
                key=f"input_{f}"
            )
        else:  # Numerical features
            # Make sure current value is within bounds
            current_val = max(config["min"], min(config["max"], float(current_val)))
            
            if f == "education":
                # Special handling for education with meaningful labels
                education_labels = {
                    1: "Less than High School",
                    2: "High School/GED",
                    3: "Some College/Vocational",
                    4: "College Degree or Higher"
                }
                
                # Ensure current_val is integer for indexing
                idx_val = int(current_val)
                default_index = list(education_labels.keys()).index(idx_val) if idx_val in education_labels else 1

                user_data[f] = st.selectbox(
                    "Education Level",
                    options=list(education_labels.keys()),
                    index=default_index,
                    format_func=lambda x: education_labels[x],
                    key=f"input_{f}"
                )
            else:
                user_data[f] = st.slider(
                    f.replace("_", " ").title(),
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(current_val),
                    step=float(config["step"]),
                    help=f"Range: {config['min']} to {config['max']}",
                    key=f"input_{f}"
                )
    
    # Form submit buttons
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button("📊 Predict Risk", use_container_width=True)
    with col2:
        reset_button = st.form_submit_button("🔄 Reset to Defaults", type="secondary", use_container_width=True)

# Handle reset button
if reset_button:
    st.session_state.input_values = {}
    st.rerun()

# Store current input values in session state
for f in FEATURES:
    st.session_state.input_values[f] = user_data.get(f, FEATURE_CONFIG.get(model_key, {}).get(f, {}).get("default", 0))

# =================================================
# Output - Only show when submit button is clicked
# =================================================
if submit_button:
    X_user = pd.DataFrame([user_data])

    risk, label, contrib = predict_with_explain(
        model, explainer, X_user, threshold
    )

    st.markdown("---")
    
    # Display risk with color coding
    risk_percent = risk * 100
    if risk_percent < 10:
        risk_color = "green"
    elif risk_percent < 20:
        risk_color = "orange"
    else:
        risk_color = "red"
    
    st.metric("10-Year CVD Risk", f"{risk_percent:.1f}%", delta_color="off")

    if label:
        st.error("⚠️ **HIGH RISK** - Consider consulting a healthcare provider")
    else:
        st.success("✅ **LOW RISK** - Maintain healthy lifestyle")

    st.subheader("📈 Top Risk Contributors")
    for k, v in contrib.items():
        direction = "🔺 Increases risk" if v > 0 else "🟢 Decreases risk"
        feature_name = k.replace("_", " ").title()
        st.write(f"**{feature_name}**: {direction}")

    st.subheader("💡 Actionable Insights")
    shown = False
    
    # Check each modifiable risk factor
    modifiable_features = ["BMI", "sysBP", "diaBP", "totChol", "glucose", "currentSmoker", "cigsPerDay"]
    
    for feat in modifiable_features:
        if feat in FEATURES and feat in HEALTHY_BASELINE:
            current_val = user_data[feat]
            healthy_val = HEALTHY_BASELINE[feat]
            
            # Only show if current value is worse than healthy baseline
            is_worse = False
            if feat == "currentSmoker" and current_val == 1:
                is_worse = True
            elif feat == "cigsPerDay" and current_val > 0:
                is_worse = True
            elif feat not in ["currentSmoker", "cigsPerDay"] and current_val > healthy_val:
                is_worse = True
                
            if is_worse:
                reduction = what_if_analysis(model, X_user, feat)
                if reduction and reduction > 0:
                    shown = True
                    
                    if feat == "currentSmoker":
                        st.write(f"🚭 **Quit Smoking**: ~{reduction}% risk reduction")
                    elif feat == "cigsPerDay":
                        st.write(f"🚭 **Reduce cigarettes from {current_val:.0f} to 0 per day**: ~{reduction}% risk reduction")
                    elif feat == "BMI":
                        target = max(healthy_val, 18.5)  # At least 18.5
                        st.write(f"⚖️ **Reduce BMI from {current_val:.1f} to {target:.1f}**: ~{reduction}% risk reduction")
                    elif feat == "sysBP":
                        st.write(f"❤️ **Lower Systolic BP from {current_val:.0f} to {healthy_val:.0f} mmHg**: ~{reduction}% risk reduction")
                    elif feat == "diaBP":
                        st.write(f"❤️ **Lower Diastolic BP from {current_val:.0f} to {healthy_val:.0f} mmHg**: ~{reduction}% risk reduction")
                    elif feat == "totChol":
                        st.write(f"🥑 **Lower Cholesterol from {current_val:.0f} to {healthy_val:.0f} mg/dL**: ~{reduction}% risk reduction")
                    elif feat == "glucose":
                        st.write(f"🍃 **Lower Glucose from {current_val:.0f} to {healthy_val:.0f} mg/dL**: ~{reduction}% risk reduction")

    if not shown:
        st.write("✅ **Good news!** No major modifiable risks detected above healthy thresholds.")

    # Display current values vs healthy targets
    st.subheader("📊 Current vs Healthy Targets")
    
    # Create appropriate columns based on available features
    modifiable_in_model = [f for f in modifiable_features if f in FEATURES]
    num_cols = min(3, len(modifiable_in_model))
    if num_cols > 0:
        cols = st.columns(num_cols)
        
        for idx, feat in enumerate(modifiable_in_model):
            with cols[idx % num_cols]:
                current_val = user_data[feat]
                healthy_val = HEALTHY_BASELINE.get(feat, 0)
                
                if feat == "currentSmoker":
                    st.metric(
                        "Smoking",
                        "Yes" if current_val == 1 else "No",
                        delta="Quit" if current_val == 1 else "Good",
                        delta_color="inverse"
                    )
                elif feat == "cigsPerDay":
                    st.metric(
                        "Cigarettes/Day",
                        f"{current_val:.0f}",
                        delta="Target: 0",
                        delta_color="inverse" if current_val > 0 else "normal"
                    )
                else:
                    delta = f"Target: {healthy_val:.1f}"
                    st.metric(
                        feat,
                        f"{current_val:.1f}",
                        delta=delta,
                        delta_color="normal"
                    )

    st.caption(
        "⚠️ **Disclaimer**: This is a decision-support tool only. Not a substitute for professional medical diagnosis or advice. Always consult with a qualified healthcare provider."
    )