# =================================================
# predict.py — Inference + Explainability (FINAL)
# =================================================

import numpy as np
import pandas as pd
import joblib
import shap

# =================================================
# 1. Load models & artifacts (PATHS UNCHANGED)
# =================================================
print("Loading models...")

model_A = joblib.load("/content/model_A.pkl")
model_B = joblib.load("/content/model_B.pkl")

FEATURES_A = joblib.load("/content/model_a_features.pkl")
FEATURES_B = joblib.load("/content/model_b_features.pkl")

THRESHOLD_A = joblib.load("/content/model_A_threshold.pkl")
THRESHOLD_B = joblib.load("/content/model_B_threshold.pkl")

# SHAP explainers (base XGB only)
explainer_A = shap.TreeExplainer(model_A.estimator)
explainer_B = shap.TreeExplainer(model_B.estimator)

print("Models & explainers loaded successfully\n")

# =================================================
# 2. Hackathon dataset preprocessing (CRITICAL FIX)
# =================================================
def preprocess_hackathon_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # gender: 1=female, 2=male
    df["male"] = (df["gender"] == 2).astype(int)

    # BMI
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Blood pressure
    df["sysBP"] = df["ap_hi"]
    df["diaBP"] = df["ap_lo"]

    # cholesterol & glucose (ordinal → approx clinical scale)
    df["totChol"] = df["cholesterol"].map({1: 180, 2: 220, 3: 260})
    df["glucose"] = df["gluc"].map({1: 90, 2: 120, 3: 160})

    # smoking
    df["currentSmoker"] = df["smoke"]

    return df[FEATURES_B]

# =================================================
# 3. Feature bounds (ONLY MODIFIABLE FEATURES)
# =================================================
FEATURE_BOUNDS = {
    "BMI": 18,
    "sysBP": 120,
    "diaBP": 80,
    "totChol": 180,
    "glucose": 100,
    "currentSmoker": 0
}

# =================================================
# 4. Core prediction + SHAP
# =================================================
def predict_with_explain(model, explainer, X_user, threshold, top_k=5):

    risk = model.predict_proba(X_user)[0, 1]
    label = int(risk >= threshold)

    shap_vals = explainer.shap_values(X_user)[0]
    contributions = dict(zip(X_user.columns, shap_vals))

    contributions = dict(
        sorted(contributions.items(),
               key=lambda x: abs(x[1]),
               reverse=True)
    )

    top_contrib = dict(list(contributions.items())[:top_k])

    return risk, label, top_contrib

# =================================================
# 5. What-if analysis (MEDICALLY SAFE)
# =================================================
def what_if_analysis(model, X_user, feature):

    if feature not in FEATURE_BOUNDS:
        return None

    current = float(X_user[feature].values[0])
    target = FEATURE_BOUNDS[feature]

    # skip impossible changes
    if feature in ["age", "male"]:
        return None

    X_new = X_user.copy()
    X_new[feature] = target

    old_risk = model.predict_proba(X_user)[0, 1]
    new_risk = model.predict_proba(X_new)[0, 1]

    return {
        "feature": feature,
        "current": current,
        "target": target,
        "risk_reduction_%": round((old_risk - new_risk) * 100, 2)
    }

# =================================================
# 6. STEP-1: Hackathon dataset prediction (MODEL-B)
# =================================================
print("STEP-1: Predicting on hackathon dataset (Model-B)\n")

hackathon_df = pd.read_csv("/content/cardiac_failure_processed.csv")

X_hack = preprocess_hackathon_df(hackathon_df)

hackathon_risk = model_B.predict_proba(X_hack)[:, 1]
hackathon_df["CVD_10yr_risk_%"] = (hackathon_risk * 100).round(2)

hackathon_df.to_csv(
    "predicted_cvd_for_hackathon_dataset.csv",
    index=False
)

print("Saved: predicted_cvd_for_hackathon_dataset.csv\n")

# =================================================
# 7. STEP-2: Interactive user prediction
# =================================================
def yes_no_input(prompt):
    while True:
        v = input(prompt + " (yes/no): ").strip().lower()
        if v in ["yes", "y"]:
            return 1
        if v in ["no", "n"]:
            return 0
        print("Please type yes or no.")

print("Choose model:")
print("1 → Model-B (8 features, fast)")
print("2 → Model-A (full medical model)")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    FEATURES = FEATURES_B
    model = model_B
    explainer = explainer_B
    threshold = THRESHOLD_B
    model_name = "Model-B"
else:
    FEATURES = FEATURES_A
    model = model_A
    explainer = explainer_A
    threshold = THRESHOLD_A
    model_name = "Model-A"

print(f"\nUsing {model_name}\n")

user_data = {}

for f in FEATURES:
    if f in ["male", "currentSmoker"]:
        user_data[f] = yes_no_input(f"Is {f}?")
    else:
        user_data[f] = float(input(f"Enter {f}: "))

X_user = pd.DataFrame([user_data])

risk, label, contrib = predict_with_explain(
    model, explainer, X_user, threshold
)

# =================================================
# 8. Output
# =================================================
print("\n================ RESULT ================")
print(f"10-Year CVD Risk: {risk*100:.2f}%")
print("Prediction:", "HIGH RISK" if label else "LOW RISK")

print("\nTop contributing factors:")
for k, v in contrib.items():
    print(f"- {k}: {'↑ increases risk' if v > 0 else '↓ reduces risk'}")

print("\nActionable risk-reduction insights:")

for feat, shap_val in contrib.items():
    if shap_val > 0:
        info = what_if_analysis(model, X_user, feat)
        if info:
            if feat == "currentSmoker":
                print(
                    f"• Quitting smoking may reduce risk by "
                    f"{info['risk_reduction_%']}%"
                )
            else:
                print(
                    f"• Reducing {feat} from {info['current']} "
                    f"towards {info['target']} "
                    f"may lower risk by ~{info['risk_reduction_%']}%"
                )

print("\n⚠ DISCLAIMER: This is a decision-support tool, not a medical diagnosis.")
