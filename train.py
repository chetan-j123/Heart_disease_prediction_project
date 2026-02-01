import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    recall_score
)
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
data = pd.read_csv("framingham_heart_study.csv")

# -------------------------------------------------
# 2. Handle missing values
# -------------------------------------------------
for col in ["education", "cigsPerDay", "totChol", "BMI", "glucose", "heartRate"]:
    data[col] = data[col].fillna(data[col].median())

data["BPMeds"] = data["BPMeds"].fillna(data["BPMeds"].mode()[0])

# -------------------------------------------------
# 3. Features / Target
# -------------------------------------------------
X = data.drop(columns=["TenYearCHD"])
y = data["TenYearCHD"]

# -------------------------------------------------
# 4. Train / Test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# -------------------------------------------------
# 5. Class imbalance
# -------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight:", scale_pos_weight)

# -------------------------------------------------
# 6. RandomizedSearchCV grid
# -------------------------------------------------
param_grid = {
    "n_estimators": randint(400, 1000),
    "learning_rate": loguniform(0.01, 0.06),
    "max_depth": randint(3, 6),
    "min_child_weight": randint(4, 15),
    "gamma": uniform(0.3, 2.0),
    "subsample": uniform(0.6, 0.25),
    "colsample_bytree": uniform(0.6, 0.25),
    "reg_lambda": loguniform(0.8, 6.0),
}

grid = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,

        # GPU SETTINGS
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor",

        n_jobs=-1
    ),
    param_distributions=param_grid,
    scoring="f1",
    n_iter=100,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    refit=True,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# -------------------------------------------------
# 7. Calibration (NO leakage)
# -------------------------------------------------
best_xgb = grid.best_estimator_

model_A = CalibratedClassifierCV(
    estimator=best_xgb,
    method="sigmoid",
    cv=5
)

model_A.fit(X_train, y_train)

# -------------------------------------------------
# 8. Probabilities
# -------------------------------------------------
y_proba = model_A.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# 9. Threshold sweep
# -------------------------------------------------
best_threshold = None

for t in np.arange(0.10, 0.51, 0.05):
    y_tmp = (y_proba >= t).astype(int)
    r = recall_score(y_test, y_tmp)
    print(f"threshold={t:.2f} | recall={r:.3f}")

    if r >= 0.75 and best_threshold is None:
        best_threshold = t

if best_threshold is None:
    best_threshold = 0.25

print("Selected threshold:", best_threshold)

# -------------------------------------------------
# 10. Final prediction
# -------------------------------------------------
y_pred = (y_proba >= best_threshold).astype(int)

# -------------------------------------------------
# 11. Evaluation
# -------------------------------------------------
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC :", average_precision_score(y_test, y_proba))

# =================================================
# ================= MODEL - B =====================
# =================================================
MODEL_B_FEATURES = [
    "age",
    "male",
    "BMI",
    "sysBP",
    "diaBP",
    "totChol",
    "glucose",
    "currentSmoker"
]

X_b = data[MODEL_B_FEATURES]
y_b = data["TenYearCHD"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_b,
    y_b,
    test_size=0.25,
    stratify=y_b,
    random_state=42
)

scale_pos_weight_b = (yb_train == 0).sum() / (yb_train == 1).sum()

param_dist_b = {
    "n_estimators": randint(100, 500),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(3, 6),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
    "reg_lambda": uniform(0.1, 5.0),
    "reg_alpha": uniform(0, 1.0),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}

random_search_b = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight_b,
        random_state=42,

        # GPU
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor",

        n_jobs=-1
    ),
    param_distributions=param_dist_b,
    n_iter=50,
    scoring="f1",
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    verbose=1,
    refit=True,
    n_jobs=-1
)

random_search_b.fit(Xb_train, yb_train)

xgb_b = random_search_b.best_estimator_

model_b = CalibratedClassifierCV(
    estimator=xgb_b,
    method="sigmoid",
    cv=5
)

model_b.fit(Xb_train, yb_train)

yb_proba = model_b.predict_proba(Xb_test)[:, 1]
yb_pred = (yb_proba >= 0.25).astype(int)

print(confusion_matrix(yb_test, yb_pred))
print(classification_report(yb_test, yb_pred))
print("ROC-AUC:", roc_auc_score(yb_test, yb_proba))
print("PR-AUC :", average_precision_score(yb_test, yb_proba))

# -------------------------------------------------
# 12. Save models
# -------------------------------------------------
joblib.dump(model_A, "model_A.pkl")
joblib.dump(model_b, "model_B.pkl")
joblib.dump(X.columns.tolist(), "model_a_features.pkl")
joblib.dump(MODEL_B_FEATURES, "model_b_features.pkl")
joblib.dump(best_threshold, "model_A_threshold.pkl")
joblib.dump(0.25, "model_B_threshold.pkl")
