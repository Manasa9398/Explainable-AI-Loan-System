import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import shap

# ── STEP A: Load the training CSV ──────────────────────────────
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
# Why: We load the dataset that has Loan_Status labels to train on

# ── STEP B: Drop rows with missing values ──────────────────────
df.dropna(inplace=True)
# Why: ML models cannot handle NaN/empty cells — we remove them

# ── STEP C: Drop Loan_ID (not useful for prediction) ───────────
df.drop("Loan_ID", axis=1, inplace=True)
# Why: Loan_ID is just an identifier, not a pattern the model should learn

# ── STEP D: Encode text columns to numbers ─────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
# Why: ML models only understand numbers, not text like "Male"/"Female"

# ── STEP E: Separate features and target ───────────────────────
X = df.drop("Loan_Status", axis=1)   # input features
y = df["Loan_Status"]                 # what we want to predict
# Why: X = questions, y = answers. Model learns to map X → y

# ── STEP F: Split into train and test sets ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Why: We keep 20% of data hidden so we can test how well the model learned
#      random_state=42 ensures same split every run (reproducibility)

# ── STEP G: Train the XGBoost model ───────────────────────────
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
# Why: XGBoost is a powerful tree-based model, great for tabular data
#      n_estimators=100 means 100 decision trees are built and combined

# ── STEP H: Check accuracy ────────────────────────────────────
preds = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
# Why: We verify the model actually learned something useful

# ── STEP I: Save model and feature names ──────────────────────
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(list(X.columns), "model/features.pkl")
print("✅ Model saved to model/loan_model.pkl")
# Why: We save so Flask can load it without retraining every time

# ── STEP J: Create and save SHAP explainer ────────────────────
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "model/shap_explainer.pkl")
print("✅ SHAP explainer saved to model/shap_explainer.pkl")
# Why: TreeExplainer is optimized for tree-based models like XGBoost
#      We save it to reuse in Flask without recomputing