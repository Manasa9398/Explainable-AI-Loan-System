from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os

app = Flask(__name__)

# ── Load saved model artifacts ─────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
model     = joblib.load(os.path.join(BASE_DIR, "model/loan_model.pkl"))
features  = joblib.load(os.path.join(BASE_DIR, "model/features.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "model/shap_explainer.pkl"))

print("✅ Model loaded successfully")
print(f"✅ Features: {features}")


@app.route("/")
def index():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Build input dataframe in correct feature order
    input_df = pd.DataFrame([data], columns=features)

    # ── Prediction ─────────────────────────────────────────────
    prob       = model.predict_proba(input_df)[0][1]
    prediction = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"

    # ── SHAP ───────────────────────────────────────────────────
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    shap_dict = {feat: round(float(val), 4) for feat, val in zip(features, sv)}
    shap_img  = generate_shap_chart(input_df)

    # ── LIME ───────────────────────────────────────────────────
    lime_explanation = generate_lime(input_df)

    return jsonify({
        "prediction":       prediction,
        "probability":      round(float(prob) * 100, 2),
        "shap_values":      shap_dict,
        "shap_plot":        shap_img,
        "lime_explanation": lime_explanation,
    })


def generate_shap_chart(input_df):
    shap_vals = explainer(input_df)
    vals      = shap_vals.values[0]
    names     = features

    sorted_idx = np.argsort(np.abs(vals))[-8:]
    colors = ['#4ade80' if vals[i] > 0 else '#f87171' for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.barh([names[i] for i in sorted_idx],
            [vals[i]  for i in sorted_idx], color=colors)
    ax.axvline(0, color='#475569', linewidth=0.8)
    ax.set_xlabel("SHAP Value (+ = towards Approval, - = towards Rejection)", color='white', fontsize=9)
    ax.set_title("Why was this decision made?", color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#0f172a')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_lime(input_df):
    background = np.random.rand(100, len(features)).astype(np.float64)

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data = background,
        feature_names = features,
        class_names   = ["Rejected", "Approved"],
        mode          = "classification"
    )

    explanation = lime_exp.explain_instance(
        data_row   = input_df.values[0].astype(np.float64),
        predict_fn = model.predict_proba,
        num_features = 8
    )

    result = []
    for feat, weight in explanation.as_list():
        direction = "✅ Pushes APPROVAL" if weight > 0 else "❌ Pushes REJECTION"
        result.append(f"{feat}  →  {direction}  (score: {round(weight, 4)})")
    return result


if __name__ == "__main__":
    app.run(debug=True)
