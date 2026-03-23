# 💡 Explainable AI Loan Prediction System

## 📌 Overview

This project is a **Loan Prediction Web Application** that uses Machine Learning along with **Explainable AI (XAI)** techniques to provide transparent and interpretable predictions.

The system predicts whether a loan application will be **approved or rejected** and explains the decision using **SHAP and LIME**, helping users understand the reasoning behind the prediction.

---

## 🚀 Features

* 📊 Loan approval prediction using ML model
* 🔍 Explainable AI using SHAP & LIME
* 📈 Feature importance visualization
* 🌐 Web interface using Flask
* ⚡ Fast and interactive predictions

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Flask
* **Libraries:**  Pandas, NumPy, joblib
* **Explainability Tools:** SHAP, LIME
* **Frontend:** HTML, CSS, JavaScript
* **IDE:** VS Code

---

## 📂 Project Structure

```
Explainable-AI-Loan-System/
│
├── model/                # Saved models and explainers
│   ├── loan_model.pkl
│   ├── features.pkl
│   ├── shap_explainer.pkl
│
├── static/               # CSS and JavaScript files
│   ├── style.css
│   ├── script.js
│
├── templates/            # HTML templates
│   └── index.html
│
├── app.py                # Flask application
├── train_model.py        # Model training script
├── train.csv             # Training dataset
├── test.csv              # Testing dataset
├── requirements.txt      # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup


. Navigate to the project folder:

```
cd Explainable-AI-Loan-System
```

3. Create virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate   # Windows
```

4. Install dependencies:

```
pip install -r requirements.txt
```

5. Run the application:

```
python app.py
```

6. Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 How It Works

1. User inputs loan details through the web interface
2. Model predicts loan approval status
3. SHAP and LIME generate explanations
4. Results are displayed with feature impact insights

---

## 🎯 Key Highlight

This project focuses on **Explainable AI**, ensuring that:

* Model decisions are transparent
* Users can understand *why* a loan is approved/rejected
* Trust in AI systems is improved

---

## 🔮 Future Enhancements

* Deploy application on cloud (Render/Heroku)
* Improve UI/UX design
* Add user authentication
* Use advanced ML models (XGBoost, Neural Networks)

---

## 📸 Output

* Loan prediction result (Approved/Rejected)
* Feature contribution explanation
* Model decision insights

---

## ⭐ Project Goal

To build a **real-world financial decision system** that combines Machine Learning with Explainable AI to improve transparency and trust.
