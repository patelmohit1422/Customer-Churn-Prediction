# Customer Churn Prediction

This is my customer churn prediction project built as part of my AI internship work. The main goal of this project is to predict whether a customer is likely to leave and understand what factors are behind that decision.

I wanted this project to look like a real student-built portfolio project, not just a notebook with some model code. So I kept it structured, practical, and easy to explain.

---

## Problem Statement

Customer churn is a serious business problem. When customers leave, the company loses revenue and has to spend more money to bring in new users.

In this project, I built a machine learning pipeline that:
- predicts churn
- compares multiple models
- shows the main reasons behind churn
- gives business-friendly insights
- includes a simple Streamlit app for prediction

---

## What This Project Does

- Predicts whether a customer will churn
- Compares Logistic Regression, Decision Tree, and Random Forest
- Handles missing values and the `TotalCharges` issue properly
- Avoids data leakage during preprocessing
- Saves EDA plots automatically
- Shows feature importance
- Generates business insights and retention suggestions
- Includes an interactive Streamlit dashboard

---

## Folder Structure

```text
customer-churn-prediction/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── eda/
│   ├── models/
│   ├── plots/
│   └── reports/
├── data/
│   └── raw/
├── models/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── inference.py
│   ├── insights.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **Telco Customer Churn** dataset.

### Main columns used
- `customerID` — customer identifier
- `gender` — customer gender
- `SeniorCitizen` — whether the customer is a senior citizen
- `Partner`, `Dependents` — household status
- `tenure` — number of months the customer stayed
- `PhoneService`, `MultipleLines`
- `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn` — target label

---

## Data Preprocessing

This was one of the most important parts of the project because the dataset is not clean by default.

What I handled:
- converted `TotalCharges` into numeric format
- handled missing values
- encoded categorical columns properly
- scaled numerical features
- split train/test before fitting preprocessing so there is no leakage

This step matters because the model is only useful if the input data is handled correctly.

---

## EDA

I saved the main visualizations automatically during the pipeline.

The EDA includes:
- churn distribution
- churn vs contract type
- churn vs tenure
- churn vs monthly charges
- correlation heatmap

These plots helped me understand which patterns are visible in the dataset before training the model.

---

## Models Trained

I trained three models:

- Logistic Regression
- Decision Tree
- Random Forest

### Metrics used
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

I selected the best model based on **F1 score** because churn prediction is not just about accuracy. It is more important to catch customers at risk than to only look good on paper.

---

## Results

After training, the project generates:
- `artifacts/model_metrics.csv`
- `artifacts/model_metrics.json`
- `artifacts/feature_importance.csv`
- `artifacts/reports/business_insights.md`
- `artifacts/models/telco_churn_model.joblib`

---

## Why I Used F1 Score

I used F1 score because churn is an imbalanced problem. In simple words, if the model misses customers who are actually going to leave, that is a problem.

F1 score gives a better balance between precision and recall, so it is more useful for this kind of business case.

---

## Business Insights

One thing I wanted from this project was not just prediction, but also some useful observations.

The project generates business-friendly insights such as:
- customers on month-to-month plans are more likely to churn
- newer customers need more attention
- high monthly charges can increase churn risk
- payment method can influence retention
- add-on services may help reduce churn

These insights are written in:
`artifacts/reports/business_insights.md`

---

## Feature Importance

I also saved feature importance so it is easier to see what the model is paying attention to.

This helps answer a simple question:
**What is actually driving churn risk?**

Saved file:
`artifacts/feature_importance.csv`

---

## Product Thinking & UX Approach

### Who would use this?

This project is mainly useful for:
- retention teams
- business analysts
- customer success teams
- founders or managers who want a quick churn signal

### How it would be used

A user opens the Streamlit app, enters customer details, and gets:
- churn probability
- risk level
- recommended retention action

### Why I kept the UI simple

I kept the UI simple on purpose. A business user should not need to read code or inspect a notebook just to understand churn risk.

The sidebar input + result cards + recommendations layout is easy to use and easy to explain.

### Simple user flow

1. Open the app
2. Enter customer details
3. Click predict
4. See churn risk
5. Read the suggestions
6. Decide what action to take

---

## My Approach & Learnings

I chose Logistic Regression, Decision Tree, and Random Forest because I wanted a fair comparison between:
- a simple baseline model
- an interpretable model
- a stronger ensemble model

The hardest part for me was dealing with the dataset cleanup, especially `TotalCharges`. It looks like a normal numeric column at first, but it needs proper handling before training.

The biggest thing I learned from this project is that churn prediction is not only about machine learning. It is also about making the result understandable and useful for a business person.

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the main pipeline
```bash
python main.py
```

This will:
- load the dataset
- run EDA
- preprocess the data
- train all models
- compare metrics
- save the best model
- generate feature importance and insights

### 3. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## Screenshots

Add these screenshots after running the app:

- `assets/screenshots/home.png`
- `assets/screenshots/prediction_result.png`
- `assets/screenshots/eda_dashboard.png`

---

## Files Generated After Training

- `artifacts/cleaned_dataset.csv`
- `artifacts/model_metrics.csv`
- `artifacts/feature_importance.csv`
- `artifacts/training_summary.md`
- `artifacts/reports/business_insights.md`
- `artifacts/models/telco_churn_model.joblib`

---

## Future Improvements

A few things I would improve later:
- try more models like XGBoost or LightGBM
- add SHAP for better explainability
- add hyperparameter tuning
- make the app more interactive
- try deployment with FastAPI or Docker

---

## Author

**Mohit Jyani**  
Student | UI/UX + AI Projects

---