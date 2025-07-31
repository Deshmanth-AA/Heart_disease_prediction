
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import joblib

st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")
st.title("💓 Heart Disease Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()
st.sidebar.header("1. Data Overview")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)

# EDA Section
st.sidebar.header("2. Exploratory Data Analysis")
st.subheader("Exploratory Data Analysis")

with st.expander("🔍 Histogram Viewer"):
    col = st.selectbox("Select feature for histogram", df.columns)
    bins = st.slider("Number of bins", 5, 100, 20)
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=bins, kde=True, ax=ax)
    st.pyplot(fig)

with st.expander("📊 Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Model Training
st.sidebar.header("3. Model Training")
model_type = st.sidebar.selectbox("Select model", ["Random Forest", "Logistic Regression", "Decision Tree"])
test_size = st.sidebar.slider("Test size (%)", 10, 50, 30)

y = df['target']
X = df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

model_filename = "models/trained_model.pkl"

if model_type == "Random Forest":
    model = RandomForestClassifier()
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = DecisionTreeClassifier()

if st.sidebar.button("Train Model"):
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    st.success("Model trained and saved!")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {acc:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=importances.values, y=importances.index, ax=ax)
        st.pyplot(fig)

# Load model if exists
try:
    model = joblib.load(model_filename)
except:
    model = RandomForestClassifier().fit(X_train, y_train)

# Live prediction
st.sidebar.header("4. Live Prediction")
st.subheader("Predict Heart Disease for a Single Patient")

input_data = {}
for col in X.columns:
    val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data[col] = val

if st.button("Predict Heart Disease"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
    st.write(f"**Predicted Class:** {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
    if prob is not None:
        st.write(f"**Prediction Probability:** {prob:.2f}")

# Batch prediction
st.sidebar.header("5. Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV for batch prediction", type="csv")
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        if all(col in batch_data.columns for col in X.columns):
            batch_preds = model.predict(batch_data)
            result_df = batch_data.copy()
            result_df['Prediction'] = batch_preds
            st.subheader("Batch Prediction Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
        else:
            st.error("Uploaded CSV does not have the required features.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
