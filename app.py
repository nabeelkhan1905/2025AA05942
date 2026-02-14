import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

# 1. Page Setup
st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")
st.title("🏦 Bank Marketing Strategy Predictor")
st.markdown("### End-to-End Machine Learning Deployment")

# 2. Model Loading Logic (Standardized to 'model/' folder)
def load_pkl(filename):
    # Ensure this matches your folder name (model vs models)
    with open(f'model/{filename}.pkl', 'rb') as file:
        return pickle.load(file)

# 3. Sidebar - Instructions & Download
st.sidebar.header("Step 1: Get Test Data")
try:
    with open("data/test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Sample Test CSV",
            data=file,
            file_name="sample_bank_data.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.sidebar.error("Sample file not found in data/ folder.")

# 4. Sidebar - File Upload
st.sidebar.header("Step 2: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file for prediction", type=["csv"])

# 5. Model Selection
st.sidebar.header("Step 3: Configuration")
model_choice = st.sidebar.selectbox(
    "Select the Model",
    ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

# 6. Main Logic
if uploaded_file is not None:
    # Load the uploaded data
    input_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", input_df.head())

    if st.sidebar.button("Run Predictions"):
        try:
            # Prepare data for prediction (Preprocessing)
            df_proc = input_df.copy()
            
            # Map target if it exists in the uploaded file (for metrics)
            has_target = 'y' in df_proc.columns
            if has_target:
                actual_y = df_proc['y'].map({'yes': 1, 'no': 0})
                df_proc = df_proc.drop('y', axis=1)

            # Encode categorical features (Exactly as done in training)
            le = LabelEncoder()
            for col in df_proc.select_dtypes(include=['object']).columns:
                df_proc[col] = le.fit_transform(df_proc[col])

            # Load Model and Scaler
            model = load_pkl(model_choice)
            scaler = load_pkl('scaler')

            # Scale data for distance-based models
            if model_choice in ["logistic_regression", "knn"]:
                X_eval = scaler.transform(df_proc)
            else:
                X_eval = df_proc

            # Predictions
            predictions = model.predict(X_eval)
            probabs = model.predict_proba(X_eval)[:, 1]

            # Display Results
            st.success(f"Model {model_choice} executed successfully!")
            
            # Add predictions to the view
            input_df['Prediction'] = ["Term Deposit (Yes)" if p == 1 else "No Subscription" for p in predictions]
            st.write("### Prediction Results", input_df)

            # If the file had the actual 'y' column, show metrics
            if has_target:
                st.divider()
                st.subheader("Performance Metrics (Based on Uploaded Data)")
                
                metrics = {
                    "Accuracy": accuracy_score(actual_y, predictions),
                    "AUC Score": roc_auc_score(actual_y, probabs),
                    "Precision": precision_score(actual_y, predictions),
                    "Recall": recall_score(actual_y, predictions),
                    "F1 Score": f1_score(actual_y, predictions),
                    "MCC Score": matthews_corrcoef(actual_y, predictions)
                }

                cols = st.columns(3)
                m_list = list(metrics.items())
                for i in range(6):
                    cols[i % 3].metric(m_list[i][0], f"{m_list[i][1]:.4f}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(actual_y, predictions)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Ensure your uploaded CSV has the same 16/17 columns as the Bank Marketing dataset.")

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.write("BITS ID: 2025AA05942")