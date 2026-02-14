# Bank Marketing Campaign Prediction - ML Assignment 2

* **BITS ID:** 2025AA05942
* **Github repository:** https://github.com/nabeelkhan1905/2025AA05942
* **Streamlit App:** https://2025aa05942-ml-assignment-2.streamlit.app/

## a. Problem Statement
The objective of this project is to develop a machine learning application that predicts whether a client will subscribe to a term deposit (target variable `y`) following a marketing campaign. By analyzing client data and previous contact history, the institution can target potential customers more effectively, thereby increasing the conversion rate and optimizing marketing resources.

## b. Dataset Description
* **Source:** Bank Marketing Dataset from the UCI Machine Learning Repository.
* **Instances:** 45,211 rows.
* **Features:** 17 columns (including age, job, marital status, education, average yearly balance, housing loan, and personal loan).
* **Target:** `y` (Binary: `yes` if the client subscribed, `no` otherwise).
* **Preprocessing:** The dataset contains categorical variables handled via Label Encoding and numerical features normalized using Standard Scaling.



## c. Models Used: Comparison Table
The following metrics represent the performance of the "Big 6" models on the unseen test data (80-20 split).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8400 | 0.7470 | 0.5000 | 0.1250 | 0.2000 | 0.1893 |
| **Decision Tree** | 0.7000 | 0.5179 | 0.1818 | 0..2500 | 02105 | 0.0316 |
| **kNN** | 0..8400 | 0.8616 | 0.5000 | 0.1250 | 0.2000 | 0.1893 |
| **Naive Bayes** | 0.7800 | 0.7649 | 0.3333 | 0.3750 | 0.3529 | 0.2215 |
| **Random Forest (Ensemble)** | 0.8400 | 0.7217 | 0.5000 | 0.1250 | 0.2000 | 0.1893 |
| **XGBoost (Ensemble)** | 0.8000 | 0.6786 | 0.3750 | 0.3750 | 0.3750 | 0.2560 |




---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Maintains high accuracy but shows very poor recall (0.125), failing to identify most potential subscribers. |
| **Decision Tree** | Lowest accuracy and MCC in the group; the high drop in precision suggests the model is struggling with noise in this sample. |
| **kNN** | Matches Logistic Regression in most metrics but achieves a higher AUC, showing better separation power due to scaling. |
| **Naive Bayes** | One of the most balanced models for this data; it achieves a high recall (0.375), successfully identifying more subscribers than the simpler models. |
| **Random Forest (Ensemble)** | Surprisingly conservative on this sample; while accuracy is high, it mirrors the low recall of the baseline Logistic model. |
| **XGBoost (Ensemble)** | **Best Overall Performer.** Achieves the highest MCC (0.2560) and F1 Score (0.3750), showing the best balance between identifying subscribers and maintaining precision. |

---

## Repository Structure
* `app.py`: Main Streamlit application.
* `data/`: `bank-full.csv` and `test_samples_50.csv`.
* `model/`: Python scripts for training and data creation.
* `models/`: All 7 `.pkl` files (6 models + 1 scaler).
