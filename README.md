# Time Series Based Store sales forecasting model

This repository contains the code for predicting item sales using a machine learning model (XGBoost) trained on a rich time-series dataset from a Kaggle competition. The project focuses on preprocessing, feature engineering, GPU-accelerated model training, and generating submission-ready predictions.

🚀 Project Overview
The goal is to predict daily sales for various product categories across multiple stores using historical data. This is a supervised regression task.

Key techniques used:

Feature engineering (date decomposition, holiday and promotion indicators)

XGBoost Regressor with GPU acceleration

Early stopping and evaluation set

RMSE as the evaluation metric

📂 Dataset
The dataset comes from Kaggle's Store Sales - Time Series Forecasting competition.

Files used:

train.csv: Training data with historical sales

test.csv: Test data (for prediction)

holidays_events.csv: Holiday metadata

oil.csv: Daily oil prices (proxy for economic conditions)

stores.csv: Store information

transactions.csv: Store-level daily transactions

🧪 Model Training
We use XGBoost with the following configuration:

Evaluation Metric:
Root Mean Squared Error (RMSE) on the validation set.

✅ Model Evaluation
Validation RMSE is compared to the standard deviation of the sales:


📤 Submission File
After generating predictions on the test set:
submission.to_csv('submission.csv', index=False)

Upload submission.csv to the Kaggle competition page.

📈 Results
Metric	Value
RMSE (val) - 179
Mean Sales	359.02
Std Dev Sales	1107.29

📌 Technologies Used
    Python
    XGBoost
    Pandas, NumPy, Scikit-learn

GPU acceleration on Kaggle (T4)

Matplotlib / Seaborn (optional for visualization)

👨‍💻 Author
Your Name – Krishna Kant Singh
 
