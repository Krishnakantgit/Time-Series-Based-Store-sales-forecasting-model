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

python
Copy
Edit
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    verbosity=1
)
Evaluation Metric:
Root Mean Squared Error (RMSE) on the validation set.

✅ Model Evaluation
Validation RMSE is compared to the standard deviation of the sales:

python
Copy
Edit
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)
This provides a good sense of how well the model is performing compared to the typical variance in sales.

📤 Submission File
After generating predictions on the test set:

python
Copy
Edit
y_test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'sales': y_test_pred
})

submission.to_csv('submission.csv', index=False)
🛠️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/sales-prediction-xgboost.git
cd sales-prediction-xgboost
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook or script:

Jupyter Notebook: sales_prediction.ipynb

Python script: python train_model.py

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

GPU acceleration on Kaggle (T4 or P100)

Matplotlib / Seaborn (optional for visualization)

📚 References
Kaggle Store Sales Forecasting Competition

XGBoost Documentation

👨‍💻 Author
Your Name – Krishna Kant Singh
 
