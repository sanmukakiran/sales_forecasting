📊 Sales Forecasting System

A Machine Learning project that predicts future sales using historical Superstore data.
This project demonstrates a complete end-to-end forecasting pipeline — from raw data ingestion to model evaluation and business-ready predictions.

🚀 Project Overview

This system:

Downloads and cleans raw sales data

Engineers time-series features

Trains a Machine Learning model

Evaluates forecasting accuracy

Generates 30-day future demand predictions

Visualizes business insights

The final output helps businesses:

Plan inventory

Reduce overstocking

Optimize staffing

Improve revenue forecasting

🏗️ Project Architecture

The project follows a structured 3-stage pipeline:

Data Collection → Feature Engineering → Model Training → Forecasting → Visualization
🔹 Step 1: Data Download & Cleaning

Script: download_data.py

Downloads Superstore dataset

Cleans column names

Saves structured CSV files

Outputs:

data/raw_superstore.csv

data/superstore_sales.csv

🔹 Step 2: Feature Engineering

Script: prepare_data.py

Transforms daily sales into ML-ready format:

Aggregates sales by date

Extracts time-based features:

Month

Day of week

Year

Creates lag features

Creates rolling averages

These features help the model learn:

Seasonality

Trends

Sales momentum

Output:

data/processed_sales.csv

🔹 Step 3: Model Training & Forecasting

Script: train_and_forecast.py

Trains a Random Forest Regressor

Splits data into train/test sets

Evaluates performance using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Predicts next 30 days of sales

Outputs:

📈 output/historical_vs_forecast.png
→ Model performance on test data

📊 output/business_forecast_30d.png
→ Future 30-day sales prediction

📌 output/feature_importance.png
→ Most influential features driving predictions

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Requests

OpenPyXL

⚙️ Setup Instructions
1️⃣ Navigate to Project
cd d:\tntern\sales_forecasting
2️⃣ Activate Virtual Environment
.\.venv\Scripts\Activate.ps1
3️⃣ Install Dependencies
pip install -r requirements.txt

(If requirements.txt is not configured, manually install:)

pip install pandas numpy scikit-learn matplotlib seaborn requests openpyxl
▶️ How to Run the Project

Execute scripts in order:

python download_data.py
python prepare_data.py
python train_and_forecast.py
📊 Model Performance

The model is evaluated using:

MAE – measures average prediction error

RMSE – penalizes large errors more heavily

These metrics help assess forecasting reliability for real-world business decisions.

📈 Business Impact

This forecasting system enables:

Inventory optimization

Demand planning

Revenue prediction

Data-driven decision making

It simulates how real businesses use Machine Learning for operational strategy.

📌 Future Improvements

Hyperparameter tuning

Deploy as REST API

Add LSTM / Prophet model comparison

Add dashboard (Streamlit / Flask)

Deploy to cloud (AWS / Render)

👨‍💻 Author

Sanmuka Kiran Mulampaka
Machine Learning & Data Science Enthusiast

