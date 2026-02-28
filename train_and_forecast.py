import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def train_and_forecast():
    input_file = os.path.join("data", "processed_sales.csv")
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Sort by date
    df = df.sort_values('order_date')
    
    features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 
                'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
    target = 'sales'
    
    # Validation split: hold out last 90 days for testing
    split_date = df['order_date'].max() - pd.Timedelta(days=90)
    train_df = df[df['order_date'] < split_date].copy()
    test_df = df[df['order_date'] >= split_date].copy()
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Training on {len(train_df)} days, Testing on {len(test_df)} days.")
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    preds = model.predict(X_test)
    test_df['predicted_sales'] = preds
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model Evaluation -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Predict future 30 days recursively
    print("Forecasting future 30 days...")
    # Retrain on all data
    model.fit(df[features], df[target])
    
    last_known_data = df.tail(30).copy()
    future_dates = pd.date_range(start=df['order_date'].max() + pd.Timedelta(days=1), periods=30)
    
    future_preds = []
    current_df = df.copy()
    
    for date in future_dates:
        # Construct feature row for this date
        new_row = {
            'order_date': date,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': date.dayofweek,
            'is_weekend': int(date.dayofweek in [5, 6]),
            'lag_1': current_df['sales'].iloc[-1],
            'lag_7': current_df['sales'].iloc[-7],
            'lag_30': current_df['sales'].iloc[-30],
            'rolling_mean_7': current_df['sales'].tail(7).mean(),
            'rolling_mean_30': current_df['sales'].tail(30).mean()
        }
        
        # Predict
        new_df = pd.DataFrame([new_row])
        pred = model.predict(new_df[features])[0]
        
        # Add prediction to new_row
        new_row['sales'] = pred
        future_preds.append(new_row)
        
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    future_df = pd.DataFrame(future_preds)
    
    # Plotting 1: Test Validation
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['order_date'][-180:], train_df['sales'][-180:], label='Train Sales (Last 180 days)')
    plt.plot(test_df['order_date'], test_df['sales'], label='Actual Test Sales')
    plt.plot(test_df['order_date'], test_df['predicted_sales'], label='Predicted Test Sales', linestyle='--')
    plt.title('Validation: Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    os.makedirs("output", exist_ok=True)
    plt.savefig('output/historical_vs_forecast.png')
    plt.close()
    
    # Plotting 2: Future Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df['order_date'][-90:], df['sales'][-90:], label='Recent Historical Sales')
    plt.plot(future_df['order_date'], future_df['sales'], label='30-Day Forecast', color='orange', linestyle='--')
    plt.title('Business 30-Day Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/business_forecast_30d.png')
    plt.close()
    
    # Plotting 3: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Sales Forecast')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    plt.close()

    print("Model Evaluation and Plots generated in 'output/' directory.")

if __name__ == "__main__":
    train_and_forecast()
