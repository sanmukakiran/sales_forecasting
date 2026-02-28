import pandas as pd
import os

def prepare_data():
    input_file = os.path.join("data", "superstore_sales.csv")
    output_file = os.path.join("data", "processed_sales.csv")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    date_col = 'order_date'
    if date_col not in df.columns:
        print("Error: 'order_date' column not found.")
        return
        
    print(f"Converting {date_col} to datetime...")
    df[date_col] = pd.to_datetime(df[date_col], format='mixed')
    
    print("Aggregating sales by date...")
    daily_sales = df.groupby(date_col)['sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values(date_col).set_index(date_col)
    
    # Resample to daily frequency filling with 0
    daily_sales = daily_sales.resample('D').sum().fillna(0).reset_index()
    
    print("Creating time-based features...")
    daily_sales['year'] = daily_sales[date_col].dt.year
    daily_sales['month'] = daily_sales[date_col].dt.month
    daily_sales['day'] = daily_sales[date_col].dt.day
    daily_sales['day_of_week'] = daily_sales[date_col].dt.dayofweek
    daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
    
    print("Creating lag features...")
    daily_sales['lag_1'] = daily_sales['sales'].shift(1)
    daily_sales['lag_7'] = daily_sales['sales'].shift(7)
    daily_sales['lag_30'] = daily_sales['sales'].shift(30)
    
    daily_sales['rolling_mean_7'] = daily_sales['sales'].rolling(window=7).mean()
    daily_sales['rolling_mean_30'] = daily_sales['sales'].rolling(window=30).mean()
    
    print(f"Data shape before dropna: {daily_sales.shape}")
    daily_sales = daily_sales.dropna()
    print(f"Data shape after dropna: {daily_sales.shape}")
    
    if len(daily_sales) > 0:
        daily_sales.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    else:
        print("Error: No data left after dropping NaNs! Check your lag features.")

if __name__ == "__main__":
    prepare_data()
