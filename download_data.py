import pandas as pd
import os
import requests

def download_data():
    urls = [
        "https://raw.githubusercontent.com/The-Codernator/Superstore-Sales-Dataset/main/Superstore.csv",
        "https://raw.githubusercontent.com/yajasarora/Superstore-Sales-Analysis-with-Tableau/master/Superstore%20sales%20dataset.csv",
        "https://raw.githubusercontent.com/yannie28/Global-Superstore/master/Global_Superstore(CSV).csv"
    ]
    
    df = None
    for url in urls:
        print(f"Trying URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            print("Success! Downloading data...")
            try:
                # Need to use io.StringIO for reading from text response
                import io
                df = pd.read_csv(io.StringIO(response.text), encoding='latin1')
                break
            except Exception as e:
                print(f"Error parsing CSV: {e}")
        else:
            print(f"Failed with status {response.status_code}")
            
    if df is None:
        print("Could not download the dataset from any of the standard URLs.")
        return
        
    print(f"Data downloaded successfully. Shape: {df.shape}")
    
    # Save raw data
    os.makedirs("data", exist_ok=True)
    raw_path = os.path.join("data", "raw_superstore.csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")
    
    # Basic cleaning: convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace('-', '_').replace(' ', '_') for col in df.columns]
    
    clean_path = os.path.join("data", "superstore_sales.csv")
    df.to_csv(clean_path, index=False)
    print(f"Cleaned data saved to {clean_path}")

if __name__ == "__main__":
    download_data()
