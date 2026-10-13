import pandas as pd

def load_dataset(path):
    print("\n[MODULE 1] DATASET COLLECTION")
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully: {path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    return df
