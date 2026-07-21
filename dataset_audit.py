"""
Dataset audit: check size, nulls, duplicates, and clean.
Run from repo root: python dataset_audit.py
"""
import pandas as pd

CSV = "Datasets/merged/final_dataset.csv"

print("Loading dataset ...")
df = pd.read_csv(CSV, usecols=["text", "label"])

print(f"\n--- Before Cleaning ---")
print(f"Total rows         : {len(df):,}")
print(f"Null text rows     : {df['text'].isna().sum():,}")
print(f"Empty text rows    : {(df['text'].astype(str).str.strip() == '').sum():,}")
print(f"Duplicate rows     : {df.duplicated().sum():,}")
print(f"Duplicate text col : {df['text'].duplicated().sum():,}")
print(f"Label distribution :")
print(df['label'].value_counts().to_string())
