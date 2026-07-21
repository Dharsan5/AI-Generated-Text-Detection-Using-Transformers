import pandas as pd 
from sklearn.model_selection import train_test_split 
def get_unified_splits(df, test_size=0.15, val_size=0.15, seed=42): 
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label']) 
    val_ratio = val_size / (1.0 - test_size) 
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=seed, stratify=train_val_df['label']) 
    return train_df, val_df, test_df 
