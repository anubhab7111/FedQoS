import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset and perform basic cleaning."""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates()
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using linear regression."""
    # Impute jitter from delay
    if df['jitter'].isnull().any():
        clean_jitter = df[['jitter', 'delay']].dropna()
        model_jitter = LinearRegression()
        model_jitter.fit(clean_jitter[['delay']], clean_jitter['jitter'])
        missing_jitter = df['jitter'].isnull()
        df.loc[missing_jitter, 'jitter'] = model_jitter.predict(df.loc[missing_jitter, ['delay']])
    
    # Impute distance from pdr
    if df['distance'].isnull().any():
        clean_dist = df[['distance', 'pdr']].dropna()
        model_dist = LinearRegression()
        model_dist.fit(clean_dist[['pdr']], clean_dist['distance'])
        missing_dist = df['distance'].isnull()
        df.loc[missing_dist, 'distance'] = model_dist.predict(df.loc[missing_dist, ['pdr']])
    
    return df

def smooth_features(df: pd.DataFrame, cols: list, filter_len: int) -> pd.DataFrame:
    """Apply rolling mean smoothing to specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].rolling(window=filter_len).mean()
    
    df = df.bfill().ffill()
    return df

def select_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Subset the dataframe to the specified feature columns."""
    return df[feature_cols]
