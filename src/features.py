import pandas as pd
import numpy as np

def basic_features(df):
    df = df.copy()
    df['avg_spend_per_month'] = df['monthly_spend'] / (df['tenure_months'].clip(lower=1))
    df['is_new_customer'] = (df['tenure_months'] < 3).astype(int)
    df['engagement_score'] = (df['calls'] * 0.4 + df['sms'] * 0.2 - df['complaints'] * 5)
    # drop identifiers
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    return df
