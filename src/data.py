import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def generate(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    # demographics
    age = rng.randint(18, 80, size=n)
    gender = rng.choice(['M','F'], size=n, p=[0.55,0.45])
    tenure_months = rng.exponential(scale=20, size=n).astype(int)
    monthly_spend = np.round(20 + tenure_months * 0.5 + rng.normal(0,10,size=n),2)
    # usage features
    calls = rng.poisson(10, size=n)
    sms = rng.poisson(20, size=n)
    complaints = rng.binomial(1, 0.05 + (tenure_months<3)*0.15, size=n)
    # a hidden propensity and label
    churn_prob = 0.05 + (tenure_months<6)*0.2 + (monthly_spend < 25)*0.1 + complaints*0.35
    churn = rng.binomial(1, np.clip(churn_prob, 0, 0.95))
    # protected attribute for fairness example
    region = rng.choice(['north','south','east','west'], size=n)

    df = pd.DataFrame({
        'customer_id': ['C{:06d}'.format(i) for i in range(n)],
        'age': age,
        'gender': gender,
        'tenure_months': tenure_months,
        'monthly_spend': monthly_spend,
        'calls': calls,
        'sms': sms,
        'complaints': complaints,
        'region': region,
        'churn': churn
    })
    return df

def leakage_check(df):
    leaked = [c for c in df.columns if 'next' in c or 'future' in c or 'leak' in c]
    return leaked

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/sample_churn.csv')
    parser.add_argument('--n', type=int, default=5000)
    args = parser.parse_args()
    df = generate(args.n)
    leaked = leakage_check(df)
    if leaked:
        print("Potential leakage columns found:", leaked)
    df.to_csv(args.output, index=False)
    print("Saved synthetic data to", args.output)
