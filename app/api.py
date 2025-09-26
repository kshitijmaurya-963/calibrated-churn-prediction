from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os, json

app = FastAPI(title='Churn Inference API')

ARTIFACTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
MODEL_PATH = os.path.join(ARTIFACTS,'model.joblib')
CAL_PATH = os.path.join(ARTIFACTS,'calibrator.joblib')
META_PATH = os.path.join(ARTIFACTS,'metadata.json')

model = None
calibrator = None
metadata = None

class Customer(BaseModel):
    age: int
    gender: str
    tenure_months: int
    monthly_spend: float
    calls: int
    sms: int
    complaints: int
    region: str

@app.on_event('startup')
def load_artifacts():
    global model, calibrator, metadata
    try:
        model = joblib.load(MODEL_PATH)
        calibrator = joblib.load(CAL_PATH)
    except Exception:
        model = None
        calibrator = None
    try:
        with open(META_PATH,'r') as f:
            metadata = json.load(f)
    except Exception:
        metadata = None

def prepare_row_df(row):
    # expects a dataframe with one row coming from Customer
    row = row.copy()
    # mirror src.features.basic_features minimal inline transforms
    row['avg_spend_per_month'] = row['monthly_spend'] / row['tenure_months'].clip(lower=1)
    row['is_new_customer'] = (row['tenure_months'] < 3).astype(int)
    row['engagement_score'] = (row['calls'] * 0.4 + row['sms'] * 0.2 - row['complaints'] * 5)
    if 'customer_id' in row.columns:
        row = row.drop(columns=['customer_id'])
    # encode categorical columns via get_dummies and align with metadata features if present
    row_encoded = pd.get_dummies(row)
    if metadata and 'features' in metadata:
        for c in metadata['features']:
            if c not in row_encoded.columns:
                row_encoded[c] = 0
        X = row_encoded[metadata['features']]
    else:
        # fallback: drop non-numeric columns then use all columns
        X = row_encoded.select_dtypes(include=[float, int])
    return X

@app.post('/predict')
def predict(item: Customer):
    row = pd.DataFrame([item.dict()])
    X = prepare_row_df(row)
    if calibrator is not None:
        prob = float(calibrator.predict_proba(X)[:,1])
        thresh = metadata.get('threshold', 0.5) if metadata else 0.5
        pred = int(prob >= thresh)
        return {'probability': prob, 'predicted_churn': pred, 'threshold': thresh}
    else:
        return {'warning': 'model artifacts not found. Place artifacts/model.joblib and artifacts/calibrator.joblib in artifacts/.'}

@app.post('/batch_predict')
def batch_predict(rows: list[Customer]):
    df = pd.DataFrame([r.dict() for r in rows])
    # same transforms as single-item
    df['avg_spend_per_month'] = df['monthly_spend'] / df['tenure_months'].clip(lower=1)
    df['is_new_customer'] = (df['tenure_months'] < 3).astype(int)
    df['engagement_score'] = (df['calls'] * 0.4 + df['sms'] * 0.2 - df['complaints'] * 5)
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    df_encoded = pd.get_dummies(df)
    if metadata and 'features' in metadata:
        for c in metadata['features']:
            if c not in df_encoded.columns:
                df_encoded[c] = 0
        X = df_encoded[metadata['features']]
    else:
        X = df_encoded.select_dtypes(include=[float,int])
    if calibrator is not None:
        probs = calibrator.predict_proba(X)[:,1].tolist()
        return {'probabilities': probs}
    else:
        return {'warning': 'model artifacts not found.'}
