import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title='Uplift Simulator', layout='wide')
st.title('Retention Uplift Simulator')

artifacts = os.path.join(os.getcwd(), 'artifacts')
cal_path = os.path.join(artifacts, 'calibrator.joblib')
meta_path = os.path.join(artifacts, 'metadata.json')

if os.path.exists(cal_path):
    calibrator = joblib.load(cal_path)
    st.success('Loaded calibrator from artifacts/')
else:
    calibrator = None
    st.warning('calibrator not found. Run training first to create artifacts/')

if os.path.exists(meta_path):
    with open(meta_path,'r') as f:
        metadata = json.load(f)
else:
    metadata = None

uploaded = st.file_uploader('Upload a CSV of customers (or leave blank to use sample)', type=['csv'])
if uploaded is None:
    df = pd.read_csv(os.path.join('data','sample_churn.csv'))
else:
    df = pd.read_csv(uploaded)

st.dataframe(df.head())

retain_cost = st.number_input('Retention cost per customer (INR)', value=50)
retain_value = st.number_input('Expected recovered revenue per retained customer (INR)', value=300)

if st.button('Run simulator'):
    from src.features import basic_features
    dfe = basic_features(df.copy())

    if calibrator is None:
        st.error('No calibrator available. Train model first.')
    else:
        # encode categorical via get_dummies and align with saved features
        dfe_encoded = pd.get_dummies(dfe)
        if metadata and 'features' in metadata:
            for c in metadata['features']:
                if c not in dfe_encoded.columns:
                    dfe_encoded[c] = 0
            X = dfe_encoded[metadata['features']]
        else:
            X = dfe_encoded.select_dtypes(include=[float,int])
        probs = calibrator.predict_proba(X)[:,1]
        df['pred_prob'] = probs
        df = df.sort_values('pred_prob', ascending=False).reset_index(drop=True)
        results = []
        for k in [10,50,100,200,500]:
            topk = df.head(k)
            true_churns = topk['churn'].sum()
            total_cost = retain_cost * k
            total_recovered = retain_value * true_churns
            net = total_recovered - total_cost
            results.append({'k':k,'net_gain':net,'true_churns':int(true_churns)})
        st.table(pd.DataFrame(results))
