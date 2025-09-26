import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
from src.features import basic_features
from src.data import leakage_check
import os, json

def value_based_threshold_search(y_true, probs, retain_cost=50, retain_value=300):
    best = {'threshold':0.5, 'value': -1e9}
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        targeted = preds==1
        if targeted.sum()==0:
            continue
        true_churns = y_true[targeted]==1
        total_cost = retain_cost * targeted.sum()
        total_recovered = retain_value * true_churns.sum()
        net = total_recovered - total_cost
        if net > best['value']:
            best = {'threshold': t, 'value': net}
    return best

def train(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    leaked = leakage_check(df)
    if leaked:
        print('Leakage columns detected:', leaked)
        df = df.drop(columns=leaked)
    df = basic_features(df)
    target = 'churn'

    # encode categorical columns automatically (exclude target)
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    if cat_cols:
        print('Encoding categorical cols:', cat_cols)
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    # calibration with Platt (sigmoid)
    calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrator.fit(X_val, y_val)

    prob_val = calibrator.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, prob_val)
    brier = brier_score_loss(y_val, prob_val)
    best = value_based_threshold_search(y_val.values, prob_val)
    print(f'AUC: {auc:.4f}, Brier: {brier:.4f}, best_threshold: {best}')

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, 'model.joblib'))
    joblib.dump(calibrator, os.path.join(output_dir, 'calibrator.joblib'))

    meta = {
        'features': features,
        'threshold': float(best['threshold']) if best.get('threshold') is not None else None,
        'value_est': int(best['value']) if best.get('value') is not None else None
    }
    with open(os.path.join(output_dir,'metadata.json'),'w') as f:
        json.dump(meta, f, indent=2)

    print('Saved artifacts to', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/sample_churn.csv')
    parser.add_argument('--output_dir', default='artifacts')
    args = parser.parse_args()
    train(args.input, args.output_dir)
