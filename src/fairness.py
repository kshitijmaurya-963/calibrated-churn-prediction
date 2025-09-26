import pandas as pd
import numpy as np

def demographic_parity_difference(y_true, y_pred, group):
    # group: array-like of group labels aligned with y arrays
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': group})
    rates = df.groupby('group')['y_pred'].mean()
    return rates.max() - rates.min(), rates.to_dict()
