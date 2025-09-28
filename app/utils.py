# app/utils.py
import pandas as pd
import os
from datetime import datetime

LOG_CSV = os.path.join(os.path.dirname(__file__), '..', 'predictions_log.csv')

def log_prediction(record: dict):
    df = pd.DataFrame([record])
    if not os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, index=False)
    else:
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)

def read_example_csv(path: str):
    return pd.read_csv(path)