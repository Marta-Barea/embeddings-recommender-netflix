import pandas as pd


def preprocess_text(text):
    return "" if pd.isna(text) else str(text).strip().lower()
