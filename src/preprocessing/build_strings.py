import pandas as pd
import json
import os
from src.config.config import DATA_PATH, OUTPUT_STRINGS_PATH
from src.preprocessing.text_processing import preprocess_text


def build_strings():
    df = pd.read_csv(DATA_PATH)
    columns = df.columns.tolist()
    show_id_col = columns[0]
    feature_columns = columns[1:]
    show_strings = {}

    for _, row in df.iterrows():
        show_id = row[show_id_col]
        string = "\n".join(
            f"{col.capitalize()}: {preprocess_text(row[col])}" for col in feature_columns)
        show_strings[show_id] = string.strip()

    os.makedirs(os.path.dirname(OUTPUT_STRINGS_PATH), exist_ok=True)
    with open(OUTPUT_STRINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(show_strings, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    build_strings()
