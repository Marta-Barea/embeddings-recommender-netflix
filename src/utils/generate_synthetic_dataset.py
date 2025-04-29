import os
import pandas as pd
import numpy as np


def generate_synthetic_data(
    input_path="data/raw/netflix_titles.csv",
    output_path="data/raw/users_netflix.csv",
    num_users=1000
):
    df_netflix = pd.read_csv(input_path)
    if not {"show_id", "title"}.issubset(df_netflix.columns):
        raise ValueError(
            "The file does not contain 'show_id' and 'title' columns.")

    shows = df_netflix[["show_id", "title"]]
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    synthetic_data = []

    for _, row in shows.iterrows():
        show_id, title = row["show_id"], row["title"]
        for _ in range(np.random.randint(10, 50)):
            synthetic_data.append({
                "show_id": show_id,
                "title": title,
                "user_id": np.random.choice(user_ids),
                "rating": np.round(np.random.uniform(1, 5), 1),
                "timestamp": pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 365), unit="d")
            })

    df_synthetic = pd.DataFrame(synthetic_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)


if __name__ == "__main__":
    generate_synthetic_data()
