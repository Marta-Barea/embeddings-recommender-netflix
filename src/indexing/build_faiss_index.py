import faiss
import numpy as np
import json
import os
from src.config.config import OUTPUT_EMBEDDINGS_PATH, INDEX_PATH, ID_MAPPING_PATH


def build_faiss_index():
    with open(OUTPUT_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings_dict = json.load(f)

    show_ids = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values())).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(ID_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(show_ids, f, indent=4)


if __name__ == "__main__":
    build_faiss_index()
