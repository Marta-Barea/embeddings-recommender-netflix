import faiss
import numpy as np
import json
from src.config.config import INDEX_PATH, ID_MAPPING_PATH, OUTPUT_EMBEDDINGS_PATH, OUTPUT_STRINGS_PATH
from src.embeddings.model_loader import load_embedding_model


def load_index():
    return faiss.read_index(INDEX_PATH)


def load_id_mapping():
    with open(ID_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings():
    with open(OUTPUT_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_strings():
    with open(OUTPUT_STRINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def recommend(query_embedding, top_k=5):
    index = load_index()
    id_mapping = load_id_mapping()
    strings_mapping = load_strings()

    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    faiss.normalize_L2(query_vector)
    _, indices = index.search(query_vector, top_k)

    array = [id_mapping[i] for i in indices[0]]
    recommendations = [strings_mapping[str(i)] for i in array]

    return recommendations


def run_recommender():
    model = load_embedding_model()
    query_text = input("📝 Enter your query to get recommendations: ")
    query_embedding = model.encode([query_text])[0]
    recommendations = recommend(query_embedding, top_k=5)

    print(f"🔍 Query: {query_text}")
    print("🚀 Top 5 Ranking Recommendations:")
    for i, recommendation in enumerate(recommendations):
        print(f"Rank {i + 1}")
        print(f"{recommendation}")
        print("")


if __name__ == "__main__":
    run_recommender()
