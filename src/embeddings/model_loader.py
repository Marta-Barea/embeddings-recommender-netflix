from sentence_transformers import SentenceTransformer


def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
