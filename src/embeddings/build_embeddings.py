import json
import os
from src.config.config import INPUT_STRINGS_PATH, OUTPUT_EMBEDDINGS_PATH
from src.embeddings.model_loader import load_embedding_model


def load_show_strings():
    with open(INPUT_STRINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)


def build_embeddings():
    model = load_embedding_model()
    show_strings = load_show_strings()
    show_ids = list(show_strings.keys())
    texts = list(show_strings.values())

    embeddings = generate_embeddings(model, texts)
    embeddings_dict = {show_id: embedding.tolist()
                       for show_id, embedding in zip(show_ids, embeddings)}

    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_PATH), exist_ok=True)
    with open(OUTPUT_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings_dict, f, indent=4)


if __name__ == "__main__":
    build_embeddings()
