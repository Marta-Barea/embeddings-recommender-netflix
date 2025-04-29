import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "../data/raw/netflix_titles.csv")
OUTPUT_STRINGS_PATH = os.path.join(BASE_DIR, "../data/processed/strings.json")
INPUT_STRINGS_PATH = OUTPUT_STRINGS_PATH
OUTPUT_EMBEDDINGS_PATH = os.path.join(
    BASE_DIR, "../data/embeddings/embeddings.json")
INDEX_PATH = os.path.join(BASE_DIR, "../data/indexes/faiss.index")
ID_MAPPING_PATH = os.path.join(BASE_DIR, "../data/indexes/id_mapping.json")
