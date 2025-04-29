import os
from src.config.config import DATA_PATH, OUTPUT_STRINGS_PATH, OUTPUT_EMBEDDINGS_PATH, INDEX_PATH
from src.preprocessing.build_strings import build_strings
from src.embeddings.build_embeddings import build_embeddings
from src.indexing.build_faiss_index import build_faiss_index
from src.recommender.recommender import run_recommender


def is_up_to_date(input_file, output_file):
    if not os.path.exists(output_file):
        return False
    return os.path.getmtime(output_file) >= os.path.getmtime(input_file)


def main():
    if not is_up_to_date(DATA_PATH, OUTPUT_STRINGS_PATH):
        print("🔄 Generating text strings...")
        build_strings()
    else:
        print("✅ Text strings are up-to-date, skipping.")

    if not is_up_to_date(OUTPUT_STRINGS_PATH, OUTPUT_EMBEDDINGS_PATH):
        print("🔄 Generating embeddings...")
        build_embeddings()
    else:
        print("✅ Embeddings are up-to-date, skipping.")

    if not is_up_to_date(OUTPUT_EMBEDDINGS_PATH, INDEX_PATH):
        print("🔄 Building FAISS index...")
        build_faiss_index()
    else:
        print("✅ FAISS index is up-to-date, skipping.")

    print("🔍 Running recommender...")
    run_recommender()


if __name__ == "__main__":
    main()
