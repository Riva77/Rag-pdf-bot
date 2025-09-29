# app/embeddings/loader.py
import pickle
import faiss
import os
from config import FAISS_INDEX_PATH

BASE_DIR = FAISS_INDEX_PATH
INDEX_FILE = os.path.join(BASE_DIR, "index.faiss")
DOCS_FILE = os.path.join(BASE_DIR, "documents.pkl")
META_FILE = os.path.join(BASE_DIR, "metadata.pkl")


def load_faiss_index():
    index = faiss.read_index(INDEX_FILE)

    with open(DOCS_FILE, "rb") as f:
        documents = pickle.load(f)

    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, documents, metadata

