# app/embeddings/embedder.py
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime

# Save paths (ensure this matches the actual folder name `embeddings/faiss_index`)
BASE_DIR = "embeddings/faiss_index"
INDEX_FILE = os.path.join(BASE_DIR, "index.faiss")
DOCS_FILE = os.path.join(BASE_DIR, "documents.pkl")
META_FILE = os.path.join(BASE_DIR, "metadata.pkl")

# Load embedding model (can be changed later)
model = SentenceTransformer("all-MiniLM-L6-v2")


def _compute_file_hash(file_path: str) -> str | None:
    """Compute SHA256 of a file. Returns None if path does not exist."""
    if not file_path or not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_existing_index():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE) and os.path.exists(META_FILE)):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        documents = pickle.load(f)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, documents, metadata


def _save_index(index, documents, metadata) -> None:
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)


def embed_chunks(chunks, source: str = "coffee.pdf", batch_size: int = 64) -> bool:
    """
    Embed provided chunks and persist into FAISS with documents and metadata.

    Idempotent behavior:
    - If index exists and already contains entries with the same file hash, skip.

    Appending behavior:
    - If index exists and it's a different file, append to existing index and extend docs/metadata.

    Returns True if embedding was performed, False if skipped.
    """
    file_hash = _compute_file_hash(source)

    existing_index, existing_docs, existing_meta = _load_existing_index()
    if existing_meta and file_hash:
        already_indexed = any(m.get("file_hash") == file_hash for m in existing_meta)
        if already_indexed:
            print("ℹ️ Skipping embedding: this PDF appears to be already embedded (hash match).")
            return False

    # Step 1: Embed in batches to control memory
    if not chunks:
        print("⚠️ No chunks to embed. Skipping.")
        return False

    embeddings_list = []
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        batch_emb = model.encode(batch, convert_to_numpy=True)
        embeddings_list.append(batch_emb)
    embeddings = np.vstack(embeddings_list)

    # Step 2: Create or append to FAISS index
    dimension = embeddings.shape[1]
    if existing_index is None:
        index = faiss.IndexFlatL2(dimension)
    else:
        index = existing_index
        if index.d != dimension:
            raise ValueError(f"Existing index dimension {index.d} != model dimension {dimension}")
    index.add(embeddings)

    # Step 3: Prepare documents and metadata
    documents = (existing_docs or []) + list(chunks)
    timestamp = datetime.utcnow().isoformat() + "Z"
    start_id = len(existing_meta or [])
    new_meta = [
        {
            "source": source,
            "file_hash": file_hash,
            "chunk_id": start_id + i,
            "created_at": timestamp,
        }
        for i in range(len(chunks))
    ]
    metadata = (existing_meta or []) + new_meta

    # Step 4: Save all artifacts
    _save_index(index, documents, metadata)

    print(f" Saved/updated FAISS index. Added {len(chunks)} chunks. Total: {len(documents)}")
    return True


def embed_single_text(text: str) -> np.ndarray:

    return model.encode([text], convert_to_numpy=True)[0]