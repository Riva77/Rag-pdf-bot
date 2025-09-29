# embeddings/query_embedder.py
from retrieval.query_processor import preprocess_query, normalize_query
from embeddings.embedder import embed_single_text

def embed_query(raw_query: str):
    """
    Process and embed a user query so it aligns with document embeddings.
    """
    # Step 1: Preprocess
    preprocessed = preprocess_query(raw_query)
    
    # Step 2: Normalize
    normalized = normalize_query(preprocessed)
    
    # Step 3: Embed using the same model as documents
    query_vector = embed_single_text(normalized)
    
    return query_vector
