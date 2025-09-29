# retrieval/faiss_search.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from embeddings.loader import load_faiss_index

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    This is the SAME function used in cache_manager.py for consistency.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1, higher = more similar)
    """
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0.0

# Global variables to cache loaded index
_faiss_index = None
_documents = None
_metadata = None


def initialize_faiss_search():
    """
    Initialize FAISS search by loading the index, documents, and metadata.
    This should be called once at the start of the application.
    """
    global _faiss_index, _documents, _metadata
    
    try:
        _faiss_index, _documents, _metadata = load_faiss_index()
        print(f" FAISS search initialized - {len(_documents)} documents loaded")
        return True
    except Exception as e:
        print(f" Failed to initialize FAISS search: {e}")
        return False


def search_similar_documents(query_vector: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Search for similar documents in FAISS index using COSINE SIMILARITY.
    Uses FAISS's optimized cosine similarity for better performance.
    
    Args:
        query_vector: The embedded query vector (384-dimensional)
        k: Number of similar documents to return
        
    Returns:
        List of dictionaries containing document text, metadata, and similarity scores
    """
    global _faiss_index, _documents, _metadata
    
    if _faiss_index is None:
        print(" FAISS not initialized. Call initialize_faiss_search() first.")
        return []
    
    try:
        # Normalize query vector for cosine similarity
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        query_vector_norm = query_vector_norm.reshape(1, -1).astype('float32')
        
        # Use FAISS's built-in cosine similarity search
        # FAISS uses L2 distance on normalized vectors = cosine similarity
        scores, indices = _faiss_index.search(query_vector_norm, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(_documents):  # Valid index
                # Convert L2 distance to cosine similarity
                # For normalized vectors: cosine_sim = 1 - (L2_distance² / 2)
                cosine_sim = 1.0 - (score / 2.0)
                
                result = {
                    "rank": i + 1,
                    "document": _documents[idx],
                    "metadata": _metadata[idx] if idx < len(_metadata) else {},
                    "similarity_score": float(cosine_sim),  # Cosine similarity (0-1, higher = more similar)
                    "document_id": idx
                }
                results.append(result)
        
        print(f" FAISS search completed - found {len(results)} similar documents using cosine similarity")
        return results
        
    except Exception as e:
        print(f" FAISS search failed: {e}")
        return []


def search_with_threshold(query_vector: np.ndarray, k: int = 5, threshold: float = 0.7) -> List[Dict]:
    """
    Search for similar documents with a cosine similarity threshold.
    
    Args:
        query_vector: The embedded query vector
        k: Maximum number of documents to return
        threshold: Minimum cosine similarity threshold (0-1, higher = more similar)
        
    Returns:
        List of documents above the threshold
    """
    results = search_similar_documents(query_vector, k)
    
    # Filter by cosine similarity threshold
    filtered_results = []
    for result in results:
        if result["similarity_score"] >= threshold:
            filtered_results.append(result)
    
    print(f" Filtered {len(filtered_results)} documents above cosine similarity threshold {threshold}")
    return filtered_results


def get_document_by_id(doc_id: int) -> Optional[Dict]:
    """
    Get a specific document by its ID.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Document dictionary or None if not found
    """
    global _documents, _metadata
    
    if _documents is None or doc_id >= len(_documents):
        return None
    
    return {
        "document": _documents[doc_id],
        "metadata": _metadata[doc_id] if doc_id < len(_metadata) else {},
        "document_id": doc_id
    }


def get_search_stats() -> Dict:
    """
    Get statistics about the FAISS index.
    
    Returns:
        Dictionary with index statistics
    """
    global _faiss_index, _documents, _metadata
    
    if _faiss_index is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "total_documents": len(_documents) if _documents else 0,
        "index_dimension": _faiss_index.d,
        "index_type": str(type(_faiss_index)),
        "total_metadata": len(_metadata) if _metadata else 0
    }
