# retrieval/cache_manager.py
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB
import numpy as np
import pickle
import time
import uuid
from typing import Dict, Optional

# Redis connection (persistent across functions)
redis_client = None


def initialize_cache(host: str | None = None, port: int | None = None, db: int | None = None):
    """
    Initialize Redis connection for caching.
    """
    global redis_client
    # Use provided args if given, else fall back to config values
    host_val = host if host is not None else REDIS_HOST
    port_val = port if port is not None else REDIS_PORT
    db_val = db if db is not None else REDIS_DB

    redis_client = redis.Redis(host=host_val, port=port_val, db=db_val)
    try:
        redis_client.ping()
        print(" Connected to Redis cache")
        return redis_client
    except redis.ConnectionError as e:
        print(" Redis connection failed:", e)
        return None


def save_to_cache(processed_query: str, query_embedding: np.ndarray, llm_answer: str):
    """
    Save processed query, its embedding, and LLM answer into cache.
    Stored as Redis hash with a unique UUID key.
    
    Args:
        processed_query: The processed/normalized query (not raw user input)
        query_embedding: Embedding vector of the processed query
        llm_answer: Response from LLM
    """
    if redis_client is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")

    try:
        key = f"query_cache:{uuid.uuid4()}"  # truly unique key
        data = {
            "query": processed_query,  # Store processed query for consistency
            "embedding": pickle.dumps(query_embedding).hex(),  # store as hex string
            "answer": llm_answer,
            "timestamp": str(time.time()),
        }
        redis_client.hset(key, mapping=data)
        print(f" Saved to cache: {processed_query[:40]}...")
        return key
    except (pickle.PickleError, redis.RedisError) as e:
        print(f" Failed to save to cache: {e}")
        return None


def check_cache_by_similarity(query_embedding: np.ndarray, threshold: float = 0.85) -> Optional[Dict]:
    """
    Perform similarity search in cache using cosine similarity.
    Returns the best match if above threshold, else None.
    """
    if redis_client is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")

    try:
        keys = redis_client.keys("query_cache:*")
        if not keys:
            return None

        best_score = -1.0
        best_entry = None

        for key in keys:
            try:
                entry = redis_client.hgetall(key)
                if not entry:
                    continue
                
                # Decode bytes properly for Redis
                if b"embedding" not in entry:
                    print(f" Missing 'embedding' key in cache entry {key}")
                    continue
                    
                if isinstance(entry[b"embedding"], bytes):
                    embedding_hex = entry[b"embedding"].decode('utf-8')
                else:
                    embedding_hex = entry[b"embedding"]
                
                cached_embedding = pickle.loads(bytes.fromhex(embedding_hex))
                sim = cosine_similarity(query_embedding, cached_embedding)
                
                if sim > best_score:
                    best_score = sim
                    best_entry = {
                        "query": entry[b"query"].decode('utf-8') if isinstance(entry[b"query"], bytes) else entry[b"query"],
                        "answer": entry[b"answer"].decode('utf-8') if isinstance(entry[b"answer"], bytes) else entry[b"answer"],
                        "similarity": sim,
                    }
            except (pickle.PickleError, ValueError, KeyError) as e:
                print(f"Failed to process cache entry {key}: {e}")
                continue

        if best_entry and best_score >= threshold:
            print(f" Cache hit (similarity={best_score:.2f}) for query: {best_entry['query']}")
            return best_entry
        else:
            print(" Cache miss (no similar query found)")
            return None
            
    except redis.RedisError as e:
        print(f" Redis error during cache search: {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Optimized version for better performance.
    """
    # Normalize vectors for cosine similarity
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Compute dot product (cosine similarity)
    return np.dot(vec1_norm, vec2_norm)


def get_cache_stats() -> Dict:
    """
    Get basic cache statistics.
    """
    keys = redis_client.keys("query_cache:*")
    return {
        "total_entries": len(keys),
    }


def cleanup_old_entries(max_age_days: int = 30):
    """
    Remove cache entries older than `max_age_days`.
    """
    cutoff = time.time() - (max_age_days * 86400)
    keys = redis_client.keys("query_cache:*")
    removed = 0
    for key in keys:
        entry = redis_client.hgetall(key)
        if not entry:
            continue
        ts = float(entry.get("timestamp", 0))
        if ts < cutoff:
            redis_client.delete(key)
            removed += 1
    print(f" Cleaned up {removed} old cache entries.")


def process_query_with_cache(processed_query: str, query_embedding: np.ndarray, threshold: float = 0.85) -> Dict:
    """
    Main integration function: Check cache first, return cached answer or None.
    
    Args:
        processed_query: The processed/normalized query
        query_embedding: Embedded query vector of the processed query
        threshold: Similarity threshold for cache hits
        
    Returns:
        Dict with 'cache_hit' (bool), 'answer' (str or None), 'similarity' (float or None)
    """
    if redis_client is None:
        print(" Cache not initialized, skipping cache check")
        return {"cache_hit": False, "answer": None, "similarity": None}
    
    # Check cache for similar queries
    cached_result = check_cache_by_similarity(query_embedding, threshold)
    
    if cached_result:
        return {
            "cache_hit": True,
            "answer": cached_result["answer"],
            "similarity": cached_result["similarity"],
            "original_query": cached_result["query"]  # This is now the processed query from cache
        }
    else:
        return {
            "cache_hit": False,
            "answer": None,
            "similarity": None
        }


def save_llm_response_to_cache(processed_query: str, query_embedding: np.ndarray, llm_response: str) -> bool:
    """
    Save LLM response to cache after processing.
    
    Args:
        processed_query: The processed/normalized query
        query_embedding: Embedded query vector of the processed query
        llm_response: Response from LLM
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if redis_client is None:
        print(" Cache not initialized, cannot save response")
        return False
    
    key = save_to_cache(processed_query, query_embedding, llm_response)
    return key is not None

