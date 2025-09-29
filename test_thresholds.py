#!/usr/bin/env python3
"""
Test script to experiment with different cache thresholds
"""

from retrieval.cache_manager import initialize_cache, process_query_with_cache
from embeddings.query_embedder import embed_query
from retrieval.query_processor import preprocess_query, normalize_query

def test_threshold(threshold_value, query, description):
    """Test a specific threshold with a query"""
    print(f"\n Testing threshold: {threshold_value} ({threshold_value*100:.0f}%)")
    print(f" Query: '{query}'")
    print(f" Description: {description}")
    
    # Process and embed query
    preprocessed = preprocess_query(query)
    normalized = normalize_query(preprocessed)
    query_vector = embed_query(query)
    
    # Test cache with threshold
    cache_result = process_query_with_cache(normalized, query_vector, threshold=threshold_value)
    
    if cache_result["cache_hit"]:
        print(f" CACHE HIT! Similarity: {cache_result['similarity']:.3f}")
        print(f" Cached query: '{cache_result['original_query']}'")
        print(f" Cached answer: {cache_result['answer'][:100]}...")
    else:
        print(f" Cache miss - would proceed to FAISS search")
    
    return cache_result

def main():
    print("🔧 Initializing cache...")
    cache_client = initialize_cache()
    if not cache_client:
        print(" Cache not available. Please start Redis first.")
        return
    
    # Test queries
    test_queries = [
        ("What is coffee?", "Original question"),
        ("What is coffee?", "Exact same question"),
        ("Tell me about coffee", "Similar question"),
        ("Coffee information", "Shorter version"),
        ("Is coffee healthy?", "Different but related"),
        ("What is tea?", "Completely different")
    ]
    
    # Test different thresholds
    thresholds = [0.95, 0.85, 0.75, 0.65, 0.55]
    
    print(f"\n Testing {len(thresholds)} different thresholds with {len(test_queries)} queries")
    print("=" * 80)
    
    for threshold in thresholds:
        print(f"\n{'='*20} THRESHOLD: {threshold} ({threshold*100:.0f}%) {'='*20}")
        
        for query, description in test_queries:
            test_threshold(threshold, query, description)
            print("-" * 60)
    
    print(f"\n Analysis:")
    print(f"   - Higher threshold (0.95): Very strict, only nearly identical queries")
    print(f"   - Medium threshold (0.75): Balanced, catches similar queries")
    print(f"   - Lower threshold (0.55): Loose, catches loosely related queries")
    print(f"   - Choose based on your use case!")

if __name__ == "__main__":
    main()
