#!/usr/bin/env python3
"""
Test script for cache integration functionality
"""
import numpy as np
from retrieval.cache_manager import initialize_cache, save_to_cache, check_cache_by_similarity, process_query_with_cache
from embeddings.query_embedder import embed_query

def test_cache_functionality():
    """Test the complete cache functionality"""
    print(" Testing Cache Integration")
    print("=" * 50)
    
    # Initialize cache
    print("1. Initializing cache...")
    cache_client = initialize_cache()
    if not cache_client:
        print(" Cache initialization failed - make sure Redis is running")
        return False
    
    # Test queries
    test_queries = [
        "what is artificial intelligence?",
        "what is ai?",  # Should be similar to above
        "tell me about machine learning",
        "what is coffee?",  # Different topic
    ]
    
    print("\n2. Testing cache functionality...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        
        # Process and embed the query (same as pipeline)
        from retrieval.query_processor import preprocess_query, normalize_query
        preprocessed = preprocess_query(query)
        normalized = normalize_query(preprocessed)
        print(f" Processed: '{normalized}'")
        
        # Embed the processed query
        query_vector = embed_query(query)  # embed_query handles preprocessing internally
        print(f" Query embedded: shape {query_vector.shape}")
        
        # Check cache first
        cache_result = process_query_with_cache(normalized, query_vector)
        
        if cache_result["cache_hit"]:
            print(f" Cache HIT! Similarity: {cache_result['similarity']:.3f}")
            print(f" Original query: {cache_result['original_query']}")
            print(f" Cached answer: {cache_result['answer']}")
        else:
            print(" Cache MISS - would need to process with LLM")
            
            # Simulate LLM response (in real app, this would come from LLM)
            mock_llm_response = f"This is a mock response for: {query}"
            
            # Save to cache
            success = save_to_cache(normalized, query_vector, mock_llm_response)
            if success:
                print(f" Saved mock response to cache")
            else:
                print(f" Failed to save to cache")
    
    print(f"\n3. Cache Statistics:")
    from retrieval.cache_manager import get_cache_stats
    stats = get_cache_stats()
    print(f"   - Total cache entries: {stats['total_entries']}")
    
    print(f"\n Cache integration test completed!")
    return True

if __name__ == "__main__":
    test_cache_functionality()
