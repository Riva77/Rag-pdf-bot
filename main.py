from app.pipeline import run_ingestion_pipeline, run_embedding_pipeline, run_query_pipeline
from retrieval.cache_manager import initialize_cache
from retrieval.faiss_search import initialize_faiss_search
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    file_path = "ingestion/coffee.pdf"

    # Initialize cache and FAISS search
    print("  Initializing RAG system...")
    cache_client = initialize_cache()
    faiss_initialized = initialize_faiss_search()
    
    if cache_client and faiss_initialized:
        print("  System ready!")
    else:
        print("  Some components unavailable - continuing...")

    # Run embedding pipeline (includes ingestion + embedding)
    print("  Preparing document embeddings...")
    run_embedding_pipeline(file_path)
    print("  Documents ready for search!")

    # Query pipeline with cache integration
    # This will: 1) Process query, 2) Embed it, 3) Check cache, 4) Return result
    result = run_query_pipeline()
    
    if result["cache_hit"]:
        print(f"\n  Cache Hit! Answer retrieved:")
        print(f"  {result['answer']}")
        print(f"  Query: {result['normalized']}")
    else:
        print(f"\n Cache miss - FAISS search completed!")
        print(f" Processed query: '{result['normalized']}'")
        print(f" Keywords: {result['keywords']}")
        
        # Show FAISS search results
        if 'similar_documents' in result and result['similar_documents']:
            print(f"\n FAISS Search Results:")
            for i, doc in enumerate(result['similar_documents'][:3]):
                print(f"   {i+1}. Score: {doc['similarity_score']:.4f}")
                print(f"      Text: {doc['document'][:150]}...")
                print()
        else:
            print(" No similar documents found in FAISS index")
        
        # Show LLM answer if available
        if 'llm_answer' in result and result.get('llm_success', False):
            print(f"\n  LLM Answer:")
            print(f"  {result['llm_answer']}")
        else:
            print(f"\n  No LLM answer available")
        
        # Show context documents if available
        if 'context_documents' in result and result['context_documents']:
            print(f"\n Context Documents Sent to LLM:")
            print(f"    {len(result['context_documents'])} documents prepared:")
            for i, doc in enumerate(result['context_documents']):
                print(f"   {i+1}. Similarity: {doc['similarity']:.4f} - {doc['content'][:100]}...")


if __name__ == "__main__":
    main()
