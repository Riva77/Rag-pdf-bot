# app/pipeline.py
from embeddings.query_embedder import embed_query
from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.splitter import chunk_text
from embeddings.embedder import embed_chunks   # import embedding part
from retrieval.query_processor import preprocess_query, normalize_query, extract_keywords
from retrieval.cache_manager import initialize_cache, process_query_with_cache, save_llm_response_to_cache
from retrieval.faiss_search import initialize_faiss_search, search_with_threshold
from llm.simple_gemini import generate_answer

def run_ingestion_pipeline(pdf_path):
    # Step 1: Load
    documents = load_pdf(pdf_path)
    print(f" Loaded {len(documents)} pages from PDF")

    # Step 2: Extract text from documents
    full_text = ""
    for doc in documents:
        full_text += doc.page_content + "\n"
    
    # Step 3: Clean
    cleaned_text = clean_text(full_text)
    print(f" Text cleaned, length: {len(cleaned_text)} characters")

    # Step 4: Split into chunks
    chunks = chunk_text(cleaned_text)
    print("\nOverlap check:")
    print("END(ch1):", chunks[0][-120:])
    print("START(ch2):", chunks[1][:120])
    print(f" Created {len(chunks)} chunks")

    # Debugging samples
    print("\nSample Cleaned Text:", cleaned_text[:300])
    print("\nSample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:100]}...")

    return chunks


def run_embedding_pipeline(pdf_path):
    """
    This function handles embedding stage:
    - Calls ingestion pipeline to get chunks
    - Embeds them using sentence-transformers
    - Saves them into FAISS (index + docs + metadata)
    """
    chunks = run_ingestion_pipeline(pdf_path)   # reuse ingestion output
    embedded = embed_chunks(chunks, source=pdf_path, batch_size=64)
    if embedded:
        print(" Embedding executed.")
    else:
        print(" Embedding skipped (already indexed or no chunks).")


def run_query_pipeline():
    """
    Run the query preprocessing pipeline with cache integration:
    - Ask user for a question
    - Preprocess and normalize query FIRST
    - Embed the processed query
    - Check cache for similar queries
    - If cache hit: return cached answer
    - If cache miss: proceed with FAISS search
    """
    raw_query = input("\n Enter your question: ")

    # Step 1: Preprocess and normalize query FIRST
    print(" Processing query...")
    preprocessed = preprocess_query(raw_query)
    print(f" Preprocessed: {preprocessed}")

    normalized = normalize_query(preprocessed)
    print(f" Normalized: {normalized}")

    # Step 2: Embed the PROCESSED query (this is what we'll use for both cache and FAISS)
    query_vector = embed_query(raw_query)  # embed_query handles preprocessing internally
    print(f" Query vector shape: {query_vector.shape}")
    
    # Step 3: Check cache for similar queries using the processed embedding
    cache_result = process_query_with_cache(normalized, query_vector)
    
    if cache_result["cache_hit"]:
        print(f" Cache Hit! Found similar query with {cache_result['similarity']:.2f} similarity")
        print(f" Original cached query: '{cache_result['original_query']}'")
        print(f" Cached answer: {cache_result['answer']}")
        return {
            "cache_hit": True,
            "answer": cache_result["answer"], # Cached answer
            "query_vector": query_vector,
            "raw_query": raw_query,
            "normalized": normalized
        }
    
    # Step 4: Cache miss - proceed with FAISS search
    print(" Cache miss - proceeding with FAISS search...")
    
    # Extract keywords
    keywords = extract_keywords(normalized)
    print(f" Keywords: {keywords}")
    
    # Search FAISS index for similar documents with quality threshold ( FAISS SEARCH HAPPENS HERE)
    similar_docs = search_with_threshold(query_vector, k=5, threshold=0.7)
    
    if similar_docs:
        print(f" Found {len(similar_docs)} similar documents:")
        for i, doc in enumerate(similar_docs[:3]):  # Show top 3
            print(f"   {i+1}. Score: {doc['similarity_score']:.4f} - {doc['document'][:100]}...")
        
        # Prepare documents for LLM
        top_docs = similar_docs[:3]  # Get top 3 documents
        print(f"\n Preparing {len(top_docs)} documents for LLM processing...")
        
        # Create context for LLM
        context_documents = []
        for i, doc in enumerate(top_docs):
            context_documents.append({
                "rank": i + 1,
                "content": doc['document'],
                "similarity": doc['similarity_score'],
                "metadata": doc.get('metadata', {})
            })
        
        # Generate LLM answer
        print(f"\n  Generating answer with Gemini...")
        answer = generate_answer(normalized, context_documents)
        
        if not answer.startswith("Error:"):
            print(f"  LLM answer generated successfully")
            
            # Save to cache
            save_llm_response_to_cache(normalized, query_vector, answer)
            print(f"  Answer saved to cache!")
            
            return {
                "cache_hit": False,
                "normalized": normalized,
                "keywords": keywords,
                "query_vector": query_vector,
                "raw_query": raw_query,
                "similar_documents": similar_docs,
                "context_documents": context_documents,
                "llm_answer": answer,
                "llm_success": True
            }
        else:
            print(f" LLM generation failed: {answer}")
        
    else:
        print(" No similar documents found in FAISS index")
        context_documents = []
    
    # Debug: Show some vector details
    print(f" Vector details:")
    print(f"   - Type: {type(query_vector)}")
    print(f"   - Dtype: {query_vector.dtype}")
    print(f"   - Size in bytes: {query_vector.nbytes}")
    print(f"   - First 5 values: {query_vector[:5]}")
    print(f"   - Last 5 values: {query_vector[-5:]}")
    print(f"   - Min value: {query_vector.min():.4f}")
    print(f"   - Max value: {query_vector.max():.4f}")
    print(f"   - Mean value: {query_vector.mean():.4f}")

    return {
        "cache_hit": False,
        "normalized": normalized,
        "keywords": keywords,
        "query_vector": query_vector,
        "raw_query": raw_query,
        "similar_documents": similar_docs,
        "context_documents": context_documents
    }

