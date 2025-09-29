# RAG System Execution Flow Tree

##  Project Structure & Execution Flow

```
main.py                                     MAIN ENTRY POINT
├──  App Initialization
│   ├── load_dotenv()                      (.env variables)
│   ├── initialize_cache()                 → retrieval/cache_manager.py
│   └── initialize_faiss_search()          → retrieval/faiss_search.py
│       └── embeddings/loader.py            (load FAISS index)
│
├──  Document Processing Pipeline
│   └── run_embedding_pipeline()           → app/pipeline.py
│       └──  Ingestion Phase
│           └── run_ingestion_pipeline()   → app/pipeline.py
│               ├── load_pdf()             → ingestion/pdf_loader.py
│               ├── clean_text()           → ingestion/cleaner.py
│               └── chunk_text()           → ingestion/splitter.py
│       └──  Embedding Phase
│           └── embed_chunks()             → embeddings/embedder.py
│               └── embeddings/loader.py  (save FAISS index)
│
└──  Query Processing Pipeline
    └── run_query_pipeline()              → app/pipeline.py
        ├──  Query Preprocessing
        │   ├── preprocess_query()         → retrieval/query_processor.py
        │   └── normalize_query()          → retrieval/query_processor.py
        │
        ├──  Query Embedding
        │   └── embed_query()              → embeddings/query_embedder.py
        │       └── embeddings/embedder.py  (reuse embed_single_text())
        │
        ├──  Cache Check
        │   └── process_query_with_cache() → retrieval/cache_manager.py
        │       └── check_cache_by_similarity()  (semantic search)
        │
        ├──  FAISS Search (if cache miss)
        │   └── search_with_threshold()    → retrieval/faiss_search.py
        │       └── search_similar_documents()  (semantic search)
        │
        └──  LLM Generation (if cache miss)
            ├── generate_answer()          → llm/simple_gemini.py
            │   └── build_rag_prompt()     → llm/prompt_builder.py
            └── save_llm_response_to_cache() → retrieval/cache_manager.py
                └── save_to_cache()       (store query+answer+embedding)
```

##  Execution Sequence

```
1️. main.py
   ↓
2️. Initialize cache + FAISS
   ↓
3️. Document Processing Pipeline
   ├── ingestion/pdf_loader.py    (PDF → text)
   ├── ingestion/cleaner.py       (clean text)
   ├── ingestion/splitter.py     (text → chunks)
   └── embeddings/embedder.py    (chunks → vectors)
   ↓
4️. Embeddings stored in FAISS
   ↓
5️. Query Pipeline starts
   ├── retrieval/query_processor.py (process user query)
   ├── embeddings/query_embedder.py (embed query)
   ├── retrieval/cache_manager.py  (check cache)
   ├── retrieval/faiss_search.py  (search documents) [if cache miss]
   ├── llm/simple_gemini.py       (generate answer) [if cache miss]
   └── retrieval/cache_manager.py (save to cache)   [if cache miss]
   ↓
6️. Return results to main.py
   ↓
7️. Display results
```

##  File Dependencies

```
Main Components:
├── main.py                    (entry point)
├── app/pipeline.py           (orchestrates workflow)
│
Data Pipeline:
├── ingestion/
│   ├── pdf_loader.py         (PDF parsing)
│   ├── cleaner.py            (text cleaning)
│   └── splitter.py          (text chunking)
│
├── embeddings/
│   ├── embedder.py          (document embeddings)
│   ├── loader.py            (FAISS index management)
│   └── query_embedder.py    (query embeddings)
│
Search & Cache:
├── retrieval/
│   ├── cache_manager.py     (Redis semantic cache)
│   ├── faiss_search.py      (FAISS document search)
│   └── query_processor.py   (query preprocessing)
│
LLM Generation:
└── llm/
    ├── simple_gemini.py     (Gemini API integration)
    └── prompt_builder.py    (prompt construction)
```

##  Key Flow Points

**Document Flow:**
`PDF → Text → Chunks → Embeddings → FAISS Index`

**Query Flow:**
`User Query → Preprocessing → Embedding → Cache Check → FAISS Search (if needed) → LLM Generation (if needed) → Cache Save`

**Return Flow:**
`Pipeline → Main.py → User Display`
