import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on PYTHONPATH so imports like `retrieval.*` work when running from ui/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import existing pipeline pieces
from retrieval.cache_manager import initialize_cache, process_query_with_cache, save_llm_response_to_cache, clear_cache
from retrieval.faiss_search import initialize_faiss_search, search_with_threshold, search_similar_documents
from embeddings.query_embedder import embed_query
from retrieval.query_processor import preprocess_query, normalize_query, extract_keywords
from llm.simple_gemini import generate_answer

load_dotenv()

st.set_page_config(page_title="RAG PDF Bot", page_icon="📄", layout="wide")

# Defaults when system is not initialized via the UI button
DEFAULT_CACHE_THRESHOLD = 0.85
DEFAULT_FAISS_THRESHOLD = 0.75

st.title("📄 RAG PDF Bot")
st.caption("Ask questions grounded in your embedded PDFs. Cache-first, FAISS-backed, Gemini answers.")

with st.sidebar:
    st.header("Settings")
    cache_threshold = st.slider("Cache similarity threshold (cosine)", 0.0, 1.0, 0.85, 0.05)
    faiss_threshold = st.slider("FAISS similarity threshold (cosine)", 0.0, 1.0, 0.75, 0.05)
    top_k = st.slider("FAISS top-k to retrieve", 1, 20, 5, 1)
    fallback = st.checkbox("Fallback to top-k when no hits", True)
    st.divider()
    if st.button("Initialize System"):
        cache_ok = initialize_cache() is not None
        faiss_ok = initialize_faiss_search()
        if cache_ok and faiss_ok:
            # Persist chosen thresholds in session on initialization
            st.session_state["cache_threshold"] = cache_threshold
            st.session_state["faiss_threshold"] = faiss_threshold
            st.session_state["system_initialized"] = True
        st.success("Initialized" if (cache_ok and faiss_ok) else "Initialized with warnings")

    # Cache maintenance
    if st.button("Clear Cache"):
        try:
            deleted = clear_cache()
            st.success(f"Cleared {deleted} cache entrie(s)")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

query = st.text_input("Enter your question")
go = st.button("Ask")

if go and query:
    with st.spinner("Processing..."):
        # Resolve thresholds: use values saved during initialization, else defaults
        cache_thr = st.session_state.get("cache_threshold", DEFAULT_CACHE_THRESHOLD)
        faiss_thr = st.session_state.get("faiss_threshold", DEFAULT_FAISS_THRESHOLD)

        pre = preprocess_query(query)
        norm = normalize_query(pre)
        qv = embed_query(query)

        cache = process_query_with_cache(norm, qv, threshold=cache_thr)
        if cache.get("cache_hit"):
            st.success(f"Cache hit (similarity={cache['similarity']:.2f})")
            st.subheader("Answer (cached)")
            st.write(cache["answer"]) 
        else:
            results = search_with_threshold(qv, k=top_k, threshold=faiss_thr)
            used_docs = results
            if not results and fallback:
                used_docs = search_similar_documents(qv, k=top_k)

            if used_docs:
                st.subheader("Retrieved Chunks")
                for i, r in enumerate(used_docs[:3]):
                    with st.expander(f"{i+1}. similarity={r['similarity_score']:.3f}"):
                        st.write(r["document"]) 

                context_documents = [
                    {
                        "rank": i + 1,
                        "content": r["document"],
                        "similarity": r["similarity_score"],
                        "metadata": r.get("metadata", {}),
                    }
                    for i, r in enumerate(used_docs[:3])
                ]

                answer = generate_answer(norm, context_documents)
                if not answer.startswith("Error:"):
                    st.subheader("Answer")
                    st.write(answer)
                    save_llm_response_to_cache(norm, qv, answer)
                    st.caption("Saved to cache")
                else:
                    st.error(answer)
            else:
                st.warning("No similar documents found in FAISS index")


