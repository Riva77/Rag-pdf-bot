# RAG PDF Bot Setup Instructions

## Quick Setup

### 1. Create/Activate Virtual Environment

Windows PowerShell:

```bash
python -m venv venv
./venv/Scripts/Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 4. Configure via config.py (.env optional)

Create a `.env` file in the project root (optional but recommended):

```bash
# .env
GOOGLE_AI_API_KEY=your_actual_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
FAISS_INDEX_PATH=embeddings/faiss_index
```

Notes:

- `llm/simple_gemini.py` uses `config.GOOGLE_AI_API_KEY` (falls back to env var).
- `retrieval/cache_manager.initialize_cache()` defaults to `config.REDIS_*` values.
- `embeddings/loader.py` uses `config.FAISS_INDEX_PATH` for index and pickle files.

### 5. Run the Bot

```bash
python main.py
```

### 6. Run the Streamlit UI (optional)

```bash
streamlit run ui/app.py
```

## 🔧 What You Need

### Required:

- **Python 3.11+**
- **Redis** (for caching). Ensure the server is running.
- **Gemini API Key** (for LLM)

### Optional:

- **Environment file** (`.env`) for API keys

## Project Structure

```
Rag-pdf-bot/
├── app/
│   └── pipeline.py          # Main RAG pipeline
├── llm/
│   ├── prompt_builder.py    # Simple prompt builder
│   └── simple_gemini.py     # Gemini integration
├── retrieval/
│   ├── cache_manager.py     # Redis cache
│   └── faiss_search.py      # FAISS search
├── embeddings/
│   ├── embedder.py          # Text embedding
│   └── loader.py            # FAISS index/documents/metadata loader (uses config path)
├── ingestion/
│   └── coffee.pdf           # Sample PDF
├── config.py                # Configuration
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

## How It Works

1. **Query Processing**: User asks question
2. **Cache Check**: Look for similar cached answers
3. **FAISS Search**: If cache miss, search document chunks
4. **LLM Generation**: Send top 3 docs to Gemini
5. **Cache Save**: Save answer for future queries

## Troubleshooting

### Redis Connection Error

```bash
# Start Redis (Windows)
redis-server | cat

# Install Redis
# Windows (option A): Use WSL and `sudo apt install redis-server`
# Windows (option B): https://github.com/microsoftarchive/redis/releases
```

### API Key Error

- Check `.env` file exists
- Verify API key is correct
- Make sure no extra spaces in key

### Import Error

```bash
pip install -r requirements.txt
```

### Select the correct Python interpreter (VS Code)

Ctrl+Shift+P → "Python: Select Interpreter" → choose the one from `venv`.

### Git ignore tips

If you committed files before adding `.gitignore`, Git already tracks them and will keep tracking until you untrack them. Do this once:

```bash
git rm -r --cached venv .venv .venv311 embeddings/faiss_index __pycache__ *.pyc *.pkl *.faiss | cat
git add .gitignore
git add .
git commit -m "Add .gitignore and remove tracked venv/artifacts"
```

If you haven't committed those files yet, adding `.gitignore` and running `git add .gitignore && git add .` is enough.
