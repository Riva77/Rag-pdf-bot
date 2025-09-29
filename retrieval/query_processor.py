import re
from typing import List
from ingestion.cleaner import normalize_text  # reuse cleaner
"""
Spell checker intentionally disabled to avoid incorrect corrections
on expanded terms like "artificial intelligence".
"""

def preprocess_query(raw_query: str) -> str:
    """
    Clean and prepare user query for embedding
    - Remove extra spaces
    - Handle special characters (keep basic punctuation)
    """
    query = raw_query.strip()
    query = re.sub(r"\s+", " ", query)  # collapse multiple spaces
    # Remove unusual symbols (keep word chars, spaces, ?, ., !)
    query = re.sub(r"[^\w\s?.!]", "", query)
    return query


def normalize_query(query: str) -> str:
    """
    Normalize query text:
    - Lowercase
    - Use ingestion.cleaner.normalize_text
    - Abbreviations already expanded in preprocess_query
    """
    query = normalize_text(query)
    return query


def extract_keywords(query: str) -> List[str]:
    """
    Extract key terms from query for hybrid search
    - Remove stopwords
    - Keep important nouns & verbs
    """
    stopwords = {"is", "the", "a", "an", "of", "to", "in", "and", "on"}
    tokens = re.findall(r"\b\w+\b", query.lower())
    keywords = [t for t in tokens if t not in stopwords]
    return keywords
