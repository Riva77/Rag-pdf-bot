import re
import unicodedata

def clean_text(text: str) -> str:
    """Apply general text cleaning pipeline."""
    text = normalize_text(text)
    text = remove_headers_footers(text)
    text = fix_encoding_issues(text)
    return text.strip()

def normalize_text(text: str) -> str:
    """Normalize whitespace and unicode characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text

def remove_headers_footers(text: str) -> str:
    """Remove common headers/footers (simple heuristic)."""
    # Example heuristic: remove repeated lines (headers/footers across pages)
    lines = text.split("\n")
    unique_lines = [line for line in lines if not re.match(r"^Page \d+$", line)]
    return " ".join(unique_lines)

def fix_encoding_issues(text: str) -> str:
    """Fix common encoding issues."""
    return text.encode("utf-8", "ignore").decode("utf-8")
