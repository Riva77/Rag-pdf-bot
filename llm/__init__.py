# llm/__init__.py
"""
Simple LLM module for RAG-based question answering.
"""

from .simple_gemini import generate_answer

__all__ = ['generate_answer']
