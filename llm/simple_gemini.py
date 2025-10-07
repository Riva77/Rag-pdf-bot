# llm/simple_gemini.py
import os
from config import GOOGLE_AI_API_KEY
import google.generativeai as genai
from typing import List, Dict
from .prompt_builder import build_rag_prompt

def generate_answer(query: str, context_documents: List[Dict]) -> str:
    """
    Simple function to generate answer using Gemini.
    
    Args:
        query: User's question
        context_documents: List of relevant documents
        
    Returns:
        Answer string
    """
    
    # Get API key (prefer config, fallback to env)
    api_key = GOOGLE_AI_API_KEY or os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        return "Error: GOOGLE_AI_API_KEY not set"
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key) 
        model = genai.GenerativeModel('gemini-2.0-flash-lite') 
        
        # Build prompt using prompt builder
        prompt = build_rag_prompt(query, context_documents)
        
        # Generate answer
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            return "No answer generated"
        
    except Exception as e:
        return f"Error: {str(e)}"
