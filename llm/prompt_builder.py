# llm/prompt_builder.py
def build_rag_prompt(query: str, context_documents: list) -> str:
    """
    Build a simple RAG prompt for Gemini.
    
    Args:
        query: User's question
        context_documents: List of relevant documents
        
    Returns:
        Formatted prompt string
    """
    
    # Extract content from documents
    context_text = "\n\n".join([doc['content'] for doc in context_documents])
    
    # Build the prompt
    prompt = f"""You are a helpful AI assistant. Your task is to answer questions using the provided information.

Context Information:
{context_text}

Question: {query}

Instructions:
- Use only the information provided in the context above
- If the answer is not in the context, say "I don't have enough information to answer this question"
- Be concise and accurate
- Provide a clear, helpful answer

Answer:"""
    
    return prompt
