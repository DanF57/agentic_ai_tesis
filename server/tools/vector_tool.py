# server/tools/vector_tool.py
"""
Herramienta MCP: Búsqueda vectorial de conocimiento.
Se conecta al índice FAISS o Qdrant para obtener los documentos más relevantes.
"""

from fastmcp import tool
from ..vector_store.faiss_store import get_vector_store

@tool()
def vector_search(query: str, k: int = 3) -> dict:
    store = get_vector_store()
    results = store.search(query, top_k=k)
    return {"results": results}

