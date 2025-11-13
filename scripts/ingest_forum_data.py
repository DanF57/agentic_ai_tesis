# scripts/ingest_forum_data.py
"""
Herramienta MCP: agrega nuevo conocimiento a la base vectorial.
Se activará cuando el agente detecte que la pregunta del alumno es útil, nueva
y debe registrarse en la base de conocimiento.
"""

from fastmcp import tool
from ..vector_store.faiss_store import get_vector_store

@tool()
def add_to_knowledge(question: str, answer: str) -> str:
    store = get_vector_store()
    store.add_entry(question, answer)
    return "Nuevo conocimiento agregado."
