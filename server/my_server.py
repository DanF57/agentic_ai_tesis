# server/my_server.py
from fastmcp import FastMCP
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.utils import Secret
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import uuid

# Haystack Imports
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

load_dotenv()

# --- Configuración Inicial ---
SERVER_DIR = Path(__file__).parent
DB_PATH = str(SERVER_DIR / "knowledge" / "chroma_db_data")

mcp = FastMCP("StudentKnowledgeTools")

# Inicialización de Componentes
document_store = ChromaDocumentStore(
    persist_path=DB_PATH, 
    collection_name="reddit_datascience_openai"
)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
query_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

try:
    web_search_component = SerperDevWebSearch(
        api_key=Secret.from_env_var("SERPER_API_KEY"),
        top_k=5
    )
except Exception:
    web_search_component = None


@mcp.tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the internal Knowledge Base [RAG].
    Returns structured JSON with results array.
    
    Args:
        query: The search query
    
    Returns:
        JSON string with structure:
        {
            "tool": "RAG",
            "query": str,
            "results": [
                {
                    "result": str,
                    "source_type": str,
                    "title": str,
                    "score": str,
                    "urls": list,
                    "content": str
                }
            ]
        }
    """

    print(f"--- [RAG] Query: {query} ")

    try:
        result = query_pipeline.run({
            "text_embedder": {"text": query},
            "retriever": {"top_k": 5}
        })
        
        documents = result["retriever"]["documents"]
        
        if not documents:
            return json.dumps({
                "tool": "RAG",
                "query": query,
                "results": []
            }, ensure_ascii=False)
        
        # Estructura los resultados
        structured_results = []
        for idx, doc in enumerate(documents):
            meta = doc.meta
            print("Meta: ", meta)
            source_type = meta.get("type", "forums").capitalize()
            score = float(doc.score) if doc.score else 0.0
            
            structured_results.append({
                "result": str(idx + 1),
                "source_type": source_type,
                "title": meta.get("post_title"),
                "score": f"{score:.4f}",
                "urls": meta.get("url_source", []) if meta.get("url_source" ) else meta.get("extracted_urls", []),
                "content": doc.content
            })
        
        response_tool = json.dumps({
            "tool": "RAG",
            "query": query,
            "results": structured_results
        }, ensure_ascii=False)

        print(response_tool)
        
        return response_tool

    except Exception as e:
        return json.dumps({
            "tool": "RAG",
            "query": query,
            "error": str(e),
            "results": []
        }, ensure_ascii=False)


@mcp.tool
def search_web(query: str, top_k: int = 5) -> str:
    """
    Searches the real-time web using Google.
    Returns structured JSON with results array.
    
    Args:
        query: The search query
        top_k: Number of results to return
    
    Returns:
        JSON string with structure:
        {
            "tool": "WEB",
            "query": str,
            "results": [
                {
                    "result": str,
                    "source_type": "Web",
                    "score": "N/A",
                    "urls": list,
                    "content": str
                }
            ]
        }
    """
        
    print(f"--- [WEB] Query: {query}")
    
    if not web_search_component:
        return json.dumps({
            "tool": "WEB",
            "query": query,
            "error": "Web search tool not configured.",
            "results": []
        }, ensure_ascii=False)

    try:
        results = web_search_component.run(query=query)
        documents = results["documents"][:top_k]
        
        if not documents:
            return json.dumps({
                "tool": "WEB",
                "query": query,
                "results": []
            }, ensure_ascii=False)
        
        # Estructura los resultados
        structured_results = []
        for idx, doc in enumerate(documents):
            title = doc.meta.get('title', 'Sin título')
            link = doc.meta.get('link', '')
            snippet = doc.content
            
            structured_results.append({
                "result": str(idx + 1),
                "source_type": "Web",
                "score": "N/A",  # Web search no tiene score de similaridad
                "urls": [link] if link else [],
                "content": f"Title: {title}\n{snippet}"
            })
        
        return json.dumps({
            "tool": "WEB",
            "query": query,
            "results": structured_results
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "tool": "WEB",
            "query": query,
            "error": str(e),
            "results": []
        }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)