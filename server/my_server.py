# server/my_server.py
from fastmcp import FastMCP
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.utils import Secret
from dotenv import load_dotenv
from pathlib import Path
import json
import sys
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import ExecutionLogger
import time
import requests
from bs4 import BeautifulSoup
from readability import Document
import time
# Haystack Imports
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

load_dotenv()

# --- INICIO DEL PARCHE WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Configuración Inicial ---
SERVER_DIR = Path(__file__).parent
DB_PATH = str(SERVER_DIR / "knowledge" / "chroma_db_data")

mcp = FastMCP("StudentKnowledgeTools")

# Inicialización de Componentes
document_store = ChromaDocumentStore(
    persist_path=DB_PATH, 
    collection_name="reddit_datascience_openai",
    distance_function="cosine"
)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
query_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")


# --- HELPER PARA FORMATEAR TEXTO AL LLM ---
def format_results_for_llm(results_list):
    """
    Convierte la lista de resultados JSON a un string legible para el LLM.
    Incluye Score y URL para que el agente pueda citar y evaluar relevancia.
    """
    if not results_list:
        return "No results found."
    
    formatted_text = ""
    for item in results_list:
        # Extraemos datos básicos
        idx = item.get('result', '?')
        title = item.get('title', 'No Title')
        content = item.get('content', '')

        source_url = item.get('source_url', [])
        url_str = ", ".join(source_url) if source_url else "N/A"

        # Formato bloque explícito para el LLM
        formatted_text += f"----- Result {idx} -----\n"
        formatted_text += f"Title: {title}\n"
        formatted_text += f"Content: {content}\n"
        formatted_text += f"Source URLS: {url_str}\n\n"
    
    return formatted_text.strip()
# --------------------------------------------------------


@mcp.tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the internal Knowledge Base [RAG].
    """
    tool_start_time = time.perf_counter()

    print(f"--- [RAG] Query: {query} ")

    response_dict = {
        "tool": "RAG",
        "query": query,
        "results": []
    }

    try:
        result = query_pipeline.run({
            "text_embedder": {"text": query},
            "retriever": {"top_k": 5}
        })

        documents = result["retriever"]["documents"]
        structured_results = []
        
        if documents:
            valid_idx = 0
            for doc in documents:
                # Menor score = documento más similar.
                score = float(doc.score)
                print(f"Score: {score}")
                print(f"ID: {doc.id}")
                print(f"Source_url: {doc.meta.get('permalink', '')}")
                meta = doc.meta or {}
                
                structured_results.append({
                    "result": str(valid_idx),
                    "id": doc.id,
                    "source_type": meta.get("type", "forum").capitalize(),
                    "title": meta.get("post_title", "Untitled"),
                    "score": f"{score:.4f}",
                    "source_url": meta.get("permalink", ""),
                    "content": doc.content
                })

        if not structured_results:
            response_dict["no_valid_documents"] = True

        response_dict["results"] = structured_results

    except Exception as e:
        response_dict["error"] = str(e)

    tool_end_time = time.perf_counter()
    response_dict["execution_time_seconds"] = tool_end_time - tool_start_time

    # 1. LOGGING: Guardamos el JSON completo y estructurado (NO TOCAR)
    ExecutionLogger.record_tool_execution("RAG", query, response_dict)
    # Print de resultados enviados (primeros 100 caracteres) para debug
    formatted_output = format_results_for_llm(response_dict["results"])
    print(f"[RAG] Preview resultados enviados al LLM (100 chars) \n: {formatted_output[:100]}")
    # 2. RETURN: Devolvemos solo texto limpio al LLM para que no se confunda
    return formatted_output


@mcp.tool
def search_web(query: str, top_k: int = 5) -> str:
    """
    Searches the real-time web using Google.
    """    
    tool_start_time = time.perf_counter()
    print(f"--- [WEB] Query: {query}")

    response_data = {
        "tool": "WEB",
        "query": query,
        "results": []
    }

    try:
        web_search_component = SerperDevWebSearch(
            api_key=Secret.from_env_var("SERPER_API_KEY"),
            top_k=5
        )
    except Exception:
        web_search_component = None

    if not web_search_component:
        response_data["error"] = "Web search tool not configured."
        ExecutionLogger.record_tool_execution("WEB", query, response_data)
        return "Error: Web search tool not configured."

    try:
        results = web_search_component.run(query=query)
        documents = results["documents"][:top_k]
        
        if documents:
            structured_results = []
            for idx, doc in enumerate(documents):
                title = doc.meta.get('title', 'Sin título')
                link = doc.meta.get('link', '')
                snippet = doc.content
                
                structured_results.append({
                    "result": str(idx + 1),
                    "source_type": "Web",
                    "score": "N/A",
                    "source_url": [link] if link else [],
                    "content": f"Title: {title}\nContent:{snippet}"
                })
            response_data["results"] = structured_results

    except Exception as e:
        response_data["error"] = str(e)

    tool_end_time = time.perf_counter()
    response_data["execution_time_seconds"] = tool_end_time - tool_start_time

    # 1. LOGGING (JSON Completo)
    ExecutionLogger.record_tool_execution("WEB", query, response_data)
    
    # 2. RETURN (Texto Limpio)
    return format_results_for_llm(response_data["results"])


@mcp.tool
def collect_posts(subreddit: str, top_n: int = 30) -> str:
    """
    Collects posts for a given subreddit, ordered by upvotes.
    Only returns posts whose title contains '?'.

    Returns a formatted string for the agent to analyze.
    """
    tool_start_time = time.perf_counter()
    
    print(f"\n--- Iniciando collect_posts ---")
    print(f"Subreddit: {subreddit}")
    print(f"Top N solicitado: {top_n}")

    response_log = {
        "tool": "Post topic collector",
        "subreddit": subreddit,
        "top_n": top_n,
        "total_candidates": 0,
        "filtered_questions": 0,
        "returned_posts": 0,
        "results": []
    }

    try:
        # 1. Recuperación por metadatos: type=post y subreddit específico
        print("\n[1] Filtrando documentos por type=post y subreddit...")
        documents = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "type", "operator": "==", "value": "post"},
                    {"field": "subreddit", "operator": "==", "value": subreddit}
                ]
            }
        )
        
        print(f"Total documentos encontrados: {len(documents)}")
        response_log["total_candidates"] = len(documents)

        if not documents:
            print(f"No se encontraron posts para el subreddit '{subreddit}'")
            return f"No posts found for subreddit '{subreddit}'."

        # 2. Filtrado: solo posts con '?' en el título
        print("\n[2] Filtrando posts que contengan '?' en el titulo...")
        question_posts = []
        for doc in documents:
            title = doc.meta.get("post_title", "") or ""
            if "?" in title:
                question_posts.append(doc)
                print(f"  - ID: {doc.id} | Titulo: {title[:80]}...")

        print(f"\nTotal posts con '?': {len(question_posts)}")
        response_log["filtered_questions"] = len(question_posts)

        if not question_posts:
            print(f"No se encontraron preguntas para el subreddit '{subreddit}'")
            return f"No question posts found for subreddit '{subreddit}'."

        # 3. Ordenar por votos ascendentemente
        print("\n[3] Ordenando por votos (ascendente)...")
        def get_votes(doc):
            return doc.meta.get("votes", 0) or 0

        question_posts.sort(key=get_votes, reverse=False)
        
        for i, doc in enumerate(question_posts[:5], 1):
            print(f"  Top {i}: {get_votes(doc)} votos - {doc.meta.get('post_title', '')[:60]}...")

        # 4. Seleccionar top N
        selected_posts = question_posts[:top_n]
        response_log["returned_posts"] = len(selected_posts)
        print(f"\n[4] Seleccionados {len(selected_posts)} posts para retornar")

        # 5. Formateo para el agente
        formatted_output = (
            f"SUBREDDIT: {subreddit}\n"
            f"QUESTIONS (ordered by lowest votes first)\n\n"
        )

        for idx, doc in enumerate(selected_posts, start=1):
            formatted_output += (
                f"[{idx}] {doc.id} | {doc.meta.get('post_title', 'No title')}\n"
            )

            response_log["results"].append({
                "rank": idx,
                "id": doc.id,
                "title":  doc.meta.get('post_title', 'Sin titulo'),
                "votes": doc.meta.get('votes', 0)
            })

        print("\n[5] Formato de salida generado")
        print(f"Longitud del string: {len(formatted_output)} caracteres")

    except Exception as e:
        print(f"\nERROR: {e}")
        response_log["error"] = str(e)
        return f"Error executing collect_posts tool: {e}"

    tool_end_time = time.perf_counter()
    response_log["execution_time_seconds"] = tool_end_time - tool_start_time
    
    print(f"\nTiempo de ejecucion: {response_log['execution_time_seconds']:.4f} segundos")

    ExecutionLogger.record_tool_execution(
        "collect_posts",
        f"subreddit={subreddit}, top_n={top_n}",
        response_log
    )

    print("--- collect_posts finalizado ---\n")
    return formatted_output.strip()

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)