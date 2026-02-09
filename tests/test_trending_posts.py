from pathlib import Path
import sys

# --- Ajuste de paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports Haystack / Chroma ---
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# --- Configuración ---
SERVER_DIR = PROJECT_ROOT / "server"
DB_PATH = str(SERVER_DIR / "knowledge" / "chroma_db_data")

document_store = ChromaDocumentStore(
    persist_path=DB_PATH,
    collection_name="reddit_datascience_openai",
    distance_function="cosine"
)

# --------------------------------------------------
# FUNCIÓN EPV (MISMA LÓGICA QUE LA TOOL)
# --------------------------------------------------
def epv_collect_posts_debug(year: int, top_n: int = 30):
    print(f"\n[EPV DEBUG] Año: {year}, Top N: {top_n}\n")

    documents = document_store.filter_documents(
        filters={
            "type": {"$eq": "post"},
            "year": {"$eq": year}
        }
    )

    print(f"Total documentos recuperados: {len(documents)}")

    if not documents:
        return

    # Filtrado de preguntas
    question_posts = []
    for doc in documents:
        title = doc.meta.get("post_title", "") or ""
        content = doc.content or ""
        if "?" in title or "?" in content:
            question_posts.append(doc)

    print(f"Posts con forma de pregunta: {len(question_posts)}")

    # Orden por votos
    question_posts.sort(
        key=lambda d: d.meta.get("votes", 0) or 0,
        reverse=True
    )

    # Top N
    selected = question_posts[:top_n]

    print("\n--- TOP POSTS ---\n")
    for idx, doc in enumerate(selected, start=1):
        print(f"{idx}. ID: {doc.id}")
        print(f"   Título: {doc.meta.get('post_title', 'Sin título')}")
        print(f"   Votes: {doc.meta.get('votes', 0)}")
        print(f"   Year: {doc.meta.get('year')}")
        print("-" * 50)


# --------------------------------------------------
# EJECUCIÓN DIRECTA
# --------------------------------------------------
if __name__ == "__main__":
    epv_collect_posts_debug(year=2025, top_n=30)
