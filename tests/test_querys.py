import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pathlib import Path

# 1. Load Environment Variables
load_dotenv()

# Configuration (Must match populate_db.py)
# Get the path relative to this test file's location
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
DB_PATH = str(PROJECT_ROOT / "server" / "knowledge" / "chroma_db_data")
COLLECTION_NAME = "reddit_datascience_openai"

def run_test():
    # Check API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file.")
        return

    print(f"--- DIAGNOSTICS: Connecting to DB at {DB_PATH} ---")

    # 2. Setup Embedding Function (CRITICAL: Must match ingestion model)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    # 3. Connect to Database
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Tip: Check if the folder './chroma_db_data' exists.")
        return

    # 4. Basic Stats
    count = collection.count()
    print(f"✅ Connection Successful!")
    print(f"📊 Total Documents in Collection: {count}")
    
    if count == 0:
        print("⚠️ WARNING: Collection is empty. Run populate_db.py first.")
        return

    # 5. Perform Test Query
    test_query = " ? " # <--- Change this to test specific topics
    print(f"\n🔍 Running Test Query: '{test_query}'")
    
    results = collection.query(
        query_texts=[test_query],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # 6. Display Results
    print("\n--- RETRIEVAL RESULTS ---")
    ids = results['ids'][0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    for i in range(len(ids)):
        score = distances[i]
        doc_type = metas[i].get('type', 'unknown').upper()
        subreddit = metas[i].get('subreddit', 'unknown')
        post_title = metas[i].get('post_title', 'No Title')
        
        # Interpretation of Score (L2 Distance)
        relevance = "⭐⭐⭐" if score < 1.0 else ("⭐⭐" if score < 1.25 else "⭐")

        print(f"\nResult #{i+1} | Score: {score:.4f} {relevance}")
        print(f"[{doc_type}] from r/{subreddit}")
        print(f"Context: {post_title}")
        print(f"Snippet: {docs[i][:200]}...") # Print first 150 chars
        print("-" * 50)
        print("Estructura de metadata:\n")
        print(metas[i])
    print("-" * 50)
    # print("Estructura de resultados:\n")
    # print(results)
    # print("-" * 50)
    print("Collection Metadata:\n")
    print(collection.metadata)
    print("-" * 50)

if __name__ == "__main__":
    run_test()