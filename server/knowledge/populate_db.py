# populate_db_v3.py
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Configuration constants
JSON_FILE_PATH = 'processed_documents_v3.json'
DB_PERSIST_PATH = './chroma_db_data'
COLLECTION_NAME = "reddit_datascience_openai"
BATCH_SIZE = 100

def load_processed_documents(file_path: str):
    """Load preprocessed documents from JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    
    print(f"Loading documents from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents):,} documents")
    return documents

def prepare_for_chromadb(documents):
    """Convert document format to ChromaDB format (ids, documents, metadatas)."""
    ids = []
    docs = []
    metadatas = []
    
    for doc in documents:
        ids.append(doc['id'])
        docs.append(doc['document'])
        metadatas.append(doc['metadata'])
    
    return ids, docs, metadatas

def main():
    # 1. Setup OpenAI API
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    # 2. Initialize Persistent Client
    print(f"Connecting to ChromaDB at {DB_PERSIST_PATH}...")
    client = chromadb.PersistentClient(path=DB_PERSIST_PATH)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION_NAME}' loaded. Current count: {collection.count()}")

    # 3. Load preprocessed data
    documents = load_processed_documents(JSON_FILE_PATH)
    
    if not documents:
        print("No documents to insert.")
        return
    
    # 4. Convert to ChromaDB format
    ids, docs, metadatas = prepare_for_chromadb(documents)
    
    # 5. Statistics
    post_count = sum(1 for m in metadatas if m['type'] == 'post')
    comment_count = sum(1 for m in metadatas if m['type'] == 'comment')
    
    print(f"\nDataset Statistics:")
    print(f"  Total documents: {len(documents):,}")
    print(f"  Posts:           {post_count:,}")
    print(f"  Comments:        {comment_count:,}")

    # 6. Insert (Upsert) Data
    total = len(ids)
    print(f"\nUpserting {total:,} documents...")
    print("This may take several minutes due to OpenAI API embedding generation.\n")
    
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )
        
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        progress = (i + len(batch_ids)) / total * 100
        
        print(f"  Batch {batch_num}/{total_batches} processed ({progress:.1f}% complete)")

    final_count = collection.count()
    print(f"\nOperation Complete!")
    print(f"Final Collection Count: {final_count:,}")
    
    # Verify counts match
    if final_count == total:
        print("SUCCESS: All documents inserted correctly.")
    else:
        print(f"WARNING: Expected {total:,} but collection has {final_count:,}")

if __name__ == "__main__":
    main()