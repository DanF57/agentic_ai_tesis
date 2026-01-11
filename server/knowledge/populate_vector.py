import json
import re
import uuid
import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv  # Needs: pip install python-dotenv

# --- CONFIGURATION ---
# Load environment variables from a .env file
load_dotenv()

# Configuration constants
JSON_FILE_PATH = 'reddit_dataset_final_1492_posts.json' # Relative path
DB_PERSIST_PATH = './chroma_db_data'  # Folder where DB will be stored locally
COLLECTION_NAME = "reddit_datascience_openai"
MIN_SCORE_COMMENT = 1
MIN_WORDS_COMMENT = 5
BATCH_SIZE = 100

url_pattern = re.compile(r'(https?://\S+)')

def clean_and_extract_urls(text: str) -> Tuple[str, List[str]]:
    """Removes URLs from text and returns cleaned text with extracted URLs."""
    if not text:
        return "", []
    urls = url_pattern.findall(text)
    clean_text = url_pattern.sub('', text).replace('  ', ' ').strip()
    return clean_text, urls

def preprocess_dataset(file_path: str) -> Tuple[List[str], List[str], List[Dict]]:
    """Preprocesses Reddit dataset and prepares data for ChromaDB insertion."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return [], [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ids = []
    documents = []
    metadatas = []
    
    stats = {'posts': 0, 'comments': 0, 'skipped_empty': 0, 'skipped_deleted': 0}
    print(f"Starting preprocessing of {len(data)} posts...")

    for post in data:
        # Use existing post_id or generate one (Note: generated UUIDs change every run!)
        post_id = post.get('post_id', str(uuid.uuid4()))
        p_title = post.get('title', '').strip()
        p_text = post.get('text', '')

        # Skip deleted/removed
        if p_text in ['[deleted]', '[removed]', 'comment deleted', 'thank you']:
            stats['skipped_deleted'] += 1
            continue

        p_clean_text, p_urls = clean_and_extract_urls(p_text)

        if not p_title and not p_clean_text:
            stats['skipped_empty'] += 1
            continue

        # Construct post document
        post_doc_content = f"Title: {p_title}\nContent: {p_clean_text}".strip()
        
        ids.append(f"post_{post_id}")
        documents.append(post_doc_content)
        metadatas.append({
            "type": "post",
            "subreddit": post.get('subreddit', 'unknown'),
            "score": post.get('score', 0),
            "author": post.get('author', 'unknown'),
            "url_source": post.get('post_url', ''),
            "extracted_urls": ", ".join(p_urls),
            "post_title": p_title
        })
        stats['posts'] += 1

        # Process comments
        if 'comments' in post and isinstance(post['comments'], list):
            for comment in post['comments']:
                c_text = comment.get('text', '')
                c_score = comment.get('score', 0)
                c_id = comment.get('comment_id', str(uuid.uuid4()))

                if c_text in ['[deleted]', '[removed]']: continue
                if c_score < MIN_SCORE_COMMENT: continue
                if len(c_text.split()) < MIN_WORDS_COMMENT: continue

                c_clean_text, c_urls = clean_and_extract_urls(c_text)
                if not c_clean_text:
                    stats['skipped_empty'] += 1
                    continue

                # Context Injection
                full_comment_doc = f"Context (Post Title): {p_title}\nComment: {c_clean_text}"

                ids.append(f"comment_{c_id}")
                documents.append(full_comment_doc)
                metadatas.append({
                    "type": "comment",
                    "parent_id": post_id,
                    "subreddit": post.get('subreddit', 'unknown'),
                    "score": c_score,
                    "author": comment.get('author', 'unknown'),
                    "post_title": p_title,
                    "extracted_urls": ", ".join(c_urls)
                })
                stats['comments'] += 1

    print("\nPreprocessing Complete:")
    print(f"  Posts: {stats['posts']}")
    print(f"  Comments: {stats['comments']} (score >= {MIN_SCORE_COMMENT})")
    print(f"  Total to insert: {len(documents)}")
    
    return ids, documents, metadatas

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
    # This will create/load the DB from the './chroma_db_data' folder
    print(f"Connecting to ChromaDB at {DB_PERSIST_PATH}...")
    client = chromadb.PersistentClient(path=DB_PERSIST_PATH)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION_NAME}' loaded. Current count: {collection.count()}")

    # 3. Process Data
    ids, documents, metadatas = preprocess_dataset(JSON_FILE_PATH)

    if not ids:
        print("No documents to insert.")
        return

    # 4. Insert (Upsert) Data
    # Upsert handles duplicates: updates existing IDs, inserts new ones.
    total = len(ids)
    print(f"\nUpserting {total} documents (this handles updates & new entries)...")
    
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )
        print(f"  Batch {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} processed.")

    print(f"Operation Complete. Final Collection Count: {collection.count()}")

if __name__ == "__main__":
    main()