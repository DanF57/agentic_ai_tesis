# tests/calibrate_threshold.py
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
DB_PATH = str(PROJECT_ROOT / "server" / "knowledge" / "chroma_db_data")
COLLECTION_NAME = "reddit_datascience_openai"

def calibrate_threshold():
    """
    Ejecuta múltiples queries de prueba para encontrar el threshold óptimo.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    
    # Queries de prueba variadas
    test_queries = [
        #test con resultado perfecto
        "Context (Post Title): Lack of Hold-Out Set Leads to State Wasting $365k\nComment: holdout set is the same as a test set, as in training set, validation set and testing set, right?" #test con resultado perfecto

        # Muy específicas (se esperan scores bajos)
        "overfitting in multiple linear regression",
        "what is eda?",    
        
        # Moderadamente específicas
        "how to evaluate machine learning models",
        "data cleaning best practices",
        
        # Generales 
        "machine learning tutorials",
        "statistics basics",
        
        # Fuera de tema
        "how to cook pasta",
        "causes of cancer"
    ]
    
    all_scores = []
    
    print("=" * 60)
    print("CALIBRATION REPORT: Score Distribution Analysis")
    print("=" * 60)
    
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["distances"]
        )
        
        distances = results['distances'][0]
        all_scores.extend(distances)
        
        print(f"\nQuery: '{query}'")
        print(f"  Min: {min(distances):.4f} | Max: {max(distances):.4f} | Avg: {sum(distances)/len(distances):.4f}")
    
    # Estadísticas globales
    all_scores.sort()
    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)
    print(f"Absolute Min Score: {min(all_scores):.4f}")
    print(f"Absolute Max Score: {max(all_scores):.4f}")
    print(f"Overall Average:    {sum(all_scores)/len(all_scores):.4f}")
    print(f"Median:             {all_scores[len(all_scores)//2]:.4f}")
    
    # Percentiles
    p25 = all_scores[len(all_scores)//4]
    p50 = all_scores[len(all_scores)//2]
    p75 = all_scores[3*len(all_scores)//4]
    p90 = all_scores[9*len(all_scores)//10]
    
    print(f"\nPercentiles:")
    print(f"  25th: {p25:.4f} (muy relevante)")
    print(f"  50th: {p50:.4f} (relevante)")
    print(f"  75th: {p75:.4f} (moderadamente relevante)")
    print(f"  90th: {p90:.4f} (poco relevante)")
    
    # Recomendaciones
    print("\n" + "=" * 60)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 60)
    print(f"Strict (high precision):    {p50:.4f}")
    print(f"Balanced (recommended):     {p75:.4f}")
    print(f"Permissive (high recall):   {p90:.4f}")
    
if __name__ == "__main__":
    calibrate_threshold()