#evaluate_logs.py
import json
import os
import glob
import torch
import numpy as np
from pathlib import Path
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- CONFIGURACIÓN ---
LOGS_DIR = Path("logs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelos para Perplejidad (Español)
PPL_MODEL_ID = "DeepESP/gpt2-spanish" 

print(f"⚙️ Iniciando evaluador en: {DEVICE}")
print("⏳ Cargando modelos (esto puede tardar la primera vez)...")

# 1. Cargar modelo para Perplejidad
try:
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_ID)
    ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_ID).to(DEVICE)
    print("✅ Modelo GPT-2 Español cargado.")
except Exception as e:
    print(f"❌ Error cargando GPT-2: {e}")
    exit()

def calculate_perplexity(text):
    """
    Calcula la perplejidad de un texto usando GPT-2.
    Menor es mejor (más fluido).
    """
    if not text or len(text.strip()) == 0:
        return None
        
    encodings = ppl_tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    
    with torch.no_grad():
        outputs = ppl_model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

def calculate_bertscore(candidate, references):
    """
    Calcula BERTScore comparando la respuesta (candidato) 
    con el contexto recuperado (referencias).
    Usa el modelo por defecto 'roberta-large' o multilingüe internamente.
    """
    if not candidate or not references:
        return None
        
    # bert_score maneja la carga de su propio modelo internamente.
    # lang="es" forzará el uso de un modelo multilingüe adecuado.
    try:
        P, R, F1 = score([candidate], [references], lang="es", verbose=False, device=DEVICE)
        return F1.mean().item() # Retornamos la media del F1 Score
    except Exception as e:
        print(f"Error en BERTScore: {e}")
        return None

def get_context_references(log_data):
    """
    Extrae el texto de los resultados de las herramientas (RAG o WEB)
    para usarlos como 'Verdad Terreno' (Ground Truth) de referencia.
    """
    references = []
    tool_calls = log_data.get("tool_calls", [])
    
    for call in tool_calls:
        # Priorizamos el contenido de los resultados
        results = call.get("results", [])
        if isinstance(results, list):
            for res in results:
                content = res.get("content", "")
                if content:
                    references.append(content)
    
    # Si hay múltiples fragmentos, los unimos como una sola referencia larga
    # o devolvemos el más relevante. Para BERTScore, unirlos suele funcionar 
    # para ver si la semántica está presente.
    if references:
        return " ".join(references)[:5000] # Limitamos largo para evitar errores de memoria
    return None

def process_logs():
    # Buscar todos los archivos JSON en la carpeta logs
    json_files = glob.glob(str(LOGS_DIR / "*.json"))
    
    count = 0
    for filepath in json_files:
        updated = False
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Archivo corrupto: {filepath}")
                continue

        # Verificar si necesita cálculo
        needs_ppl = data.get("perplexity") is None
        needs_bert = data.get("bert_score") is None
        
        if not (needs_ppl or needs_bert):
            continue # Ya está procesado

        print(f"📊 Procesando: {os.path.basename(filepath)}")
        final_answer = data.get("final_answer", "")

        # 1. Calcular Perplejidad
        if needs_ppl and final_answer:
            ppl_val = calculate_perplexity(final_answer)
            data["perplexity"] = ppl_val
            print(f"   ↳ Perplexity: {ppl_val:.4f}")
            updated = True

        # 2. Calcular BERTScore (Fidelidad con el contexto)
        if needs_bert and final_answer:
            # Usamos el contenido recuperado por las tools como referencia
            reference_text = get_context_references(data)
            
            if reference_text:
                bs_val = calculate_bertscore(final_answer, reference_text)
                data["bert_score"] = bs_val
                print(f"   ↳ BERTScore (F1): {bs_val:.4f}")
                updated = True
            else:
                print("   ↳ Salta BERTScore (No hay contexto de tools para comparar)")

        # Guardar cambios si hubo actualizaciones
        if updated:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            count += 1

    print(f"\n✅ Proceso terminado. {count} logs actualizados.")

if __name__ == "__main__":
    process_logs()