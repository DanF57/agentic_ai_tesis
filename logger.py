# logger.py
import json
import time
import os
import uuid
from datetime import datetime
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
TEMP_TOOL_LOG = LOGS_DIR / "temp_tool_stream.jsonl"

# Asegurar que existan los directorios
LOGS_DIR.mkdir(exist_ok=True)

class ExecutionLogger:
    def __init__(self):
        self.start_timestamp = None
        self.end_timestamp = None
        self.execution_id = str(uuid.uuid4())
        self.log_data = {
            "user_rol": "student",
            "timestamp": None,
            "provider": None,
            "user_question": None,
            "conversation_history": [],
            "start_time": None,
            "end_time": None,
            "execution_time_seconds": None,
            "tool_calls": [],
            "agent_reasoning": None,
            "final_answer": None,
            "bert_score": None,
            "perplexity": None
        }

    def start(self, user_question, provider, history):
        """Inicia el temporizador y guarda metadatos."""
        self.start_timestamp = time.time()
        self.log_data["start_time"] = self.start_timestamp
        self.log_data["timestamp"] = datetime.now().isoformat()
        self.log_data["user_question"] = user_question
        self.log_data["provider"] = provider
        # Copiamos el historial para evitar referencias mutables
        self.log_data["conversation_history"] = json.loads(json.dumps(history))

    @staticmethod
    def record_tool_execution(tool_name, query, results_dict):
        """
        Método estático usado por el SERVER para registrar ejecuciones al vuelo.
        Escribe en un archivo JSONL (append-only).
        """
        entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "query": query,
            "results": results_dict.get("results", [])
        }
        
        # Escribir en archivo temporal compartido de forma segura
        try:
            with open(TEMP_TOOL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error logging tool execution: {e}")

    def end(self, full_response_text):
        """Finaliza el temporizador, procesa la respuesta y guarda el archivo."""
        self.end_timestamp = time.time()
        self.log_data["end_time"] = self.end_timestamp
        self.log_data["execution_time_seconds"] = self.end_timestamp - self.start_timestamp

        # 1. Separar Reasoning de Final Answer
        if "FINAL ANSWER" in full_response_text:
            parts = full_response_text.split("FINAL ANSWER", 1)
            self.log_data["agent_reasoning"] = parts[0].strip()
            self.log_data["final_answer"] = parts[1].strip()
        else:
            self.log_data["agent_reasoning"] = "" # O todo el texto si no hay separación
            self.log_data["final_answer"] = full_response_text

        # 2. Recuperar Tool Calls del archivo temporal
        self._harvest_tool_logs()

        # 3. Guardar archivo final
        filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.execution_id[:8]}.json"
        filepath = LOGS_DIR / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Log guardado en: {filepath}")

    def _harvest_tool_logs(self):
        """Lee el archivo temporal y extrae las herramientas ejecutadas durante esta sesión."""
        if not TEMP_TOOL_LOG.exists():
            return

        relevant_tools = []
        try:
            with open(TEMP_TOOL_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Verificar si el timestamp de la herramienta cae dentro de la ejecución actual
                        # Agregamos un pequeño margen de error (0.1s) por latencia de escritura
                        if self.start_timestamp - 0.1 <= record["timestamp"] <= self.end_timestamp + 0.5:
                            # Eliminamos el timestamp del registro final para limpiar el JSON
                            del record["timestamp"]
                            relevant_tools.append(record)
                    except json.JSONDecodeError:
                        continue
            
            self.log_data["tool_calls"] = relevant_tools
            
            # Opcional: Limpiar el archivo temporal si crece mucho, 
            # pero para una tesis es mejor dejarlo como backup o limpiarlo manualmente.
            
        except Exception as e:
            print(f"Error reading temp tool logs: {e}")