# logger.py
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================

BASE_DIR = Path(__file__).parent

# Logs por rol
LOGS_ESTUDIANTE_DIR = BASE_DIR / "logs_estudiante"
LOGS_DOCENTE_DIR = BASE_DIR / "logs_docente"

# Logs técnicos (docente)
EXECUTIONS_DIR = LOGS_DOCENTE_DIR / "executions"
AGENTE_1_EXEC_DIR = EXECUTIONS_DIR / "agente_1"
AGENTE_2_EXEC_DIR = EXECUTIONS_DIR / "agente_2"

# Reportes consumibles
REPORTS_BASE_DIR = LOGS_DOCENTE_DIR / "reports"
AGENTE_1_REPORTS_DIR = REPORTS_BASE_DIR / "agente_1_topics"
AGENTE_2_REPORTS_DIR = REPORTS_BASE_DIR / "agente_2_addie"

# Índice para UI
INDEX_FILE = LOGS_DOCENTE_DIR / "index.json"

# Log temporal de tools (server → client)
TEMP_TOOL_LOG = LOGS_ESTUDIANTE_DIR / "temp_tool_stream.jsonl"

# Crear directorios necesarios
LOGS_ESTUDIANTE_DIR.mkdir(exist_ok=True)
LOGS_DOCENTE_DIR.mkdir(exist_ok=True)

EXECUTIONS_DIR.mkdir(exist_ok=True)
AGENTE_1_EXEC_DIR.mkdir(exist_ok=True)
AGENTE_2_EXEC_DIR.mkdir(exist_ok=True)

REPORTS_BASE_DIR.mkdir(exist_ok=True)
AGENTE_1_REPORTS_DIR.mkdir(exist_ok=True)
AGENTE_2_REPORTS_DIR.mkdir(exist_ok=True)


# =========================
# LOGGER PRINCIPAL
# =========================

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
            "error_flag": False
        }

    # -------------------------
    # START
    # -------------------------
    def start(self, user_question, provider, history, user_role="student"):
        self.start_timestamp = time.time()
        self.log_data["start_time"] = self.start_timestamp
        self.log_data["timestamp"] = datetime.now().isoformat()
        self.log_data["user_rol"] = user_role
        self.log_data["user_question"] = user_question
        self.log_data["provider"] = provider
        self.log_data["conversation_history"] = json.loads(json.dumps(history))

    # -------------------------
    # TOOL LOGGING (SERVER)
    # -------------------------
    @staticmethod
    def record_tool_execution(tool_name, query, results_dict):
        entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "query": query,
            "execution_time_seconds": results_dict.get("execution_time_seconds"),
            "results": results_dict.get("results", [])
        }

        try:
            with open(TEMP_TOOL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Error logging tool execution: {e}")

    # -------------------------
    # END
    # -------------------------
    def end(self, full_response_text):
        self.end_timestamp = time.time()
        self.log_data["end_time"] = self.end_timestamp
        self.log_data["execution_time_seconds"] = self.end_timestamp - self.start_timestamp

        # Validación de respuesta
        if not full_response_text:
            self.log_data["agent_reasoning"] = "[ERROR: Sin razonamiento]"
            self.log_data["final_answer"] = "[ERROR: El agente no pudo generar una respuesta]"
            self.log_data["error_flag"] = True
        else:
            if "[ANSWER]" in full_response_text:
                parts = full_response_text.split("[ANSWER]", 1)
                self.log_data["agent_reasoning"] = parts[0].strip()
                self.log_data["final_answer"] = parts[1].strip()
            else:
                self.log_data["agent_reasoning"] = ""
                self.log_data["final_answer"] = full_response_text

        # Recuperar llamadas a tools
        self._harvest_tool_logs()

        # Si falló, no persistimos nada
        if self.log_data.get("error_flag"):
            print("⚠️ Ejecución fallida, no se guarda log ni reportes.")
            return

        # -------------------------
        # DETERMINAR DESTINO
        # -------------------------
        if self.log_data.get("user_rol") == "teacher":
            user_question = (self.log_data.get("user_question") or "").lower()

            if "subreddit" in user_question:
                target_dir = AGENTE_1_EXEC_DIR
                agente = "agente_1"
            else:
                target_dir = AGENTE_2_EXEC_DIR
                agente = "agente_2"
        else:
            target_dir = LOGS_ESTUDIANTE_DIR
            agente = "student"

        # -------------------------
        # GUARDAR LOG TÉCNICO
        # -------------------------
        filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.execution_id[:8]}.json"
        filepath = target_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Log guardado ({agente}): {filepath}")

        # -------------------------
        # AGENTE 1 → TOPIC REPORT
        # -------------------------
        if agente == "agente_1":
            subreddit, top_n = self._parse_subreddit_and_top_n(user_question)
            timestamp = self.log_data["timestamp"]

            report_info = self._save_topic_report(
                final_answer=self.log_data["final_answer"],
                subreddit=subreddit,
                top_n=top_n,
                timestamp=timestamp
            )

            self._update_index_topic_reports({
                "report_id": report_info["report_id"],
                "subreddit": subreddit,
                "top_n": top_n,
                "generated_at": timestamp,
                "file": report_info["file"],
                "model": self.log_data.get("provider")
            })

            print(f"📘 Topic report guardado: {report_info['file']}")

        # --- AGENTE 2: guardar planificación ADDIE ---
        if agente == "agente_2":
            try:
                addie_info = self._save_addie_plan(
                    plan_content=self.log_data["final_answer"],
                    source_report_id=self.log_data.get("source_report_id"),
                    selected_topic=self.log_data.get("selected_topic"),
                    timestamp=self.log_data["timestamp"]
                )

                self._update_index_lesson_plans({
                    "plan_id": addie_info["plan_id"],
                    "source_report_id": self.log_data.get("source_report_id"),
                    "selected_topic": self.log_data.get("selected_topic"),
                    "generated_at": self.log_data["timestamp"],
                    "file": addie_info["file"]
                })

                print(f"📘 ADDIE guardado: {addie_info['file']}")

            except Exception as e:
                print(f"❌ Error guardando ADDIE: {e}")


    # =========================
    # HELPERS
    # =========================

    def _harvest_tool_logs(self):
        if not TEMP_TOOL_LOG.exists():
            return

        relevant_tools = []

        with open(TEMP_TOOL_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if self.start_timestamp - 0.1 <= record["timestamp"] <= self.end_timestamp + 0.5:
                        del record["timestamp"]
                        relevant_tools.append(record)
                except Exception:
                    continue

        self.log_data["tool_calls"] = relevant_tools

    def _parse_subreddit_and_top_n(self, user_question: str):
        """
        Formatos soportados:
        - subreddit: datascience
        - subreddit: datascience top 10
        """
        cleaned = user_question.replace("subreddit", "").replace(":", "").strip()
        parts = cleaned.split()

        subreddit = parts[0] if parts else "unknown"
        top_n = 30

        if "top" in parts:
            try:
                idx = parts.index("top")
                top_n = int(parts[idx + 1])
            except Exception:
                pass

        return subreddit, top_n

    def _save_topic_report(self, final_answer: str, subreddit: str, top_n: int, timestamp: str):
        timestamp_clean = (
            timestamp.replace(":", "")
            .replace("-", "")
            .replace(".", "")
            .replace("T", "_")
        )

        safe_subreddit = subreddit.replace(" ", "_")

        report_id = f"topic_{timestamp_clean}"

        report_data = {
            "report_id": report_id,
            "source": {
                "subreddit": subreddit,
                "top_n": top_n,
                "model": self.log_data.get("provider"),
            },
            "generated_at": timestamp,
            "report_content": final_answer
        }

        filename = f"{safe_subreddit}_{timestamp_clean}.json"
        filepath = AGENTE_1_REPORTS_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return {
            "report_id": report_id,
            "file": str(filepath)
        }

    def _save_addie_plan(
        self,
        plan_content: str,
        source_report_id: str,
        selected_topic: str,
        timestamp: str
    ):
        timestamp_clean = (
            timestamp.replace(":", "")
            .replace("-", "")
            .replace(".", "")
            .replace("T", "_")
        )

        plan_id = f"addie_{timestamp_clean}"

        plan_data = {
            "plan_id": plan_id,
            "source_report_id": source_report_id,
            "selected_topic": selected_topic,
            "generated_at": timestamp,
            "model": {
                "provider": self.log_data.get("provider"),
                "agent": "agente_2"
            },
            "plan_content": plan_content
        }

        filename = f"{plan_id}.json"
        filepath = AGENTE_2_REPORTS_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)

        return {
            "plan_id": plan_id,
            "file": str(filepath)
        }


    def _update_index_topic_reports(self, report_meta: dict):
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = {
                "topic_reports": [],
                "lesson_plans": []
            }

        index_data["topic_reports"].append(report_meta)

        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)


    def _update_index_lesson_plans(self, plan_meta: dict):
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = {
                "topic_reports": [],
                "lesson_plans": []
            }

        index_data["lesson_plans"].append(plan_meta)

        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

