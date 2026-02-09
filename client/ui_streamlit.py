import streamlit as st
import sys
import json
from pathlib import Path
import re

# =========================
# PATH DEL PROYECTO
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# IMPORTS INTERNOS
# =========================
from haystack.dataclasses import ChatMessage
from logger import ExecutionLogger
from agent_client import (
    create_student_agent,
    create_analizer_agent,
    create_planification_agent
)



# =========================
# CONFIGURACIÓN STREAMLIT
# =========================
st.set_page_config(
    page_title="Agentic AI – Ciencias de Datos",
    page_icon="🤖",
    layout="wide"
)

INDEX_FILE = Path("logs_docente/index.json")

# =========================
# HELPERS
# =========================
def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"topic_reports": [], "lesson_plans": []}


def reset_session(full_reset=True):
    st.session_state.messages = []
    st.session_state.agent = None
    st.session_state.agent_type = None
    if full_reset:
        st.session_state.provider = None


def parse_answer_blocks(text: str):
    explanation = None
    answer = text

    if "[EXPLANATION]" in text:
        _, rest = text.split("[EXPLANATION]", 1)
        if "[ANSWER]" in rest:
            explanation, answer = rest.split("[ANSWER]", 1)
        else:
            explanation = rest
            answer = ""
    elif "[ANSWER]" in text:
        _, answer = text.split("[ANSWER]", 1)

    return explanation.strip() if explanation else None, answer.strip()

def extract_groups(report_content: str):
    """
    Devuelve una lista de títulos de grupo, por ejemplo:
    'Group 1: Train/Test splitting and holdouts'
    """
    pattern = r"### (Group\s+\d+:\s+.+)"
    return re.findall(pattern, report_content)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_type" not in st.session_state:
    st.session_state.agent_type = None

if "provider" not in st.session_state:
    st.session_state.provider = None

if "user_role" not in st.session_state:
    st.session_state.user_role = "student"

# =========================
# SIDEBAR – CONFIGURACIÓN
# =========================
st.sidebar.title("⚙️ Configuración")

# ---- Rol ----
role = st.sidebar.radio(
    "Modo de uso:",
    ["student", "teacher"],
    index=0 if st.session_state.user_role == "student" else 1
)

if role != st.session_state.user_role:
    st.session_state.user_role = role
    reset_session(full_reset=True)
    st.rerun()

st.sidebar.divider()

# ---- Proveedor (BOTONES, no selectbox) ----
st.sidebar.subheader("🔌 Proveedor de modelo")

if st.session_state.provider is None:
    if st.sidebar.button("🔵 OpenAI", use_container_width=True):
        st.session_state.provider = "openai"
        reset_session(full_reset=False)
        st.rerun()

    if st.sidebar.button("🔴 Gemini", use_container_width=True):
        st.session_state.provider = "gemini"
        reset_session(full_reset=False)
        st.rerun()
else:
    st.sidebar.success(f"Proveedor activo: {st.session_state.provider.upper()}")

    if st.sidebar.button("🔁 Cambiar proveedor", use_container_width=True):
        reset_session(full_reset=True)
        st.rerun()

st.sidebar.divider()

# ---- Controles ----
if st.sidebar.button("🧹 Limpiar chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🔄 Reiniciar sesión", use_container_width=True):
    reset_session(full_reset=True)
    st.rerun()

# =========================
# BLOQUEO SIN PROVEEDOR
# =========================
if st.session_state.provider is None:
    st.info("👋 Selecciona un proveedor de modelo para comenzar.")
    st.stop()

# =========================
# MODO ESTUDIANTE
# =========================
if st.session_state.user_role == "student":

    st.title("👩‍🎓 Modo Estudiante")

    if st.session_state.agent is None:
        st.session_state.agent = create_student_agent(st.session_state.provider)
        st.session_state.agent_type = "student"

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("explanation"):
                with st.expander("📘 Explicación"):
                    st.markdown(msg["explanation"])

    if prompt := st.chat_input("Escribe tu pregunta sobre Ciencias de Datos..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        logger = ExecutionLogger()
        logger.start(
            user_question=prompt,
            provider=st.session_state.provider,
            history=conversation_history,
            user_role="student"
        )

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.agent.run(
                    messages=[ChatMessage.from_user(prompt)]
                )
                raw_text = response["messages"][-1].text

        logger.end(raw_text)

        explanation, answer = parse_answer_blocks(raw_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "explanation": explanation
        })

        st.rerun()

# =========================
# MODO DOCENTE
# =========================
else:
    st.title("👩‍🏫 Modo Docente")

    index_data = load_index()

    # ---- Sidebar: reportes existentes ----
    st.sidebar.subheader("📂 Reportes existentes")
    if index_data["topic_reports"]:
        for r in index_data["topic_reports"]:
            st.sidebar.markdown(
                f"- **{r['subreddit']}** | top {r['top_n']}  \n{r['generated_at'][:10]} - {r['generated_at'][11:16]} - **{r['model']}** "
            )
    else:
        st.sidebar.markdown("_No hay reportes aún_")

    st.sidebar.divider()

    action = st.sidebar.radio(
        "Acción:",
        [
            "Analizar temas desde subreddit (Agente 1)",
            "Crear planificación de clase (Agente 2)"
        ]
    )

    # =========================
    # AGENTE 1
    # =========================
    if action == "Analizar temas desde subreddit (Agente 1)":

        if st.session_state.agent_type != "agente_1":
            st.session_state.agent = create_analizer_agent(st.session_state.provider)
            st.session_state.agent_type = "agente_1"

        st.subheader("🧠 Análisis de temas")

        subreddit = st.selectbox(
            "Subreddit",
            ["datascience", "dataanalysis", "dataisbeautiful", "AskStatistics", "datascienceproject"]
        )

        top_n = st.selectbox(
            "Número de posts",
            [10, 20, 30, 40, 50]
        )



        if st.button("Ejecutar análisis"):

            prompt = f"subreddit: {subreddit} top {top_n}"

            logger = ExecutionLogger()
            logger.start(
                user_question=prompt,
                provider=st.session_state.provider,
                history=[],
                user_role="teacher"
            )

            with st.spinner("Ejecutando Agente 1..."):
                response = st.session_state.agent.run(
                    messages=[ChatMessage.from_user(prompt)]
                )
                result = response["messages"][-1].text

            logger.end(result)

            st.markdown("### 📊 Reporte generado")
            st.text_area(
                "Salida del Agente 1",
                value=result,
                height=500
            )

    # =========================
    # AGENTE 2 (placeholder)
    # =========================
    if action == "Crear planificación de clase (Agente 2)":

        if st.session_state.agent_type != "agente_2":
            st.session_state.agent = create_planification_agent(
                st.session_state.provider
            )
            st.session_state.agent_type = "agente_2"

        st.subheader("📘 Planificación de clase (Modelo ADDIE)")

        index_data = load_index()

        if not index_data["topic_reports"]:
            st.warning("No hay reportes de temas disponibles.")
            st.stop()

        report_map = {
            f"{r['subreddit']} | top {r['top_n']} - {r['generated_at'][:10]} - {r['generated_at'][11:16]} - {r['model']} ": r
            for r in index_data["topic_reports"]
        }

        selected_report_label = st.selectbox(
            "Selecciona un reporte de análisis:",
            list(report_map.keys())
        )

        selected_report = report_map[selected_report_label]

        # Cargar contenido del reporte
        with open(selected_report["file"], "r", encoding="utf-8") as f:
            report_json = json.load(f)

        report_content = report_json["report_content"]
        report_id = report_json["report_id"]

        groups = extract_groups(report_content)

        if not groups:
            st.error("No se pudieron detectar grupos en el reporte.")
            st.stop()

        selected_topic = st.selectbox(
            "Selecciona el tema para la planificación:",
            groups
        )
        if st.button("Generar planificación ADDIE"):

                    # Construir prompt
            prompt = f"""
                {st.session_state.agent.system_prompt}

                TEMA SELECCIONADO:
                {selected_topic}

                REPORTE COMPLETO:
                {report_content}
                """

            logger = ExecutionLogger()
            logger.start(
                user_question=f"ADDIE plan for {selected_topic}",
                provider=st.session_state.provider,
                history=[],
                user_role="teacher"
            )

            # Metadata para el logger
            logger.log_data["source_report_id"] = report_id
            logger.log_data["selected_topic"] = selected_topic

            with st.spinner("Generando planificación ADDIE..."):
                response = st.session_state.agent.run(
                    messages=[ChatMessage.from_user(prompt)]
                )
                result = response["messages"][-1].text

            logger.end(result)

            st.markdown("### 📘 Planificación generada")
            st.text_area(
                "Resultado (ADDIE)",
                value=result,
                height=600
            )
