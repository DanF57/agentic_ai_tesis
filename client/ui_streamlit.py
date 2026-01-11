# client/ui_streamlit.py
import streamlit as st
import sys
from pathlib import Path
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from haystack.dataclasses import ChatMessage
from client.agent_client import create_agent
from client.utils.interaction_logger import InteractionLogger

# Page config
st.set_page_config(
    page_title="Agent de Ciencias de Datos",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_reasoning" not in st.session_state:
    st.session_state.current_reasoning = ""
if "agent" not in st.session_state:
    st.session_state.agent = None
if "provider" not in st.session_state:
    st.session_state.provider = None
if "logger" not in st.session_state:
    st.session_state.logger = InteractionLogger(log_dir="logs")


def capture_reasoning(chunk):
    """Capture agent reasoning in real-time"""
    if chunk.content:
        st.session_state.current_reasoning += chunk.content


# Main UI
st.title("🤖 Asistente de Ciencias de Datos")

# Provider selection
if st.session_state.agent is None:
    st.info("👋 Bienvenido! Selecciona un proveedor de LLM para comenzar.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔵 OpenAI", use_container_width=True):
            st.session_state.provider = "openai"
            with st.spinner("Inicializando agente con OpenAI..."):
                st.session_state.agent = create_agent("openai", capture_reasoning)
            st.rerun()
    
    with col2:
        if st.button("🔴 Gemini", use_container_width=True):
            st.session_state.provider = "gemini"
            with st.spinner("Inicializando agente con Gemini..."):
                st.session_state.agent = create_agent("gemini", capture_reasoning)
            st.rerun()
    
    st.stop()

# Sidebar
st.sidebar.success(f"Proveedor: **{st.session_state.provider.upper()}**")
st.sidebar.divider()
st.sidebar.subheader("⚙️ Controles")

if st.sidebar.button("🗑️ Limpiar Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.current_reasoning = ""
    st.rerun()

if st.sidebar.button("🔄 Reiniciar Sesión", use_container_width=True):
    st.session_state.messages = []
    st.session_state.current_reasoning = ""
    st.session_state.agent = None
    st.session_state.provider = None
    st.rerun()

st.sidebar.divider()
st.sidebar.info("""
**Acerca de este Asistente:**

Este agente está especializado en Ciencias de Datos y utiliza:
- 📚 Base de conocimiento interna
- 🌐 Búsqueda web como respaldo
- 🔄 Protocolo ReAct para razonamiento

**Temas que cubre:**
- Machine Learning
- Estadística
- Análisis de Datos
- Visualización
- Y más...
""")

# Main chat area
st.subheader("💬 Conversación")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "reasoning" in message:
            with st.expander("🧠 Ver razonamiento del agente"):
                st.code(message["reasoning"], language=None)

# Chat input
if prompt := st.chat_input("Escribe tu pregunta sobre Ciencias de Datos..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Clear previous reasoning
    st.session_state.current_reasoning = ""
    
    # Build conversation history for logging
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    
    # Start logging
    st.session_state.logger.start_interaction(
        user_question=prompt,
        conversation_history=conversation_history,
        provider=st.session_state.provider or "unknown"
    )
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Build Haystack conversation history
            haystack_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    haystack_messages.append(ChatMessage.from_user(msg["content"]))
                elif msg["role"] == "assistant":
                    haystack_messages.append(ChatMessage.from_assistant(msg["content"]))
            
            # Run agent
            response = st.session_state.agent.run(messages=haystack_messages)
            
            # Extract intermediate steps
            intermediate_steps = response.get("intermediate_steps", [])
            
            # Process intermediate steps
            reasoning_parts = []
            
            for step in intermediate_steps:
                # Capture model output (reasoning)
                if "model_output" in step:
                    model_output = step["model_output"]
                    model_text = model_output.text if hasattr(model_output, "text") else str(model_output)
                    reasoning_parts.append(model_text)
                    st.session_state.logger.log_reasoning(model_text)
                
                # Capture tool calls with RAW responses
                if "tool" in step and "observation" in step:
                    tool_name = step["tool"]
                    arguments = step.get("arguments", {})
                    
                    # Parse arguments to get query
                    if isinstance(arguments, str):
                        import json
                        try:
                            arguments = json.loads(arguments)
                        except:
                            arguments = {"raw": arguments}
                    
                    query = arguments.get("query", "")
                    
                    # RAW response (exactly what the LLM sees)
                    raw_response = str(step["observation"])
                    
                    # Extract similarity scores from RAG responses
                    similarity_scores = []
                    if "Similarity Score:" in raw_response:
                        scores = re.findall(r'Similarity Score: ([\d.]+)', raw_response)
                        similarity_scores = [float(s) for s in scores]
                    
                    # Log complete tool call
                    if query:
                        st.session_state.logger.log_tool_call(
                            tool_name=tool_name,
                            query=query,
                            raw_response=raw_response,
                            similarity_scores=similarity_scores
                        )
                    
                    # Add to reasoning display
                    reasoning_parts.append(f"\n[TOOL CALL] {tool_name}(query='{query}')")
                    reasoning_parts.append(f"[OBSERVATION] {raw_response[:200]}...")  # Truncate for display
            
            # Update reasoning for non-streaming providers
            if not st.session_state.current_reasoning:
                st.session_state.current_reasoning = "\n".join(reasoning_parts)
            
            # Get final response
            final_response = response["messages"][-1].text
            
            # Extract FINAL ANSWER
            if "FINAL ANSWER" in final_response:
                parts = final_response.split("FINAL ANSWER", 1)
                display_response = parts[1].strip() if len(parts) > 1 else final_response
            else:
                display_response = final_response
            
            # End logging
            st.session_state.logger.end_interaction(final_answer=display_response)
            
            # Display response
            st.markdown(display_response)
            
            # Show reasoning
            if st.session_state.current_reasoning:
                with st.expander("🧠 Ver razonamiento del agente"):
                    st.code(st.session_state.current_reasoning, language=None)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_response,
                "reasoning": st.session_state.current_reasoning
            })
    
    st.rerun()