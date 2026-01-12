# client/ui_streamlit.py
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from haystack.dataclasses import ChatMessage
from client.agent_client import create_agent

# Page config
st.set_page_config(
    page_title="Agent de Ciencias de Datos",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "provider" not in st.session_state:
    st.session_state.provider = None


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
                st.session_state.agent = create_agent("openai")
            st.rerun()
    
    with col2:
        if st.button("🔴 Gemini", use_container_width=True):
            st.session_state.provider = "gemini"
            with st.spinner("Inicializando agente con Gemini..."):
                st.session_state.agent = create_agent("gemini")
            st.rerun()
    
    st.stop()

# Sidebar
st.sidebar.success(f"Proveedor: **{st.session_state.provider.upper()}**")
st.sidebar.divider()
st.sidebar.subheader("⚙️ Controles")

if st.sidebar.button("🗑️ Limpiar Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🔄 Reiniciar Sesión", use_container_width=True):
    st.session_state.messages = []
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

""")

# Main chat area
st.subheader("💬 Conversación")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "reasoning" in message and message["reasoning"]:
            with st.expander("🧠 Ver razonamiento del agente"):
                st.code(message["reasoning"], language=None)

# Chat input
if prompt := st.chat_input("Escribe tu pregunta sobre Ciencias de Datos..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Build conversation history for logging
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    
    
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
            
            # Get complete response (includes reasoning + FINAL ANSWER)
            final_response = response["messages"][-1].text
            
            # Split reasoning from final answer
            if "FINAL ANSWER" in final_response:
                parts = final_response.split("FINAL ANSWER", 1)
                reasoning_text = parts[0].strip()  # Everything before FINAL ANSWER
                display_response = parts[1].strip()  # Everything after FINAL ANSWER
            else:
                # No FINAL ANSWER marker found
                reasoning_text = ""
                display_response = final_response
            
            
            # Display response (only the final answer part)
            st.markdown(display_response)
            
            # Show reasoning in expander (everything before FINAL ANSWER)
            if reasoning_text:
                with st.expander("🧠 Ver razonamiento del agente"):
                    st.code(reasoning_text, language=None)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_response,
                "reasoning": reasoning_text
            })
    
    st.rerun()