# Agentic AI - Sistema de Agentes para Ciencias de Datos

Repositorio del sistema de agentes inteligentes desarrollado para la tesis de grado en Ciencias de Computación. Este proyecto implementa un sistema multi-agente que asiste tanto a estudiantes como a docentes en el dominio de Ciencias de Datos mediante técnicas de RAG (Retrieval-Augmented Generation) y herramientas especializadas.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Componentes Principales](#componentes-principales)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Sistema](#uso-del-sistema)
- [Sistema de Logging y Evaluación](#sistema-de-logging-y-evaluación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Herramientas Disponibles](#herramientas-disponibles)

## 🎯 Descripción General

Este sistema implementa tres agentes especializados que trabajan con una base de conocimiento construida a partir de posts de Reddit sobre Ciencias de Datos:

1. **Agente Estudiante**: Asistente académico que responde preguntas de estudiantes utilizando RAG y búsqueda web.
2. **Agente Analizador (Agente 1)**: Analiza y agrupa preguntas similares de subreddits para identificar temas recurrentes.
3. **Agente de Planificación (Agente 2)**: Genera planificaciones de clase siguiendo el modelo ADDIE basándose en análisis temáticos previos.

### Características Principales

- ✅ **RAG (Retrieval-Augmented Generation)**: Base de conocimiento vectorial con ChromaDB
- ✅ **Búsqueda Web en Tiempo Real**: Integración con SerperDev para búsquedas actualizadas
- ✅ **Multi-Proveedor LLM**: Soporte para OpenAI (GPT) y Google Gemini
- ✅ **Sistema de Logging Completo**: Registro detallado para evaluación de recuperación y generación
- ✅ **Interfaz Streamlit**: UI intuitiva para estudiantes y docentes
- ✅ **Arquitectura MCP**: Comunicación mediante Model Context Protocol

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaz Streamlit                        │
│              (Modo Estudiante / Modo Docente)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cliente (Agent Client)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Agente       │  │ Agente       │  │ Agente       │     │
│  │ Estudiante   │  │ Analizador   │  │ Planificación│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                           │                                │
│                    LLM Factory                             │
│              (OpenAI / Gemini)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Protocol
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Servidor MCP                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ RAG Tool     │  │ Web Search   │  │ Collect Posts│     │
│  │ (Knowledge  │  │ Tool         │  │ Tool         │     │
│  │  Base)       │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ChromaDB    │  │  SerperDev   │  │  ChromaDB    │
│  (Vector DB) │  │  (Web API)   │  │  (Metadata)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 🔧 Componentes Principales

### 1. Servidor MCP (`server/my_server.py`)

Servidor FastMCP que expone herramientas mediante Model Context Protocol:

- **`search_knowledge_base`**: Búsqueda semántica en la base de conocimiento vectorial
- **`search_web`**: Búsqueda web en tiempo real usando Google Search
- **`collect_posts`**: Recolección y filtrado de posts de subreddits específicos

### 2. Cliente de Agentes (`client/agent_client.py`)

Factory para crear los tres tipos de agentes:

- **`create_student_agent()`**: Agente con herramientas RAG y Web Search
- **`create_analizer_agent()`**: Agente con herramienta de recolección de posts
- **`create_planification_agent()`**: Agente con herramienta de búsqueda web para planificación ADDIE

### 3. Sistema de Logging (`logger.py`)

Sistema completo de registro para evaluación:

- **Logs de Ejecución**: Registro de cada interacción con timestamps, queries, respuestas y llamadas a herramientas
- **Métricas**: Tiempo de ejecución, modelo utilizado, provider
- **Estructura de Logs**:
  - `logs_estudiante/`: Logs de interacciones de estudiantes
  - `logs_docente/executions/`: Logs técnicos de agentes docente
  - `logs_docente/reports/`: Reportes generados (análisis temáticos y planificaciones ADDIE)

### 4. Interfaz de Usuario (`client/ui_streamlit.py`)

Interfaz Streamlit con dos modos:

- **Modo Estudiante**: Chat interactivo con el agente asistente
- **Modo Docente**: 
  - Análisis de temas desde subreddits (Agente 1)
  - Generación de planificaciones ADDIE (Agente 2)

### 5. Base de Conocimiento (`server/knowledge/`)

- **ChromaDB**: Base de datos vectorial con embeddings OpenAI (`text-embedding-3-small`)
- **Dataset**: Posts de Reddit procesados y vectorizados
- Scripts de procesamiento y población de la base de datos

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- Cuenta de OpenAI o Google Gemini (API keys)
- Cuenta de SerperDev (opcional, para búsqueda web)

### Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd agentic_ai_tesis
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:

Crear archivo `.env` en la raíz del proyecto:
```env
# OpenAI (requerido para embeddings y opcional para LLM)
OPENAI_API_KEY=tu_api_key_openai

# Google Gemini (opcional, para usar Gemini como LLM)
GOOGLE_API_KEY=tu_api_key_gemini

# SerperDev (opcional, para búsqueda web)
SERPER_API_KEY=tu_api_key_serper
```

5. **Configurar modelos**:

Editar `config/models.yaml` para seleccionar los modelos:
```yaml
providers:
  openai:
    name: "OpenAI"
    model: "gpt-4o-mini"  # o el modelo que prefieras
  gemini:
    name: "Gemini"
    model: "gemini-2.0-flash-exp"  # o el modelo que prefieras
```

### Inicializar Base de Conocimiento

Si la base de conocimiento no está poblada:

```bash
cd server/knowledge
python populate_db.py
```

## 💻 Uso del Sistema

### Iniciar el Servidor MCP

En una terminal:
```bash
cd server
python my_server.py
```

El servidor se iniciará en `http://localhost:8000`

### Iniciar la Interfaz Streamlit

En otra terminal:
```bash
streamlit run client/ui_streamlit.py
```

La interfaz estará disponible en `http://localhost:8501`

### Flujo de Uso

#### Modo Estudiante

1. Seleccionar proveedor (OpenAI o Gemini)
2. Escribir pregunta sobre Ciencias de Datos
3. El agente:
   - Busca primero en la base de conocimiento (RAG)
   - Si es necesario, busca en la web
   - Genera respuesta estructurada con `[EXPLANATION]` y `[ANSWER]`

#### Modo Docente - Agente 1 (Análisis Temático)

1. Seleccionar subreddit (ej: `datascience`, `dataanalysis`)
2. Seleccionar número de posts a analizar
3. El agente:
   - Recolecta posts con preguntas (títulos con `?`)
   - Los agrupa por temas similares
   - Genera reporte de análisis temático

#### Modo Docente - Agente 2 (Planificación ADDIE)

1. Seleccionar un reporte de análisis temático previo
2. Seleccionar un tema específico del reporte
3. El agente:
   - Genera planificación de clase siguiendo modelo ADDIE
   - Incluye: Análisis, Diseño, Desarrollo, Implementación, Evaluación

## 📊 Sistema de Logging y Evaluación

El sistema registra información detallada para evaluación:

### Datos Registrados

- **Query original y query reconstruida**: Para evaluar transformaciones
- **Modelo utilizado**: Provider y modelo específico
- **Tiempo de ejecución**: Métricas de rendimiento
- **Llamadas a herramientas**: 
  - Herramienta utilizada (RAG, WEB, collect_posts)
  - Query enviada a la herramienta
  - Resultados obtenidos
  - Tiempo de ejecución por herramienta
- **Razonamiento del agente**: Proceso de pensamiento
- **Respuesta final**: Respuesta generada al usuario
- **Caso de uso**: Tipo de interacción (estudiante/docente)

### Estructura de Logs

```
logs_estudiante/
  └── log_YYYYMMDD_HHMMSS_<id>.json

logs_docente/
  ├── executions/
  │   ├── agente_1/
  │   │   └── log_YYYYMMDD_HHMMSS_<id>.json
  │   └── agente_2/
  │       └── log_YYYYMMDD_HHMMSS_<id>.json
  ├── reports/
  │   ├── agente_1_topics/
  │   │   └── <subreddit>_YYYYMMDD_HHMMSS.json
  │   └── agente_2_addie/
  │       └── addie_YYYYMMDD_HHMMSS.json
  └── index.json
```

### Evaluación

Los logs permiten evaluar:
- **Recuperación**: Relevancia de documentos recuperados (scores, top-k)
- **Generación**: Calidad de respuestas generadas
- **Perplejidad**: (Para modelos generativos, si se implementa)
- **Eficiencia**: Tiempos de ejecución por componente

## 📁 Estructura del Proyecto

```
agentic_ai_tesis/
├── client/                    # Cliente y UI
│   ├── agent_client.py        # Factory de agentes
│   ├── ui_streamlit.py        # Interfaz Streamlit
│   └── utils/
│       └── llm_factory.py     # Factory de LLMs
├── server/                    # Servidor MCP
│   ├── my_server.py           # Servidor principal
│   └── knowledge/             # Base de conocimiento
│       ├── chroma_db_data/    # Base de datos vectorial
│       ├── populate_db.py     # Script de población
│       ├── process_json.py    # Procesamiento de datos
│       └── *.json             # Datasets procesados
├── config/
│   └── models.yaml            # Configuración de modelos
├── tests/                     # Tests unitarios
│   ├── test_vector_db.py
│   ├── test_querys.py
│   ├── test_trending_posts.py
│   └── test_llm_factory.py
├── logs_estudiante/           # Logs de estudiantes
├── logs_docente/              # Logs y reportes de docentes
├── logger.py                  # Sistema de logging
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🛠️ Herramientas Disponibles

### 1. `search_knowledge_base(query: str)`

Búsqueda semántica en la base de conocimiento vectorial.

**Parámetros**:
- `query`: Consulta en texto natural

**Retorna**: Top 5 documentos más relevantes con:
- Título del post
- Contenido
- URL fuente
- Score de similitud

**Uso**: Herramienta principal del Agente Estudiante para recuperar información contextual.

### 2. `search_web(query: str, top_k: int = 5)`

Búsqueda web en tiempo real usando Google Search (SerperDev).

**Parámetros**:
- `query`: Consulta de búsqueda
- `top_k`: Número de resultados (default: 5)

**Retorna**: Resultados de búsqueda web con:
- Título
- Snippet
- URL

**Uso**: Complemento cuando RAG no tiene información suficiente.

### 3. `collect_posts(subreddit: str, top_n: int = 30)`

Recolecta y filtra posts de un subreddit específico.

**Parámetros**:
- `subreddit`: Nombre del subreddit (ej: `datascience`)
- `top_n`: Número de posts a retornar (default: 30)

**Retorna**: Lista de posts filtrados (solo con `?` en título) ordenados por votos ascendentes.

**Uso**: Herramienta del Agente Analizador para identificar temas recurrentes.

## 🔍 Notas Técnicas

### Distinción entre Tarea y Recurso

El sistema mantiene coherencia en la nomenclatura:
- **Tarea**: Operación que realiza un agente (ej: "responder pregunta", "analizar temas")
- **Recurso**: Herramienta o fuente de datos (ej: "RAG", "Web Search", "Knowledge Base")

### Model Context Protocol (MCP)

El sistema utiliza MCP para comunicación entre cliente y servidor, permitiendo:
- Desacoplamiento entre agentes y herramientas
- Escalabilidad y extensibilidad
- Protocolo estándar para integración de herramientas

### Embeddings

- **Modelo**: `text-embedding-3-small` de OpenAI
- **Base de datos**: ChromaDB con distancia coseno
- **Colección**: `reddit_datascience_openai`

## 📝 Próximas Mejoras

- [ ] Diagramas de arquitectura detallados
- [ ] Implementación de métricas de perplejidad
- [ ] Dashboard de evaluación de logs
- [ ] Soporte para más proveedores LLM
- [ ] Mejoras en el sistema de caching

## 📄 Licencia

Este proyecto es parte de una tesis de grado y está destinado a uso académico.


## 👤 Autor

- Daniel Eduardo Flores Serrano

Desarrollado para la tesis de grado en Ciencias de Computación - UTPL

---

**Nota**: Este README está diseñado para documentar el proyecto completo y facilitar su comprensión y uso en el contexto de la investigación de tesis.
