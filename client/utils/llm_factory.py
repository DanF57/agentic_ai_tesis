from haystack.utils import Secret

def create_llm(provider: str, model: str, tools: list = None):
    """
    Fábrica de modelos LLM para Agentes Haystack.
    Soporta: OpenAI, Gemini y HuggingFace (ej: DeepSeek).
    """

    provider = provider.lower()

    # =======================
    # 1. OPENAI
    # =======================
    if provider == "openai":
        try:
            from haystack.components.generators.chat import OpenAIChatGenerator
        except ImportError:
            raise ImportError("Falta OpenAIChatGenerator. Actualiza haystack-ai.")

        return OpenAIChatGenerator(
            model=model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            tools=tools,
        )

    # =======================
    # 2. GEMINI (Google)
    # =======================
    if provider in ("gemini"):
        try:
            from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
        except ImportError:
            raise ImportError("Falta GoogleGenAIChatGenerator. Instala google-generativeai.")

        return GoogleGenAIChatGenerator(
            model=model,
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            tools=tools,
        )

    # =======================
    # 3. HUGGING FACE (DeepSeek u otros)
    # =======================
    if provider in ("huggingface"):
        try:
            from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
            from haystack.utils.hf import HFGenerationAPIType
            
            api_type = HFGenerationAPIType.SERVERLESS_INFERENCE_API
           
        except ImportError:
            raise ImportError("Falta HuggingFaceAPIChatGenerator. Actualiza haystack-ai.")

        return HuggingFaceAPIChatGenerator(api_type=api_type,
                                                    api_params={"model": model,
                                                                "provider": "together"},
                                                    token=Secret.from_env_var("HF_API_KEY"),
                                                    tools=tools,
                                                    )

    raise ValueError(
        f"Proveedor '{provider}' no soportado. Usa: openai, gemini, huggingface."
    )

