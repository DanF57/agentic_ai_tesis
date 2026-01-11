from haystack.utils import Secret


def create_llm(provider: str, model: str, tools: list = None, streaming_callback=None):
    """
    Factory para crear generadores LLM para Agentes Haystack.
    Soporta: OpenAI y Gemini.
    
    Args:
        provider: "openai" o "gemini"
        model: Nombre del modelo específico
        tools: Lista de herramientas disponibles
        streaming_callback: Callback para streaming
        
    Returns:
        Chat generator instance
    """
    provider = provider.lower()

    # OpenAI
    if provider == "openai":
        try:
            from haystack.components.generators.chat import OpenAIChatGenerator
        except ImportError:
            raise ImportError("Missing OpenAIChatGenerator. Update haystack-ai.")

        return OpenAIChatGenerator(
            model=model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            streaming_callback=streaming_callback,
            tools=tools,
        )

    # Gemini (Google)
    if provider == "gemini":
        try:
            from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
        except ImportError:
            raise ImportError("Missing GoogleGenAIChatGenerator. Install google-generativeai.")

        return GoogleGenAIChatGenerator(
            model=model,
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            tools=tools,
            streaming_callback=streaming_callback,
        )

    raise ValueError(f"Provider '{provider}' not supported. Use: openai or gemini.")