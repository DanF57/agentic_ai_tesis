# client/utils/llm_factory.py
from haystack.utils import Secret


def create_llm(provider: str, model: str, tools: list = None, streaming_callback=None):
    """
    Factory para crear generadores LLM para Agentes Haystack.
    Soporta: OpenAI y Gemini.
    
    Args:
        provider: "openai" o "gemini"
        model: Nombre del modelo específico
        tools: Lista de herramientas disponibles
        
    Returns:
        Chat generator instance
    """
    provider = provider.lower()

    if provider not in ["openai", "gemini"]:
        raise ValueError(f"Provider '{provider}' not supported. Use: openai or gemini.")

    try:
        # OpenAI
        if provider == "openai":
            from haystack.components.generators.chat import OpenAIChatGenerator
            
            return OpenAIChatGenerator(
                model=model,
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                tools=tools,
                streaming_callback=streaming_callback
            )

        # Gemini (Google)
        elif provider == "gemini":
            from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
            
            return GoogleGenAIChatGenerator(
                model=model,
                api_key=Secret.from_env_var("GOOGLE_API_KEY"),
                tools=tools,
                streaming_callback=streaming_callback
            )
    
    except ImportError as e:
        raise ImportError(
            f"Failed to import {provider} generator. "
            f"Install required package: {str(e)}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create {provider} LLM: {str(e)}")