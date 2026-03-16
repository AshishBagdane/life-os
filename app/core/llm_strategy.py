import os
from google.adk.models.lite_llm import LiteLlm


class LLMStrategyProvider:
    """
    Resolves the correct LLM model strategy based on the active environment.
    """

    @staticmethod
    def get_model():
        env = os.getenv("ENVIRONMENT", "PROD").upper()

        if env == "DEV":
            # Strategy: Local Ollama (Free, Private, fast iteration)
            print("INIT: Injecting Local Ollama Strategy (DEV Mode)")
            return LiteLlm(model="ollama_chat/gemma3:1b")

        elif env == "PROD":
            # Strategy: Google Gemini (High-fidelity, Production)
            print("INIT: Injecting Gemini Cloud Strategy (PROD Mode)")
            return "gemini-2.5-flash"

        else:
            # Fallback for safety
            return "gemini-2.5-flash"