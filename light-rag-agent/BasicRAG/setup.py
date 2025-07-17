    
"""
Basic setup for PydanticAI examples.
"""

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class OllamaModel:
    LLAMA3_3 = "llama3.3:latest" #ok
    QWEN2_5_14b = "qwen2.5:14b" #ok
    QWEN3_14B = "qwen3:14b" #ok
    LLAMA3 = "llama3:latest"
    MISTRAL = "mistral:7b"
    QWQ_32b = "qwq:32b" #ok
    GEMMA3_27b = "gemma3:27b"
    LLAMA3_1_8B = "llama3.1:8b"
    
    def __init__(self, model_name=QWQ_32b):
        self.model_name = model_name
    
    def get_model(self):
        return OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url='http://localhost:11434/v1')
        )