import os
import asyncio

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
# import dotenv
import httpx

# Load environment variables from .env file
# dotenv.load_dotenv()

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# URL of the Pydantic AI documentation
PYDANTIC_DOCS_URL = "https://ai.pydantic.dev/llms.txt"

def fetch_pydantic_docs() -> str:
    """Fetch the Pydantic AI documentation from the URL.
    
    Returns:
        The content of the documentation
    """
    try:
        response = httpx.get(PYDANTIC_DOCS_URL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching Pydantic AI documentation: {e}")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
             func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
             )            
    ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance and insert Pydantic documentation
    rag = asyncio.run(initialize_rag())
    rag.insert(fetch_pydantic_docs())

if __name__ == "__main__":
    main()