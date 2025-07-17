"""Pydantic AI agent that leverages RAG with a local ChromaDB for Pydantic documentation using local Ollama."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional
import asyncio
import chromadb

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from setup import OllamaModel
from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context
)


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str

model = OllamaModel(OllamaModel.LLAMA3_1_8B).get_model()

# Use Ollama model (e.g., mistral, llama3, etc.)
_agent_ollama = Agent(
    model=model,
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions about Pydantic AI based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the Pydantic AI documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response."
)

def get_agent_olama():
    return _agent_ollama


@_agent_ollama.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query."""
    print(f'---> get_or_create_collection ')
    
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    print(f'---> after get_or_create_collection : {collection} ')

    query_results = query_collection(collection, search_query, n_results=n_results)
    print(f'---> after query_collection : {collection} ')
    
    return format_results_as_context(query_results)


async def run_rag_agent(
    question: str,
    collection_name: str = "pydantic_docs",
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-minilm:latest",
    n_results: int = 5
) -> str:
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    result = await _agent_ollama.run(question, deps=deps)
    return result.data


def main():
    print('------> rag agent ollama main ')
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB and Ollama")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default="pydantic_docs", help="Name of the ChromaDB collection")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()

    if not args.question:
        print("Error: You must provide a --question argument.")
        sys.exit(1)
    
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=args.collection,
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
