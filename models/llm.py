"""
models/llm.py

Thin wrapper around the local Ollama LLM.

Supported models :
   llama3           Meta Llama-3 8B  
   mistral          Mistral 7B
   llama3:instruct  chat-tuned variant

"""

import os
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

#  Configuration 
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL",    "llama3")
EMBED_MODEL: str     = os.getenv("EMBED_MODEL",     "nomic-embed-text")


def get_llm(temperature: float = 0.7) -> OllamaLLM:
    """
    Return a configured OllamaLLM instance.

    Args:
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        OllamaLLM ready for inference.
    """
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        # Keep responses focused; large contexts slow generation
        num_ctx=4096,
    )


def get_embeddings() -> OllamaEmbeddings:
    """
    Return OllamaEmbeddings for FAISS indexing & retrieval.

    Falls back to `nomic-embed-text` (fast, high-quality 768-dim model).
    Pull it once with:  ollama pull nomic-embed-text
    """
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
