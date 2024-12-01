from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from functools import lru_cache

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore()

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()