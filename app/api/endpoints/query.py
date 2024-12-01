from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Annotated
from app.models.schemas import QueryRequest, QueryResponse
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from app.dependencies import get_vector_store, get_llm_service
from sentence_transformers import SentenceTransformer
from app.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_code(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service)
):
    try:
        logger.info(f"Received query: {request.query}")
        
        # Generate embedding if not provided
        if request.query_embedding is None:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = model.encode(request.query).tolist()
        else:
            query_embedding = request.query_embedding

        # Get relevant code chunks from vector store
        results = await vector_store.query(
            query_embedding=query_embedding,
            n_results=5
        )

        if not results or not results.get("documents"):
            return QueryResponse(
                answer="No relevant code found for your query.",
                source_files=[],
                model=settings.LLM_MODEL
            )

        # Flatten and process documents
        flat_documents = [item for sublist in results["documents"] for item in sublist]
        context = "\n\n".join(flat_documents)

        # Generate response using LLM
        response = await llm_service.generate_response(
            query=request.query,
            context=context
        )

        return QueryResponse(
            answer=response["response"],
            source_files=[doc["file_path"] for doc in results["metadatas"][0]],
            model=response["model"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))