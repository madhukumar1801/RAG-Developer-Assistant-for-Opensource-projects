from typing import Dict, List, Optional
import logging
from sentence_transformers import SentenceTransformer
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from app.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for input text."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"File: {chunk['metadata']['file_path']}\n"
                f"Repository: {chunk['metadata']['repo_name']}\n"
                f"Content:\n{chunk['content']}\n"
            )
        return "\n---\n".join(context_parts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _retrieve_relevant_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve relevant code chunks from vector store."""
        try:
            results = await self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            chunks = []
            for i in range(len(results['documents'][0])):
                chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    async def process_query(
        self,
        query: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of relevant chunks to retrieve
            system_prompt: Optional system prompt to guide LLM
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            
            # Retrieve relevant chunks
            relevant_chunks = await self._retrieve_relevant_chunks(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Prepare context from chunks
            context = self._prepare_context(relevant_chunks)
            
            # Generate response using LLM
            response = await self.llm_service.generate_response(
                query=query,
                context=context,
                system_prompt=system_prompt
            )
            
            # Prepare source information
            sources = [
                {
                    'file_path': chunk['metadata']['file_path'],
                    'repo_name': chunk['metadata']['repo_name'],
                    'file_type': chunk['metadata']['file_type']
                }
                for chunk in relevant_chunks
            ]
            
            return {
                'answer': response['response'],
                'sources': sources,
                'model': response['model'],
                'metadata': {
                    'chunks_retrieved': len(relevant_chunks),
                    'prompt_tokens': response.get('prompt_tokens'),
                    'completion_tokens': response.get('completion_tokens')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise