from typing import Dict, Optional
import logging
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful code assistant. Using the provided code context and reasoning abilities, 
            answer the user's query. Provide relevant code snippets when applicable.

            Context:
            {context}

            Query:
            {query}

            Answer (include code where applicable):
            """
        )

    def _initialize_llm(self):
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL
        )

    async def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> Dict:
        try:
            prompt = self.prompt_template.format(
                context=context,
                query=query
            )
            
            response = await self.llm.agenerate([prompt])
            
            return {
                "response": response.generations[0][0].text,
                "model": settings.LLM_MODEL
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise