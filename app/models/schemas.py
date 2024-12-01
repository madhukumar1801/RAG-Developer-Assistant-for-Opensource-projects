from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from fastapi import Body

class QueryRequest(BaseModel):
    query: str = Field(
        ...,  # ... means required
        description="The query string to search for relevant code",
        min_length=1,
        max_length=1000,
        example="How does the chunking work?"
    )
    query_embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional pre-computed query embedding vector",
        example=None
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How does the chunking work?",
                    "query_embedding": None
                }
            ]
        }
    }

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the LLM")
    source_files: List[str] = Field(..., description="List of source files used for context")
    model: str = Field(..., description="The LLM model used for generation")