from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import query
from app.core.indexing import setup_indexing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Code Assistant API",
    description="RAG-based Code Assistant for Gerrit Repositories",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    await setup_indexing()

app.include_router(query.router, prefix="/api/v1", tags=["query"])

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}