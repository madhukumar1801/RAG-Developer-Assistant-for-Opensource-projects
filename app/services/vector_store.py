import chromadb
from chromadb.config import Settings
from app.config import settings
import logging
import uuid


logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTORDB_COLLECTION
        )

    async def add_documents(self, documents: list, embeddings: list, metadatas: list):
        try:
            # Generate unique IDs using UUIDs
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    async def query(self, query_embedding: list, n_results: int = 5):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise