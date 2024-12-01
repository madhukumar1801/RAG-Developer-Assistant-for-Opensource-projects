import asyncio
import datetime
import tempfile
import tarfile
import zipfile
import os
from typing import List, Dict, Set
import logging
from sentence_transformers import SentenceTransformer
from app.core.gerrit import GerritClient
from app.core.chunking import CodeChunker
from app.services.vector_store import VectorStore
from app.config import settings
import glob
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class RepositoryIndexer:
    def __init__(self):
        self.gerrit_client = GerritClient()
        self.chunker = CodeChunker()
        self.vector_store = VectorStore()
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.indexed_files: Set[str] = set()
        
    async def _process_file(self, file_path: str, repo_name: str) -> List[Dict]:
        """Process a single file and return chunks with embeddings."""
        try:
            # Use regular open for binary files first to detect encoding
            with open(file_path, 'rb') as f:
                content = f.read()
                
            try:
                # Try to decode as UTF-8 first
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Fallback to latin-1
                    text_content = content.decode('latin-1')
                except UnicodeDecodeError:
                    logger.warning(f"Unable to decode file {file_path}, skipping")
                    return []

            # Only process certain file types
            if not file_path.endswith((
                '.py', '.java', '.cpp', '.js', '.ts', '.tsx', '.jsx', '.css', '.html', '.yaml', '.json',
                '.xml', '.log', '.txt', '.config', '.env', '.md'
            )):
                return []

            # Chunk the content
            chunks = self.chunker.chunk_code(text_content, file_path)
            
            # Generate embeddings for chunks
            embeddings = []
            documents = []
            metadatas = []

            # Generate file hash for tracking changes
            file_hash = hashlib.sha256(content).hexdigest()
            file_id = f"{repo_name}:{file_path}:{file_hash}"

            if file_id in self.indexed_files:
                logger.debug(f"Skipping already indexed file: {file_path}")
                return []
            
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk['content'], convert_to_tensor=False)
                embeddings.append(embedding.tolist())
                documents.append(chunk['content'])
                
                # Enhanced metadata
                metadata = chunk['metadata']
                metadata.update({
                    'repo_name': repo_name,
                    'file_id': file_id,
                    'file_hash': file_hash,
                    'file_type': Path(file_path).suffix[1:],
                    'indexed_at': datetime.datetime.now().isoformat()
                })
                metadatas.append(metadata)

            self.indexed_files.add(file_id)
            
            return [{
                'embeddings': embeddings,
                'documents': [chunk['content'] for chunk in chunks],
                'metadatas': metadatas
            }]
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    async def _extract_archive(self, archive_path: str, extract_dir: str) -> None:
        """Extracts a tar.gz or zip archive."""
        try:
            if archive_path.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.startswith("/") or ".." in member.name:
                            logger.warning(f"Suspicious path in tar: {member.name}, skipping")
                            continue
                    tar.extractall(path=extract_dir)
                logger.info(f"Extracted tar.gz archive: {archive_path}")

            elif archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    for file in zip_ref.namelist():
                        if file.startswith("/") or ".." in file:
                            logger.warning(f"Suspicious path in zip: {file}, skipping")
                            continue
                    zip_ref.extractall(path=extract_dir)
                logger.info(f"Extracted zip archive: {archive_path}")

            else:
                raise ValueError(f"Unsupported archive format: {archive_path}")

        except (tarfile.TarError, zipfile.BadZipFile) as e:
            logger.error(f"Error extracting archive {archive_path}: {e}")
            raise

    async def _process_repository(self, repo_name: str) -> None:
        """Process a Gerrit repository."""
        try:
            logger.info(f"Processing Gerrit repository: {repo_name}")

            # Fetch repository content
            tar_content = await self.gerrit_client.get_repository_content(repo_name)
            if not tar_content:
                logger.error(f"Empty tar content received for {repo_name}, skipping")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = os.path.join(temp_dir, "repo.tar.gz")
                with open(archive_path, "wb") as f:
                    f.write(tar_content)

                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)

                # Extract the tar.gz file
                await self._extract_archive(archive_path, extract_dir)

                # Process files
                tasks = []
                for file_path in glob.glob(os.path.join(extract_dir, "**/*"), recursive=True):
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, extract_dir)
                        logger.debug(f"Processing file: {rel_path}")
                        tasks.append(self._process_file(file_path, repo_name))

                results = await asyncio.gather(*tasks)

                # Index files in vector store
                for result in results:
                    if result:
                        for batch in result:
                            await self.vector_store.add_documents(
                                documents=batch["documents"],
                                embeddings=batch["embeddings"],
                                metadatas=batch["metadatas"],
                            )

        except Exception as e:
            logger.error(f"Error processing Gerrit repository {repo_name}: {e}")
            raise

    async def _process_github_repository(self, repo_url: str) -> None:
        """Process a GitHub repository."""
        try:
            logger.info(f"Processing GitHub repository: {repo_url}")

            # Fetch repository content
            zip_content = await self.gerrit_client.download_github_repository(repo_url)
            if not zip_content:
                logger.error(f"Empty zip content received for {repo_url}, skipping")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = os.path.join(temp_dir, "repo.zip")
                with open(archive_path, "wb") as f:
                    f.write(zip_content)

                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)

                # Extract the zip file
                await self._extract_archive(archive_path, extract_dir)

                # Process files
                tasks = []
                for file_path in glob.glob(os.path.join(extract_dir, "**/*"), recursive=True):
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, extract_dir)
                        logger.debug(f"Processing file: {rel_path}")
                        tasks.append(self._process_file(file_path, repo_url))

                results = await asyncio.gather(*tasks)

                # Index files in vector store
                for result in results:
                    if result:
                        for batch in result:
                            await self.vector_store.add_documents(
                                documents=batch["documents"],
                                embeddings=batch["embeddings"],
                                metadatas=batch["metadatas"],
                            )

        except Exception as e:
            logger.error(f"Error processing GitHub repository {repo_url}: {e}")
            raise


    async def index_repositories(self) -> None:
        """Index all repositories from Gerrit and GitHub."""
        try:
            # Process Gerrit repositories
            gerrit_repos = await self.gerrit_client.get_repositories()
            # if not gerrit_repos:
            #     logger.error("No Gerrit repositories found, skipping Gerrit indexing")
            # else:
            semaphore = asyncio.Semaphore(5)
            async def process_gerrit_repo(repo_name):
                async with semaphore:
                    await self._process_repository(repo_name)

            gerrit_tasks = [process_gerrit_repo(repo) for repo in gerrit_repos]
            
            # Process GitHub repositories
            github_tasks = [self._process_github_repository(repo_url) for repo_url in settings.GITHUB_REPOS]

            # Run all tasks concurrently
            await asyncio.gather(*github_tasks)
            # await asyncio.gather(*gerrit_tasks)

        except Exception as e:
            logger.error(f"Error indexing repositories: {e}")
            raise

class IndexingManager:
    def __init__(self):
        self.indexer = RepositoryIndexer()
        self._indexing_task = None
        self._last_indexed = None
    
    async def start_indexing(self):
        """Start the indexing process."""
        if self._indexing_task and not self._indexing_task.done():
            logger.warning("Indexing is already in progress")
            return
        
        logger.info("Indexing is in progress")
        self._indexing_task = asyncio.create_task(self._run_indexing())
    
    async def _run_indexing(self):
        """Run the indexing process periodically."""
        while True:
            try:
                logger.info("Starting repository indexing")
                await self.indexer.index_repositories()
                self._last_indexed = datetime.datetime.now()
                logger.info("Repository indexing completed")
                
                # Wait for 6 hours before next indexing
                await asyncio.sleep(6 * 60 * 60)
                
            except Exception as e:
                logger.error(f"Error during indexing: {e}")
                # Wait for 15 minutes before retry on error
                await asyncio.sleep(15 * 60)

# Initialize the indexing manager
indexing_manager = IndexingManager()

async def setup_indexing():
    """Setup function to be called during application startup."""
    await indexing_manager.start_indexing()