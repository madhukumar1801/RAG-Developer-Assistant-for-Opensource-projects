from httpx import AsyncClient
from typing import List, Dict
from app.config import settings
import logging
import requests

logger = logging.getLogger(__name__)

class GerritClient:
    def __init__(self):
        self.base_url = settings.GERRIT_URL
        self.auth_token = settings.GERRIT_TOKEN
        self.headers = {
            "Authorization": f"Basic {self.auth_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    async def get_repositories(self) -> List[Dict]:
        """Fetch list of repositories from Gerrit."""
        async with AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/a/projects/",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    async def get_repository_content(self, repo_name: str, ref: str = "HEAD") -> bytes:
        async with AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/a/plugins/gitiles/{repo_name}/+archive/refs/heads/master.tar.gz",
                headers=self.headers
            )
            logger.debug('Successfully fetched')
            response.raise_for_status()
            return response.content
        
    
    async def download_github_repository(self, repo_url: str) -> bytes:
        zip_url = f"{repo_url}/archive/refs/heads/master.zip"
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        return response.content