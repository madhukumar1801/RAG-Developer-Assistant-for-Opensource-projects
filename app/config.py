from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server Settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    WORKERS: int = 4

    # Gerrit Settings
    GERRIT_URL: str = "https://gerrit-review.com"
    GERRIT_TOKEN: str = "blahblahblah"
    GERRIT_USERNAME: str = 'blah'
    
    # GitHub Opensource Repositories
    GITHUB_REPOS: list[str] = [
    "https://github.com/backstage/backstage",
    # "https://github.com/grafana/k6",
    ]

    # Vector DB Settings
    VECTORDB_HOST: str = '0.0.0.0'
    VECTORDB_PORT: int = 8000
    VECTORDB_COLLECTION: str = 'default'

    # LLM Settings
    LLM_TYPE: str = 'ollama'
    LLM_MODEL: str = 'codellama'
    OLLAMA_BASE_URL: str = 'http://host.docker.internal:11434'

    # Embedding Settings
    EMBEDDING_MODEL: str = 'sentence-transformers/all-mpnet-base-v2'

    class Config:
        env_file = "../.env"

settings = Settings()