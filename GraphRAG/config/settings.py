import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Database Configuration
    vector_db_type: str = "chroma"  # chroma, faiss, pinecone
    chroma_persist_directory: str = "./data/vectors"
    chroma_collection_name: str = "documents"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval Configuration
    default_top_k: int = 10
    max_retrieval_attempts: int = 3
    retrieval_threshold: float = 0.7
    rerank_top_k: int = 5
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunk_size: int = 1024
    min_chunk_size: int = 100
    
    # Redis Configuration (for caching)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 3600  # 1 hour
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Document Processing
    supported_formats: list = [".pdf", ".docx", ".txt", ".md", ".html"]
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
