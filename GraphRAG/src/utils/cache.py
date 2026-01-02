import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import structlog
import asyncio
import aioredis
from functools import wraps

logger = structlog.get_logger()

class RedisCache:
    """Redis-based caching for RAG system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis_client = None
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0

# Global cache instance
cache = RedisCache()

def cached(prefix: str, ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_data = {
                "args": args,
                "kwargs": sorted(kwargs.items())
            }
            cache_key = cache._generate_key(prefix, cache_data)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {prefix}")
            
            return result
        
        return wrapper
    return decorator

class QueryCache:
    """Specialized cache for query results"""
    
    def __init__(self, cache_client: RedisCache):
        self.cache = cache_client
        self.prefix = "query"
    
    async def get_query_result(self, query: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        cache_data = {"query": query, "session_id": session_id}
        key = self.cache._generate_key(self.prefix, cache_data)
        return await self.cache.get(key)
    
    async def cache_query_result(self, query: str, result: Dict[str, Any], 
                                session_id: str = None, ttl: int = 1800):
        """Cache query result (30 minutes default)"""
        cache_data = {"query": query, "session_id": session_id}
        key = self.cache._generate_key(self.prefix, cache_data)
        await self.cache.set(key, result, ttl)
    
    async def invalidate_query(self, query: str, session_id: str = None):
        """Invalidate cached query result"""
        cache_data = {"query": query, "session_id": session_id}
        key = self.cache._generate_key(self.prefix, cache_data)
        await self.cache.delete(key)

class EmbeddingCache:
    """Specialized cache for embeddings"""
    
    def __init__(self, cache_client: RedisCache):
        self.cache = cache_client
        self.prefix = "embedding"
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self.cache._generate_key(self.prefix, text)
        result = await self.cache.get(key)
        return result if result else None
    
    async def cache_embedding(self, text: str, embedding: List[float], ttl: int = 86400):
        """Cache embedding (24 hours default)"""
        key = self.cache._generate_key(self.prefix, text)
        await self.cache.set(key, embedding, ttl)

class RetrievalCache:
    """Specialized cache for retrieval results"""
    
    def __init__(self, cache_client: RedisCache):
        self.cache = cache_client
        self.prefix = "retrieval"
    
    async def get_retrieval_result(self, query_embedding: List[float], 
                                 top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval result"""
        cache_data = {
            "query_embedding": query_embedding,
            "top_k": top_k
        }
        key = self.cache._generate_key(self.prefix, cache_data)
        return await self.cache.get(key)
    
    async def cache_retrieval_result(self, query_embedding: List[float], 
                                   docs: List[Dict[str, Any]], top_k: int, 
                                   ttl: int = 1800):
        """Cache retrieval result (30 minutes default)"""
        cache_data = {
            "query_embedding": query_embedding,
            "top_k": top_k
        }
        key = self.cache._generate_key(self.prefix, cache_data)
        await self.cache.set(key, docs, ttl)

# Initialize specialized caches
query_cache = QueryCache(cache)
embedding_cache = EmbeddingCache(cache)
retrieval_cache = RetrievalCache(cache)

async def initialize_cache():
    """Initialize cache connection"""
    await cache.connect()

async def cleanup_cache():
    """Cleanup cache connection"""
    await cache.disconnect()
