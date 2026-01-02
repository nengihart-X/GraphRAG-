import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid

from .document_processor import DocumentProcessor
from .chunker import AdaptiveChunker
from ..models.schemas import DocumentChunk
from config.settings import settings

logger = structlog.get_logger()

class VectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to vector store"""
        if not chunks:
            return []
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in chunks:
            chunk_id = chunk.get('id', str(uuid.uuid4()))
            ids.append(chunk_id)
            documents.append(chunk['content'])
            
            # Prepare metadata (flatten for ChromaDB)
            metadata = {
                'source': chunk.get('source', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'char_count': chunk.get('metadata', {}).get('char_count', 0),
                'word_count': chunk.get('metadata', {}).get('word_count', 0),
                'chunk_strategy': chunk.get('metadata', {}).get('chunk_strategy', 'unknown')
            }
            
            # Add any additional metadata
            for key, value in chunk.get('metadata', {}).items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            metadatas.append(metadata)
            
            # Add embedding if available
            if chunk.get('embedding'):
                embeddings.append(chunk['embedding'])
            else:
                embeddings.append(None)
        
        # Filter out chunks without embeddings for now
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        
        if not valid_indices:
            logger.warning("No chunks with embeddings found")
            return []
        
        # Add to ChromaDB
        valid_ids = [ids[i] for i in valid_indices]
        valid_documents = [documents[i] for i in valid_indices]
        valid_metadatas = [metadatas[i] for i in valid_indices]
        valid_embeddings = [embeddings[i] for i in valid_indices]
        
        try:
            self.collection.add(
                ids=valid_ids,
                documents=valid_documents,
                metadatas=valid_metadatas,
                embeddings=valid_embeddings
            )
            
            logger.info(f"Added {len(valid_ids)} chunks to vector store")
            return valid_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    async def search(self, query_embedding: List[float], 
                    top_k: int = 10,
                    where: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunks = []
            for i in range(len(results['ids'][0])):
                chunk = DocumentChunk(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    embedding=None,  # Not returned by default
                    score=1 - results['distances'][0][i],  # Convert distance to similarity
                    source=results['metadatas'][0][i].get('source', ''),
                    chunk_index=results['metadatas'][0][i].get('chunk_index', 0)
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': settings.chroma_collection_name,
                'persist_directory': settings.chroma_persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

class DocumentIngestionPipeline:
    """Complete document ingestion pipeline"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = AdaptiveChunker()
        self.vector_store = VectorStore()
    
    async def ingest_document(self, file_path: str, 
                           chunking_strategy: str = 'adaptive') -> Dict[str, Any]:
        """Ingest a single document"""
        try:
            logger.info(f"Starting ingestion for {file_path}")
            
            # Extract text and metadata
            extracted = self.processor.extract_text(file_path)
            
            # Chunk the document
            chunks = self.chunker.chunk_document(
                extracted['full_text'], 
                extracted['metadata'],
                strategy=chunking_strategy
            )
            
            # Generate embeddings
            chunks = self.chunker.generate_embeddings(chunks)
            
            # Add to vector store
            chunk_ids = await self.vector_store.add_documents(chunks)
            
            result = {
                'status': 'success',
                'file_path': file_path,
                'chunks_created': len(chunks),
                'chunks_stored': len(chunk_ids),
                'metadata': extracted['metadata'],
                'chunking_strategy': chunking_strategy
            }
            
            logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }
    
    async def ingest_directory(self, directory_path: str, 
                             chunking_strategy: str = 'adaptive') -> List[Dict[str, Any]]:
        """Ingest all supported documents in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        results = []
        supported_files = []
        
        # Find all supported files
        for ext in settings.supported_formats:
            supported_files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} files to ingest")
        
        # Process files concurrently (with some limit to avoid overwhelming)
        semaphore = asyncio.Semaphore(5)  # Process max 5 files concurrently
        
        async def process_file(file_path):
            async with semaphore:
                return await self.ingest_document(str(file_path), chunking_strategy)
        
        tasks = [process_file(file_path) for file_path in supported_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing file: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Successfully ingested {len(successful_results)} out of {len(supported_files)} files")
        return successful_results
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingestion pipeline"""
        stats = self.vector_store.get_collection_stats()
        return {
            'vector_store': stats,
            'chunking_config': {
                'chunk_size': self.chunker.chunk_size,
                'overlap': self.chunker.overlap,
                'min_chunk_size': self.chunker.min_chunk_size,
                'max_chunk_size': self.chunker.max_chunk_size
            }
        }
