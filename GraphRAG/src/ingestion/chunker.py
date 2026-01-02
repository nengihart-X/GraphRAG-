import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog
from sentence_transformers import SentenceTransformer
import numpy as np

from ..models.schemas import DocumentChunk
from config.settings import settings

logger = structlog.get_logger()

@dataclass
class ChunkingStrategy:
    """Configuration for chunking strategy"""
    chunk_size: int
    overlap: int
    min_chunk_size: int
    max_chunk_size: int
    strategy: str  # 'fixed', 'semantic', 'adaptive'

class AdaptiveChunker:
    """Adaptive document chunking with multiple strategies"""
    
    def __init__(self, embedding_model: Optional[str] = None):
        self.chunk_size = settings.chunk_size
        self.overlap = settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size
        self.max_chunk_size = settings.max_chunk_size
        
        # Initialize embedding model for semantic chunking
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = SentenceTransformer(settings.embedding_model)
    
    def chunk_document(self, text: str, metadata: Dict[str, Any], 
                      strategy: str = 'adaptive') -> List[DocumentChunk]:
        """Chunk document using specified strategy"""
        
        if strategy == 'fixed':
            return self._fixed_size_chunking(text, metadata)
        elif strategy == 'semantic':
            return self._semantic_chunking(text, metadata)
        elif strategy == 'adaptive':
            return self._adaptive_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_size_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap"""
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            chunk = DocumentChunk(
                id=f"{metadata.get('filename', 'doc')}_chunk_{i}",
                content=chunk_text,
                metadata={**metadata, 'chunk_strategy': 'fixed', 'chunk_index': i},
                embedding=None,
                score=None,
                source=metadata.get('source', ''),
                chunk_index=i // (self.chunk_size - self.overlap)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Semantic chunking based on sentence embeddings"""
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Generate embeddings for sentences
        sentence_embeddings = self.embedding_model.encode(sentences)
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_text = ""
        chunk_index = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            # Check if adding this sentence would exceed max chunk size
            if len(current_chunk_text + sentence) > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                if len(current_chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk_from_sentences(
                        current_chunk_sentences, current_chunk_text, 
                        metadata, chunk_index, 'semantic'
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_text = sentence
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_text += sentence
        
        # Handle remaining sentences
        if current_chunk_sentences and len(current_chunk_text) >= self.min_chunk_size:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, current_chunk_text, 
                metadata, chunk_index, 'semantic'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _adaptive_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Adaptive chunking that combines multiple strategies"""
        # Analyze text characteristics
        text_stats = self._analyze_text(text)
        
        # Choose best strategy based on text characteristics
        if text_stats['has_clear_structure'] and text_stats['avg_sentence_length'] < 100:
            # Use semantic chunking for well-structured text
            chunks = self._semantic_chunking(text, metadata)
        else:
            # Use fixed-size chunking for less structured text
            chunks = self._fixed_size_chunking(text, metadata)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks, metadata)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics for adaptive chunking"""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return {'has_clear_structure': False, 'avg_sentence_length': 0}
        
        avg_sentence_length = np.mean([len(s) for s in sentences])
        
        # Check for structural indicators
        has_clear_structure = (
            len(re.findall(r'\n\n', text)) > 0 or  # Multiple paragraphs
            len(re.findall(r'\d+\.', text)) > 0 or  # Numbered lists
            len(re.findall(r'[A-Z][a-z]+:', text)) > 0  # Headers
        )
        
        return {
            'has_clear_structure': has_clear_structure,
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': len(sentences),
            'text_length': len(text)
        }
    
    def _create_chunk_from_sentences(self, sentences: List[str], chunk_text: str,
                                   metadata: Dict[str, Any], chunk_index: int,
                                   strategy: str) -> DocumentChunk:
        """Create a DocumentChunk from sentences"""
        filename = metadata.get('filename', 'doc')
        
        chunk = DocumentChunk(
            id=f"{filename}_chunk_{chunk_index}",
            content=chunk_text.strip(),
            metadata={
                **metadata,
                'chunk_strategy': strategy,
                'chunk_index': chunk_index,
                'sentence_count': len(sentences),
                'char_count': len(chunk_text)
            },
            embedding=None,
            score=None,
            source=metadata.get('source', ''),
            chunk_index=chunk_index
        )
        
        return chunk
    
    def _post_process_chunks(self, chunks: List[DocumentChunk], 
                           metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Post-process chunks to ensure quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Remove chunks that are too short or contain mostly whitespace
            content_stripped = chunk['content'].strip()
            
            if (len(content_stripped) < self.min_chunk_size or 
                len(content_stripped.split()) < 5):
                continue
            
            # Update chunk content
            chunk['content'] = content_stripped
            chunk['metadata']['char_count'] = len(content_stripped)
            chunk['metadata']['word_count'] = len(content_stripped.split())
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks"""
        if not chunks:
            return chunks
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
