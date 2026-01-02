from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

class DocumentChunk(TypedDict):
    """Represents a chunk of document content"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    score: Optional[float]
    source: str
    chunk_index: int

class RAGState(TypedDict):
    """State object for LangGraph RAG workflow"""
    # Query information
    query: str
    rewritten_query: str
    query_intent: str  # 'qa', 'summarization', 'comparison'
    
    # Retrieval results
    retrieved_docs: List[DocumentChunk]
    ranked_docs: List[DocumentChunk]
    
    # Answer generation
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    
    # Control flow
    retrieval_attempts: int
    validation_passed: bool
    error_message: Optional[str]
    
    # Metadata
    timestamp: datetime
    session_id: str

class QueryAnalysis(TypedDict):
    """Result of query analysis"""
    intent: str
    keywords: List[str]
    entities: List[str]
    rewritten_query: str
    search_strategy: str  # 'vector', 'hybrid', 'keyword'

class RetrievalResult(TypedDict):
    """Result from retrieval step"""
    docs: List[DocumentChunk]
    total_results: int
    search_time: float
    avg_score: float

class ValidationResult(TypedDict):
    """Result from answer validation"""
    is_grounded: bool
    hallucination_score: float
    confidence_score: float
    issues: List[str]
    needs_regeneration: bool
