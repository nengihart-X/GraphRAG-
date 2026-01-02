from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    stream: bool = Field(False, description="Whether to stream the response")

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    validation_passed: bool
    validation_result: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    session_id: str
    timestamp: datetime
    workflow_steps: Dict[str, Any]

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    chunking_strategy: str = Field("adaptive", description="Chunking strategy: 'fixed', 'semantic', or 'adaptive'")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    status: str
    message: str
    file_info: Dict[str, Any]
    ingestion_result: Dict[str, Any]
    timestamp: datetime

class SourcesResponse(BaseModel):
    """Response model for sources endpoint"""
    sources: List[Dict[str, Any]]
    total_count: int
    collection_stats: Dict[str, Any]

class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint"""
    session_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    issues: List[str] = Field(default_factory=list, description="List of issues with the response")

class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint"""
    status: str
    message: str
    feedback_id: str
    timestamp: datetime

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
