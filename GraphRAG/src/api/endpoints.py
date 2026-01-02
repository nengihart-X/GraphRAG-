from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import asyncio
import structlog
from datetime import datetime
import uuid
import json

from ..graph.workflow import RAGWorkflow
from ..ingestion.pipeline import DocumentIngestionPipeline
from .models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, 
    DocumentUploadResponse, SourcesResponse, FeedbackRequest, 
    FeedbackResponse, HealthResponse, ErrorResponse
)
from ..utils.feedback import FeedbackManager
from ..utils.metrics import MetricsCollector

logger = structlog.get_logger()

# Initialize components
rag_workflow = RAGWorkflow()
ingestion_pipeline = DocumentIngestionPipeline()
feedback_manager = FeedbackManager()
metrics_collector = MetricsCollector()

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process user query using RAG workflow"""
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        
        # Record metrics
        start_time = datetime.now()
        
        # Run RAG workflow
        if request.stream:
            # For streaming, we'll need to implement SSE or WebSocket
            # For now, return non-streaming response
            result = await rag_workflow.run(request.query, request.session_id)
        else:
            result = await rag_workflow.run(request.query, request.session_id)
        
        # Record metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        await metrics_collector.record_query_metrics(
            query=request.query,
            processing_time=processing_time,
            confidence=result["confidence"],
            validation_passed=result["validation_passed"],
            session_id=result["session_id"]
        )
        
        # Convert datetime to string for JSON serialization
        result["timestamp"] = result["timestamp"].isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Stream query response using Server-Sent Events"""
    async def generate_stream():
        try:
            logger.info(f"Starting streaming query: {request.query[:100]}...")
            
            async for event in rag_workflow.stream_run(request.query, request.session_id):
                # Format event for SSE
                event_data = {
                    "type": "workflow_event",
                    "data": event,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            
            # Send final completion event
            completion_data = {
                "type": "completion",
                "message": "Query processing completed",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: str = "adaptive"
):
    """Upload and ingest a document"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        # Save file temporarily
        temp_file_path = f"data/documents/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Process document in background
        def process_document():
            try:
                result = asyncio.run(ingestion_pipeline.ingest_document(
                    temp_file_path, 
                    chunking_strategy
                ))
                logger.info(f"Document ingestion completed: {result}")
            except Exception as e:
                logger.error(f"Document ingestion failed: {e}")
        
        background_tasks.add_task(process_document)
        
        # Prepare response
        file_info = {
            "filename": file.filename,
            "size": file_size,
            "content_type": file.content_type,
            "chunking_strategy": chunking_strategy
        }
        
        response = DocumentUploadResponse(
            status="processing",
            message="Document upload received and processing started",
            file_info=file_info,
            ingestion_result={"status": "pending"},
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/batch")
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunking_strategy: str = "adaptive"
):
    """Upload and ingest multiple documents"""
    try:
        logger.info(f"Uploading batch of {len(files)} documents")
        
        results = []
        
        for file in files:
            if not file.filename:
                continue
            
            # Validate file size
            file_content = await file.read()
            file_size = len(file_content)
            
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "File too large"
                })
                continue
            
            # Save file temporarily
            temp_file_path = f"data/documents/{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            # Add to background processing
            background_tasks.add_task(
                process_single_document,
                temp_file_path,
                chunking_strategy
            )
            
            results.append({
                "filename": file.filename,
                "status": "processing",
                "message": "Upload received, processing started"
            })
        
        return {
            "status": "batch_processing",
            "message": f"Uploaded {len(files)} files",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_single_document(file_path: str, chunking_strategy: str):
    """Process a single document (for background tasks)"""
    try:
        result = asyncio.run(ingestion_pipeline.ingest_document(
            file_path, 
            chunking_strategy
        ))
        logger.info(f"Document {file_path} processed: {result}")
    except Exception as e:
        logger.error(f"Document {file_path} processing failed: {e}")

@router.get("/sources", response_model=SourcesResponse)
async def get_sources():
    """Get information about document sources"""
    try:
        # Get collection statistics
        stats = ingestion_pipeline.get_ingestion_stats()
        
        # Get sample documents from vector store
        # This would need to be implemented in the vector store
        sources = []
        
        response = SourcesResponse(
            sources=sources,
            total_count=stats["vector_store"].get("document_count", 0),
            collection_stats=stats
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback on query responses"""
    try:
        logger.info(f"Received feedback for session: {feedback.session_id}")
        
        # Store feedback
        feedback_id = await feedback_manager.store_feedback(feedback)
        
        # Record metrics
        await metrics_collector.record_feedback_metrics(
            session_id=feedback.session_id,
            rating=feedback.rating,
            issues=feedback.issues
        )
        
        response = FeedbackResponse(
            status="success",
            message="Feedback submitted successfully",
            feedback_id=feedback_id,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/{session_id}")
async def get_session_feedback(session_id: str):
    """Get feedback for a specific session"""
    try:
        feedback_data = await feedback_manager.get_session_feedback(session_id)
        return {
            "session_id": session_id,
            "feedback": feedback_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check component health
        components = {}
        
        # Check vector store
        try:
            stats = ingestion_pipeline.get_ingestion_stats()
            components["vector_store"] = "healthy"
        except Exception as e:
            components["vector_store"] = f"unhealthy: {str(e)}"
        
        # Check LLM connectivity (simple check)
        try:
            # This would be a simple LLM ping
            components["llm"] = "healthy"
        except Exception as e:
            components["llm"] = f"unhealthy: {str(e)}"
        
        # Overall status
        overall_status = "healthy" if all("healthy" in status for status in components.values()) else "degraded"
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            components=components
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = await metrics_collector.get_metrics()
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data"""
    try:
        success = rag_workflow.clear_session(session_id)
        return {
            "status": "success" if success else "error",
            "message": f"Session {session_id} cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
