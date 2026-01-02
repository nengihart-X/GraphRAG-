from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from .endpoints import router
from ..utils.metrics import MetricsCollector
from config.settings import settings

logger = structlog.get_logger()

# Initialize metrics collector
metrics_collector = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Agentic RAG Platform")
    
    # Load existing metrics
    await metrics_collector.load_metrics()
    
    # Start background tasks
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic RAG Platform")
    cleanup_task.cancel()
    
    # Save final metrics
    await metrics_collector._save_metrics()

async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        try:
            await asyncio.sleep(24 * 60 * 60)  # Run daily
            await metrics_collector.cleanup_old_metrics()
            logger.info("Completed periodic cleanup")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Platform",
    description="Production-grade Retrieval-Augmented Generation system using LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        processing_time=processing_time
    )
    
    return response

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agentic RAG Platform",
        "version": "1.0.0",
        "description": "Production-grade RAG system with LangGraph",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "query": "/api/v1/query",
            "query_stream": "/api/v1/query/stream",
            "upload": "/api/v1/upload",
            "sources": "/api/v1/sources",
            "feedback": "/api/v1/feedback",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
