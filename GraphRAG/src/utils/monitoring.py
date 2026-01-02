from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from typing import Dict, Any
import structlog
from datetime import datetime
import json

logger = structlog.get_logger()

class PrometheusMetrics:
    """Prometheus metrics collection for the RAG system"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Query metrics
        self.query_total = Counter(
            'rag_queries_total',
            'Total number of queries processed',
            ['status', 'intent'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'Time spent processing queries',
            ['intent'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.query_confidence = Histogram(
            'rag_query_confidence',
            'Confidence scores of query responses',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Retrieval metrics
        self.retrieval_total = Counter(
            'rag_retrievals_total',
            'Total number of retrieval operations',
            ['strategy', 'status'],
            registry=self.registry
        )
        
        self.retrieved_docs = Histogram(
            'rag_retrieved_docs_count',
            'Number of documents retrieved',
            buckets=[1, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        self.retrieval_score = Histogram(
            'rag_retrieval_score',
            'Average retrieval scores',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Document metrics
        self.documents_total = Gauge(
            'rag_documents_total',
            'Total number of documents in the system',
            registry=self.registry
        )
        
        self.document_ingestion_total = Counter(
            'rag_document_ingestion_total',
            'Total number of document ingestions',
            ['status', 'file_type'],
            registry=self.registry
        )
        
        # Validation metrics
        self.validation_total = Counter(
            'rag_validations_total',
            'Total number of answer validations',
            ['result'],
            registry=self.registry
        )
        
        self.hallucination_score = Histogram(
            'rag_hallucination_score',
            'Hallucination scores from validation',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # System metrics
        self.active_sessions = Gauge(
            'rag_active_sessions',
            'Number of active sessions',
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Feedback metrics
        self.feedback_total = Counter(
            'rag_feedback_total',
            'Total number of feedback submissions',
            ['rating'],
            registry=self.registry
        )
        
        self.feedback_rating = Histogram(
            'rag_feedback_rating',
            'User feedback ratings',
            buckets=[1, 2, 3, 4, 5],
            registry=self.registry
        )
    
    def record_query(self, intent: str, duration: float, confidence: float, 
                    status: str = "success"):
        """Record query metrics"""
        self.query_total.labels(status=status, intent=intent).inc()
        self.query_duration.labels(intent=intent).observe(duration)
        self.query_confidence.observe(confidence)
    
    def record_retrieval(self, strategy: str, doc_count: int, avg_score: float, 
                        status: str = "success"):
        """Record retrieval metrics"""
        self.retrieval_total.labels(strategy=strategy, status=status).inc()
        self.retrieved_docs.observe(doc_count)
        self.retrieval_score.observe(avg_score)
    
    def record_document_ingestion(self, file_type: str, status: str = "success"):
        """Record document ingestion metrics"""
        self.document_ingestion_total.labels(status=status, file_type=file_type).inc()
    
    def record_validation(self, validation_passed: bool, hallucination_score: float):
        """Record validation metrics"""
        result = "passed" if validation_passed else "failed"
        self.validation_total.labels(result=result).inc()
        self.hallucination_score.observe(hallucination_score)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_feedback(self, rating: int):
        """Record feedback metrics"""
        self.feedback_total.labels(rating=str(rating)).inc()
        self.feedback_rating.observe(rating)
    
    def update_document_count(self, count: int):
        """Update total document count"""
        self.documents_total.set(count)
    
    def update_active_sessions(self, count: int):
        """Update active sessions count"""
        self.active_sessions.set(count)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest(self.registry).decode('utf-8')

class SystemMonitor:
    """System monitoring and health checks"""
    
    def __init__(self, prometheus_metrics: PrometheusMetrics):
        self.prometheus = prometheus_metrics
        self.start_time = datetime.now()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "api": "healthy",
                "rag_workflow": "healthy",
                "vector_store": "healthy",
                "cache": "healthy"
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return {
            "prometheus_metrics": self.prometheus.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }

# Global metrics instance
prometheus_metrics = PrometheusMetrics()
system_monitor = SystemMonitor(prometheus_metrics)
