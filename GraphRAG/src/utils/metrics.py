from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import structlog
from pathlib import Path
from collections import defaultdict, deque
import asyncio

logger = structlog.get_logger()

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics for real-time access
        self.query_metrics = deque(maxlen=1000)  # Last 1000 queries
        self.feedback_metrics = deque(maxlen=1000)  # Last 1000 feedbacks
        self.performance_metrics = defaultdict(list)
        
        # Aggregated metrics
        self.daily_stats = defaultdict(lambda: {
            "queries": 0,
            "avg_confidence": 0,
            "avg_rating": 0,
            "validation_pass_rate": 0
        })
    
    async def record_query_metrics(self, query: str, processing_time: float, 
                                 confidence: float, validation_passed: bool, 
                                 session_id: str):
        """Record metrics for a query"""
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
                "processing_time": processing_time,
                "confidence": confidence,
                "validation_passed": validation_passed,
                "session_id": session_id
            }
            
            self.query_metrics.append(metric)
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_stats[today]["queries"] += 1
            
            # Update performance metrics
            self.performance_metrics["processing_times"].append(processing_time)
            self.performance_metrics["confidences"].append(confidence)
            
            # Keep only last 1000 performance metrics
            if len(self.performance_metrics["processing_times"]) > 1000:
                self.performance_metrics["processing_times"].pop(0)
                self.performance_metrics["confidences"].pop(0)
            
            # Periodically save to file
            if len(self.query_metrics) % 100 == 0:
                await self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error recording query metrics: {e}")
    
    async def record_feedback_metrics(self, session_id: str, rating: int, issues: List[str]):
        """Record metrics for feedback"""
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "rating": rating,
                "issue_count": len(issues),
                "issues": issues
            }
            
            self.feedback_metrics.append(metric)
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            current_avg = self.daily_stats[today]["avg_rating"]
            current_count = self.daily_stats[today]["queries"]
            
            if current_count > 0:
                new_avg = (current_avg * current_count + rating) / (current_count + 1)
                self.daily_stats[today]["avg_rating"] = new_avg
            
            # Periodically save to file
            if len(self.feedback_metrics) % 50 == 0:
                await self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error recording feedback metrics: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        try:
            # Recent metrics (last hour)
            recent_metrics = self._get_recent_metrics(hours=1)
            
            # Performance metrics
            performance = self._calculate_performance_metrics()
            
            # Daily stats
            daily_stats = dict(self.daily_stats)
            
            # System health
            health = self._calculate_health_metrics()
            
            return {
                "recent": recent_metrics,
                "performance": performance,
                "daily_stats": daily_stats,
                "health": health,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def _get_recent_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_queries = [
            m for m in self.query_metrics 
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        recent_feedback = [
            m for m in self.feedback_metrics 
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if not recent_queries:
            return {"queries": 0, "avg_confidence": 0, "avg_processing_time": 0}
        
        avg_confidence = sum(q["confidence"] for q in recent_queries) / len(recent_queries)
        avg_processing_time = sum(q["processing_time"] for q in recent_queries) / len(recent_queries)
        validation_pass_rate = sum(1 for q in recent_queries if q["validation_passed"]) / len(recent_queries)
        
        return {
            "queries": len(recent_queries),
            "avg_confidence": round(avg_confidence, 3),
            "avg_processing_time": round(avg_processing_time, 3),
            "validation_pass_rate": round(validation_pass_rate, 3),
            "feedback_count": len(recent_feedback)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        processing_times = self.performance_metrics["processing_times"]
        confidences = self.performance_metrics["confidences"]
        
        if not processing_times:
            return {
                "avg_processing_time": 0,
                "p95_processing_time": 0,
                "avg_confidence": 0
            }
        
        # Processing time metrics
        avg_time = sum(processing_times) / len(processing_times)
        sorted_times = sorted(processing_times)
        p95_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        
        # Confidence metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "avg_processing_time": round(avg_time, 3),
            "p95_processing_time": round(p95_time, 3),
            "avg_confidence": round(avg_confidence, 3),
            "total_queries": len(processing_times)
        }
    
    def _calculate_health_metrics(self) -> Dict[str, Any]:
        """Calculate system health metrics"""
        recent_metrics = self._get_recent_metrics(hours=24)  # Last 24 hours
        
        # Health scoring
        health_score = 100
        issues = []
        
        # Check processing time
        if recent_metrics["avg_processing_time"] > 10:  # > 10 seconds
            health_score -= 20
            issues.append("High processing time")
        
        # Check confidence
        if recent_metrics["avg_confidence"] < 0.5:
            health_score -= 15
            issues.append("Low confidence scores")
        
        # Check validation pass rate
        if recent_metrics["validation_pass_rate"] < 0.7:
            health_score -= 25
            issues.append("Low validation pass rate")
        
        # Check query volume
        if recent_metrics["queries"] == 0:
            health_score -= 10
            issues.append("No recent queries")
        
        health_status = "excellent"
        if health_score < 60:
            health_status = "poor"
        elif health_score < 80:
            health_status = "fair"
        elif health_score < 95:
            health_status = "good"
        
        return {
            "health_score": max(0, health_score),
            "health_status": health_status,
            "issues": issues,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_data = {
                "query_metrics": list(self.query_metrics),
                "feedback_metrics": list(self.feedback_metrics),
                "performance_metrics": dict(self.performance_metrics),
                "daily_stats": dict(self.daily_stats),
                "timestamp": datetime.now().isoformat()
            }
            
            metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def load_metrics(self):
        """Load metrics from file"""
        try:
            metrics_files = sorted(self.metrics_dir.glob("metrics_*.json"), reverse=True)
            
            if metrics_files:
                latest_file = metrics_files[0]
                with open(latest_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.query_metrics = deque(metrics_data.get("query_metrics", []), maxlen=1000)
                self.feedback_metrics = deque(metrics_data.get("feedback_metrics", []), maxlen=1000)
                self.performance_metrics = defaultdict(list, metrics_data.get("performance_metrics", {}))
                self.daily_stats = defaultdict(lambda: {
                    "queries": 0,
                    "avg_confidence": 0,
                    "avg_rating": 0,
                    "validation_pass_rate": 0
                }, metrics_data.get("daily_stats", {}))
                
                logger.info(f"Loaded metrics from {latest_file}")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    async def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for file_path in self.metrics_dir.glob("metrics_*.json"):
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_date < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Deleted old metrics file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")
