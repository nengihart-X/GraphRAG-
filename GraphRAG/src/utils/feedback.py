from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import structlog
from pathlib import Path

from ..api.models import FeedbackRequest

logger = structlog.get_logger()

class FeedbackManager:
    """Manages user feedback storage and analysis"""
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    async def store_feedback(self, feedback: FeedbackRequest) -> str:
        """Store feedback data"""
        try:
            feedback_id = f"{feedback.session_id}_{datetime.now().timestamp()}"
            
            feedback_data = {
                "feedback_id": feedback_id,
                "session_id": feedback.session_id,
                "query": feedback.query,
                "answer": feedback.answer,
                "rating": feedback.rating,
                "feedback_text": feedback.feedback_text,
                "issues": feedback.issues,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store feedback in JSON file
            feedback_file = self.feedback_dir / f"{feedback_id}.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Feedback stored: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            raise
    
    async def get_session_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a session"""
        try:
            feedback_files = list(self.feedback_dir.glob(f"{session_id}_*.json"))
            feedback_data = []
            
            for file_path in feedback_files:
                with open(file_path, 'r') as f:
                    feedback = json.load(f)
                    feedback_data.append(feedback)
            
            # Sort by timestamp
            feedback_data.sort(key=lambda x: x.get('timestamp', ''))
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error getting session feedback: {e}")
            return []
    
    async def get_all_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback across all sessions"""
        try:
            feedback_files = sorted(
                self.feedback_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]
            
            feedback_data = []
            for file_path in feedback_files:
                with open(file_path, 'r') as f:
                    feedback = json.load(f)
                    feedback_data.append(feedback)
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error getting all feedback: {e}")
            return []
    
    async def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        try:
            all_feedback = await self.get_all_feedback(1000)  # Analyze last 1000 feedbacks
            
            if not all_feedback:
                return {"message": "No feedback data available"}
            
            # Calculate metrics
            ratings = [f["rating"] for f in all_feedback]
            avg_rating = sum(ratings) / len(ratings)
            
            # Rating distribution
            rating_dist = {i: 0 for i in range(1, 6)}
            for rating in ratings:
                rating_dist[rating] += 1
            
            # Common issues
            all_issues = []
            for feedback in all_feedback:
                all_issues.extend(feedback.get("issues", []))
            
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # Sort issues by frequency
            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            analysis = {
                "total_feedback": len(all_feedback),
                "average_rating": round(avg_rating, 2),
                "rating_distribution": rating_dist,
                "top_issues": top_issues,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return {"error": str(e)}
    
    async def export_feedback(self, format: str = "json") -> str:
        """Export feedback data"""
        try:
            all_feedback = await self.get_all_feedback()
            
            if format.lower() == "json":
                export_data = {
                    "feedback": all_feedback,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_count": len(all_feedback)
                }
                
                export_file = self.feedback_dir / f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                return str(export_file)
            
            elif format.lower() == "csv":
                import csv
                
                export_file = self.feedback_dir / f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                with open(export_file, 'w', newline='') as csvfile:
                    if all_feedback:
                        fieldnames = all_feedback[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_feedback)
                
                return str(export_file)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            raise
