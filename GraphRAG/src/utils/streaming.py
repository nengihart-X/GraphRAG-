from typing import AsyncGenerator, Dict, Any, List
import json
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()

class StreamingResponseHandler:
    """Handles streaming responses for real-time RAG workflow updates"""
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
    
    async def create_stream(self, stream_id: str) -> AsyncGenerator[str, None]:
        """Create a new streaming connection"""
        queue = asyncio.Queue()
        self.active_streams[stream_id] = queue
        
        try:
            while True:
                # Wait for data with timeout
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=300)  # 5 minute timeout
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Check for end signal
                    if data.get("type") == "end":
                        break
                        
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
        
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def send_event(self, stream_id: str, event_data: Dict[str, Any]):
        """Send event to specific stream"""
        if stream_id in self.active_streams:
            queue = self.active_streams[stream_id]
            try:
                await queue.put(event_data)
            except asyncio.QueueFull:
                logger.warning(f"Stream {stream_id} queue full, dropping event")
    
    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast event to all active streams"""
        for stream_id, queue in self.active_streams.items():
            try:
                await queue.put(event_data)
            except asyncio.QueueFull:
                logger.warning(f"Stream {stream_id} queue full, dropping event")
    
    async def close_stream(self, stream_id: str):
        """Close a specific stream"""
        if stream_id in self.active_streams:
            await self.send_event(stream_id, {"type": "end", "timestamp": datetime.now().isoformat()})

class WorkflowStreamer:
    """Streams LangGraph workflow execution events"""
    
    def __init__(self, response_handler: StreamingResponseHandler):
        self.response_handler = response_handler
    
    async def stream_workflow(self, workflow, initial_state: Dict[str, Any], 
                            stream_id: str):
        """Stream workflow execution"""
        try:
            # Send start event
            await self.response_handler.send_event(stream_id, {
                "type": "workflow_start",
                "timestamp": datetime.now().isoformat(),
                "query": initial_state.get("query", "")
            })
            
            # Stream workflow execution
            async for event in workflow.astream(initial_state):
                # Parse workflow event
                event_data = self._parse_workflow_event(event)
                
                # Send to client
                await self.response_handler.send_event(stream_id, {
                    "type": "workflow_step",
                    "step": event_data["node"],
                    "data": event_data["data"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for completion
                if event_data.get("finished", False):
                    break
            
            # Send completion event
            await self.response_handler.send_event(stream_id, {
                "type": "workflow_complete",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error streaming workflow: {e}")
            await self.response_handler.send_event(stream_id, {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def _parse_workflow_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LangGraph workflow event"""
        # LangGraph events have specific structure
        for node_name, node_data in event.items():
            if isinstance(node_data, dict):
                return {
                    "node": node_name,
                    "data": node_data,
                    "finished": "__end__" in node_data
                }
        
        return {
            "node": "unknown",
            "data": event,
            "finished": False
        }

class ProgressTracker:
    """Tracks and streams progress of long-running operations"""
    
    def __init__(self, response_handler: StreamingResponseHandler):
        self.response_handler = response_handler
    
    async def track_ingestion(self, stream_id: str, file_paths: List[str]):
        """Track document ingestion progress"""
        total_files = len(file_paths)
        
        await self.response_handler.send_event(stream_id, {
            "type": "ingestion_start",
            "total_files": total_files,
            "timestamp": datetime.now().isoformat()
        })
        
        for i, file_path in enumerate(file_paths):
            try:
                # Simulate ingestion progress
                await self.response_handler.send_event(stream_id, {
                    "type": "ingestion_progress",
                    "file": file_path,
                    "current": i + 1,
                    "total": total_files,
                    "progress": ((i + 1) / total_files) * 100,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Simulate processing time
                await asyncio.sleep(0.1)
                
            except Exception as e:
                await self.response_handler.send_event(stream_id, {
                    "type": "ingestion_error",
                    "file": file_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        await self.response_handler.send_event(stream_id, {
            "type": "ingestion_complete",
            "total_files": total_files,
            "timestamp": datetime.now().isoformat()
        })

# Global streaming handler
streaming_handler = StreamingResponseHandler()
workflow_streamer = WorkflowStreamer(streaming_handler)
progress_tracker = ProgressTracker(streaming_handler)

async def create_stream_response(stream_id: str):
    """Create streaming response for SSE"""
    async def generate():
        async for chunk in streaming_handler.create_stream(stream_id):
            yield chunk
    
    return generate
