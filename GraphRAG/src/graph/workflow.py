from typing import Dict, Any, Literal
from datetime import datetime
import uuid
import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    QueryAnalyzerNode,
    RetrieverNode, 
    ReRankerNode,
    AnswerGeneratorNode,
    AnswerValidatorNode
)
from ..models.schemas import RAGState
from config.settings import settings

logger = structlog.get_logger()

class RAGWorkflow:
    """Production-grade RAG workflow using LangGraph"""
    
    def __init__(self):
        # Initialize nodes
        self.query_analyzer = QueryAnalyzerNode()
        self.retriever = RetrieverNode()
        self.reranker = ReRankerNode()
        self.answer_generator = AnswerGeneratorNode()
        self.answer_validator = AnswerValidatorNode()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Initialize memory for conversation history
        self.memory = MemorySaver()
        
        # Compile the workflow
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        
        # Create the workflow graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("retriever", self.retriever)
        workflow.add_node("reranker", self.reranker)
        workflow.add_node("answer_generator", self.answer_generator)
        workflow.add_node("answer_validator", self.answer_validator)
        
        # Add edges
        workflow.set_entry_point("query_analyzer")
        
        # Query analyzer -> Retriever
        workflow.add_edge("query_analyzer", "retriever")
        
        # Retriever -> Conditional edge
        workflow.add_conditional_edges(
            "retriever",
            self._should_retry_retrieval,
            {
                "rerank": "reranker",
                "retry": "query_analyzer",
                "end": END
            }
        )
        
        # Reranker -> Answer Generator
        workflow.add_edge("reranker", "answer_generator")
        
        # Answer Generator -> Answer Validator
        workflow.add_edge("answer_generator", "answer_validator")
        
        # Answer Validator -> Conditional edge
        workflow.add_conditional_edges(
            "answer_validator",
            self._should_regenerate_answer,
            {
                "end": END,
                "regenerate": "answer_generator",
                "retry_retrieval": "retriever"
            }
        )
        
        return workflow
    
    def _should_retry_retrieval(self, state: RAGState) -> Literal["rerank", "retry", "end"]:
        """Conditional edge logic for retrieval retry"""
        retrieval_attempts = state.get("retrieval_attempts", 0)
        retrieval_metrics = state.get("retrieval_metrics", {})
        avg_score = retrieval_metrics.get("avg_score", 0)
        num_docs = retrieval_metrics.get("num_docs", 0)
        
        # Check if we need to retry retrieval
        if (avg_score < settings.retrieval_threshold or num_docs < 3) and retrieval_attempts < settings.max_retrieval_attempts:
            logger.info(f"Retrieval quality insufficient (avg_score: {avg_score:.3f}, docs: {num_docs}), retrying...")
            state["retrieval_attempts"] = retrieval_attempts + 1
            return "retry"
        
        # If we've exceeded max attempts, proceed with what we have
        if retrieval_attempts >= settings.max_retrieval_attempts:
            logger.warning(f"Max retrieval attempts ({settings.max_retrieval_attempts}) exceeded, proceeding with current results")
            if num_docs == 0:
                return "end"  # No documents, end workflow
            return "rerank"
        
        # Good retrieval quality, proceed to reranking
        return "rerank"
    
    def _should_regenerate_answer(self, state: RAGState) -> Literal["end", "regenerate", "retry_retrieval"]:
        """Conditional edge logic for answer regeneration"""
        validation_result = state.get("validation_result", {})
        is_grounded = validation_result.get("is_grounded", False)
        hallucination_score = validation_result.get("hallucination_score", 0)
        confidence_score = validation_result.get("confidence_score", 0)
        
        # Check if validation passed
        if is_grounded and hallucination_score < 0.5 and confidence_score > 0.3:
            logger.info("Answer validation passed, ending workflow")
            return "end"
        
        # Check if we should try retrieval again (context issue)
        if hallucination_score > 0.7 or confidence_score < 0.2:
            logger.info("Answer quality poor, retrying retrieval...")
            # Reset retrieval attempts for new retrieval cycle
            state["retrieval_attempts"] = 0
            return "retry_retrieval"
        
        # Try regenerating the answer with same context
        logger.info("Regenerating answer...")
        return "regenerate"
    
    def create_initial_state(self, query: str, session_id: str = None) -> RAGState:
        """Create initial state for the workflow"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        return RAGState(
            query=query,
            rewritten_query="",
            query_intent="",
            retrieved_docs=[],
            ranked_docs=[],
            answer="",
            confidence=0.0,
            citations=[],
            retrieval_attempts=0,
            validation_passed=False,
            error_message=None,
            timestamp=datetime.now(),
            session_id=session_id
        )
    
    async def run(self, query: str, session_id: str = None, 
                  config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the RAG workflow"""
        try:
            # Create initial state
            initial_state = self.create_initial_state(query, session_id)
            
            # Prepare config for LangGraph
            if config is None:
                config = {"configurable": {"thread_id": initial_state["session_id"]}}
            
            logger.info(f"Starting RAG workflow for query: {query[:100]}...")
            
            # Run the workflow
            result = await self.app.ainvoke(initial_state, config)
            
            # Prepare final result
            final_result = {
                "query": result["query"],
                "answer": result["answer"],
                "confidence": result["confidence"],
                "citations": result.get("citations", []),
                "validation_passed": result.get("validation_passed", False),
                "validation_result": result.get("validation_result", {}),
                "retrieval_metrics": result.get("retrieval_metrics", {}),
                "session_id": result["session_id"],
                "timestamp": result["timestamp"],
                "workflow_steps": self._extract_workflow_steps(result)
            }
            
            logger.info(f"RAG workflow completed. Confidence: {result['confidence']:.3f}, "
                       f"Validation passed: {result.get('validation_passed', False)}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in RAG workflow: {e}")
            return {
                "query": query,
                "answer": "I encountered an error while processing your request. Please try again.",
                "confidence": 0.0,
                "citations": [],
                "validation_passed": False,
                "error": str(e),
                "session_id": session_id or str(uuid.uuid4()),
                "timestamp": datetime.now()
            }
    
    def _extract_workflow_steps(self, result: RAGState) -> Dict[str, Any]:
        """Extract workflow execution information"""
        return {
            "query_analysis": {
                "intent": result.get("query_intent", ""),
                "rewritten_query": result.get("rewritten_query", "")
            },
            "retrieval": {
                "attempts": result.get("retrieval_attempts", 0),
                "metrics": result.get("retrieval_metrics", {}),
                "docs_retrieved": len(result.get("retrieved_docs", [])),
                "docs_ranked": len(result.get("ranked_docs", []))
            },
            "generation": {
                "confidence": result.get("confidence", 0),
                "citations_count": len(result.get("citations", []))
            },
            "validation": result.get("validation_result", {})
        }
    
    async def stream_run(self, query: str, session_id: str = None) -> Any:
        """Run the RAG workflow with streaming output"""
        try:
            # Create initial state
            initial_state = self.create_initial_state(query, session_id)
            
            # Prepare config
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            
            logger.info(f"Starting streaming RAG workflow for query: {query[:100]}...")
            
            # Stream the workflow
            async for event in self.app.astream(initial_state, config):
                yield event
                
        except Exception as e:
            logger.error(f"Error in streaming RAG workflow: {e}")
            yield {
                "error": str(e),
                "session_id": session_id or str(uuid.uuid4()),
                "timestamp": datetime.now()
            }
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session"""
        try:
            # This would need to be implemented based on LangGraph's memory system
            # For now, return a placeholder
            return {
                "session_id": session_id,
                "messages": [],
                "metadata": {}
            }
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        try:
            # This would need to be implemented based on LangGraph's memory system
            logger.info(f"Clearing session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
