from typing import Dict, Any, List, Optional
import asyncio
import structlog
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

from ..models.schemas import RAGState, DocumentChunk, RetrievalResult
from ..ingestion.pipeline import VectorStore
from config.settings import settings

logger = structlog.get_logger()

class RetrieverNode:
    """LangGraph node for document retrieval with hybrid search"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.default_top_k = settings.default_top_k
        self.retrieval_threshold = settings.retrieval_threshold
    
    async def retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on query"""
        try:
            query_embedding = state.get("query_embedding")
            if not query_embedding:
                # Generate embedding if not present
                query_text = state.get("rewritten_query", state["query"])
                query_embedding = self.embedding_model.encode(query_text).tolist()
                state["query_embedding"] = query_embedding
            
            query_analysis = state.get("query_analysis", {})
            search_strategy = query_analysis.get("search_strategy", "vector")
            
            logger.info(f"Retrieving documents with strategy: {search_strategy}")
            
            # Perform retrieval based on strategy
            if search_strategy == "hybrid":
                retrieved_docs = await self._hybrid_search(
                    query_embedding, 
                    state["rewritten_query"],
                    top_k=self.default_top_k
                )
            elif search_strategy == "keyword":
                retrieved_docs = await self._keyword_search(
                    state["rewritten_query"],
                    top_k=self.default_top_k
                )
            else:  # vector search (default)
                retrieved_docs = await self._vector_search(
                    query_embedding,
                    top_k=self.default_top_k
                )
            
            # Update state
            state["retrieved_docs"] = retrieved_docs
            
            # Calculate retrieval metrics
            if retrieved_docs:
                avg_score = np.mean([doc.get("score", 0) for doc in retrieved_docs])
                max_score = max([doc.get("score", 0) for doc in retrieved_docs])
                
                state["retrieval_metrics"] = {
                    "num_docs": len(retrieved_docs),
                    "avg_score": float(avg_score),
                    "max_score": float(max_score),
                    "strategy": search_strategy
                }
                
                logger.info(f"Retrieved {len(retrieved_docs)} docs, avg_score: {avg_score:.3f}")
            else:
                state["retrieval_metrics"] = {
                    "num_docs": 0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "strategy": search_strategy
                }
                logger.warning("No documents retrieved")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            state["retrieved_docs"] = []
            state["retrieval_metrics"] = {"error": str(e)}
            return state
    
    async def _vector_search(self, query_embedding: List[float], 
                           top_k: int = 10) -> List[DocumentChunk]:
        """Perform vector similarity search"""
        try:
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _keyword_search(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """Perform keyword-based search using BM25"""
        try:
            # This is a simplified implementation
            # In production, you'd want to maintain a BM25 index
            # For now, we'll fall back to vector search with keyword filtering
            
            # Extract keywords from query
            keywords = query.lower().split()
            
            # Get all documents and filter by keywords
            all_docs = await self.vector_store.search(
                query_embedding=self.embedding_model.encode(query).tolist(),
                top_k=100  # Get more docs for filtering
            )
            
            # Filter documents that contain keywords
            filtered_docs = []
            for doc in all_docs:
                content_lower = doc["content"].lower()
                keyword_matches = sum(1 for kw in keywords if kw in content_lower)
                
                if keyword_matches > 0:
                    doc["score"] = doc.get("score", 0) * (1 + keyword_matches * 0.1)
                    filtered_docs.append(doc)
            
            # Sort by score and return top_k
            filtered_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            return filtered_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    async def _hybrid_search(self, query_embedding: List[float], 
                           query_text: str, top_k: int = 10) -> List[DocumentChunk]:
        """Perform hybrid search combining vector and keyword search"""
        try:
            # Get vector search results
            vector_docs = await self._vector_search(query_embedding, top_k * 2)
            
            # Get keyword search results
            keyword_docs = await self._keyword_search(query_text, top_k * 2)
            
            # Combine and deduplicate results
            combined_docs = {}
            
            # Add vector results with weight
            for doc in vector_docs:
                doc_id = doc["id"]
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["vector_score"] = doc.get("score", 0)
                combined_docs[doc_id]["keyword_score"] = 0
            
            # Add keyword results and merge scores
            for doc in keyword_docs:
                doc_id = doc["id"]
                if doc_id in combined_docs:
                    combined_docs[doc_id]["keyword_score"] = doc.get("score", 0)
                    # Combine scores (weighted average)
                    vector_score = combined_docs[doc_id]["vector_score"]
                    keyword_score = combined_docs[doc_id]["keyword_score"]
                    combined_docs[doc_id]["score"] = (vector_score * 0.7 + keyword_score * 0.3)
                else:
                    doc["vector_score"] = 0
                    doc["keyword_score"] = doc.get("score", 0)
                    doc["score"] = doc["keyword_score"] * 0.3
                    combined_docs[doc_id] = doc
            
            # Sort by combined score and return top_k
            final_docs = list(combined_docs.values())
            final_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return final_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to vector search
            return await self._vector_search(query_embedding, top_k)
    
    def should_retrieve_more(self, state: RAGState) -> bool:
        """Determine if we should retrieve more documents"""
        retrieval_metrics = state.get("retrieval_metrics", {})
        avg_score = retrieval_metrics.get("avg_score", 0)
        num_docs = retrieval_metrics.get("num_docs", 0)
        retrieval_attempts = state.get("retrieval_attempts", 0)
        
        # Check if retrieval quality is poor and we haven't exceeded max attempts
        if (avg_score < self.retrieval_threshold or num_docs < 3) and retrieval_attempts < settings.max_retrieval_attempts:
            return True
        
        return False
    
    def __call__(self, state: RAGState) -> RAGState:
        """Make the node callable for LangGraph"""
        return asyncio.run(self.retrieve(state))
