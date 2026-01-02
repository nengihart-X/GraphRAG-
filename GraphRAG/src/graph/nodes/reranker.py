from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import structlog
import numpy as np

from ..models.schemas import RAGState, DocumentChunk
from config.settings import settings

logger = structlog.get_logger()

class ReRankingResult(BaseModel):
    """Pydantic model for re-ranking output"""
    rankings: List[Dict[str, Any]] = Field(description="List of document rankings with scores and reasoning")
    summary: str = Field(description="Summary of re-ranking decisions")

class ReRankerNode:
    """LangGraph node for re-ranking retrieved documents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1
        )
        self.rerank_top_k = settings.rerank_top_k
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=ReRankingResult)
        
        # Setup prompts
        self.rerank_prompt = PromptTemplate(
            template="""You are an expert information retrieval specialist. Your task is to re-rank the following retrieved documents based on their relevance to the query.

Query: {query}
Query Intent: {query_intent}

Retrieved Documents:
{documents}

Your task is to:
1. Analyze each document's relevance to the query
2. Consider factors like:
   - Semantic similarity to the query
   - Quality and completeness of information
   - Presence of key entities and concepts
   - Likelihood of containing the answer
3. Remove any documents that seem irrelevant or low-quality
4. Rank the remaining documents from most to least relevant
5. Provide a brief reasoning for each ranking

{format_instructions}

Provide your re-ranking analysis:""",
            input_variables=["query", "query_intent", "documents"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # Cross-encoder for scoring (simplified version)
        self.cross_encoder_prompt = PromptTemplate(
            template="""Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.

Query: {query}
Document: {document}

Consider:
- Does the document directly address the query?
- Is the information accurate and complete?
- Does it contain relevant facts or details?
- Is it well-structured and readable?

Provide only a numerical score (0.0 to 1.0):""",
            input_variables=["query", "document"]
        )
    
    async def rerank(self, state: RAGState) -> RAGState:
        """Re-rank retrieved documents"""
        try:
            retrieved_docs = state.get("retrieved_docs", [])
            if not retrieved_docs:
                logger.warning("No documents to re-rank")
                state["ranked_docs"] = []
                return state
            
            logger.info(f"Re-ranking {len(retrieved_docs)} documents")
            
            # Use LLM-based re-ranking for smaller sets
            if len(retrieved_docs) <= 10:
                ranked_docs = await self._llm_rerank(state)
            else:
                # Use cross-encoder scoring for larger sets
                ranked_docs = await self._cross_encoder_rerank(state)
            
            # Keep only top-k documents
            state["ranked_docs"] = ranked_docs[:self.rerank_top_k]
            
            logger.info(f"Re-ranked to {len(state['ranked_docs'])} documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            # Fallback to original order
            state["ranked_docs"] = state.get("retrieved_docs", [])[:self.rerank_top_k]
            return state
    
    async def _llm_rerank(self, state: RAGState) -> List[DocumentChunk]:
        """Use LLM for intelligent re-ranking"""
        try:
            # Prepare documents for prompt
            docs_text = ""
            for i, doc in enumerate(state["retrieved_docs"]):
                docs_text += f"\nDocument {i+1}:\n{doc['content'][:500]}...\n"
            
            # Generate re-ranking
            chain = self.rerank_prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "query": state["query"],
                "query_intent": state.get("query_intent", "general"),
                "documents": docs_text
            })
            
            # Apply rankings
            ranked_docs = []
            for ranking in result.rankings:
                doc_index = ranking.get("doc_index", 0) - 1  # Convert to 0-based
                if 0 <= doc_index < len(state["retrieved_docs"]):
                    doc = state["retrieved_docs"][doc_index].copy()
                    doc["rerank_score"] = ranking.get("score", 0.0)
                    doc["rerank_reasoning"] = ranking.get("reasoning", "")
                    ranked_docs.append(doc)
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"LLM re-ranking error: {e}")
            return state["retrieved_docs"]
    
    async def _cross_encoder_rerank(self, state: RAGState) -> List[DocumentChunk]:
        """Use cross-encoder style scoring for re-ranking"""
        try:
            query = state["query"]
            scored_docs = []
            
            for doc in state["retrieved_docs"]:
                # Generate relevance score
                chain = self.cross_encoder_prompt | self.llm
                score_text = await chain.ainvoke({
                    "query": query,
                    "document": doc["content"][:800]  # Limit length
                })
                
                # Extract numerical score
                try:
                    score = float(score_text.content.strip())
                    score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                except:
                    score = doc.get("score", 0.5)  # Fallback to original score
                
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = score
                scored_docs.append(doc_copy)
            
            # Sort by re-rank score
            scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return scored_docs
            
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking error: {e}")
            return state["retrieved_docs"]
    
    async def _filter_low_quality_docs(self, docs: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter out low-quality documents"""
        filtered_docs = []
        
        for doc in docs:
            content = doc["content"]
            
            # Basic quality checks
            if len(content.strip()) < 50:  # Too short
                continue
            
            if len(content.split()) < 10:  # Too few words
                continue
            
            # Check for repetitive content
            words = content.lower().split()
            if len(set(words)) / len(words) < 0.3:  # Low diversity
                continue
            
            # Check for gibberish (simplified)
            if content.count('?') + content.count('!') > len(content) * 0.1:
                continue
            
            filtered_docs.append(doc)
        
        return filtered_docs
    
    def __call__(self, state: RAGState) -> RAGState:
        """Make the node callable for LangGraph"""
        import asyncio
        return asyncio.run(self.rerank(state))
