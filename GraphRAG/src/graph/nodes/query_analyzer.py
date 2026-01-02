from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import structlog
from sentence_transformers import SentenceTransformer

from ..models.schemas import RAGState, QueryAnalysis
from config.settings import settings

logger = structlog.get_logger()

class QueryAnalysisResult(BaseModel):
    """Pydantic model for query analysis output"""
    intent: str = Field(description="Query intent: 'qa', 'summarization', 'comparison', or 'general'")
    keywords: List[str] = Field(description="Key keywords for retrieval")
    entities: List[str] = Field(description="Named entities mentioned in the query")
    rewritten_query: str = Field(description="Optimized query for better retrieval")
    search_strategy: str = Field(description="Search strategy: 'vector', 'hybrid', or 'keyword'")

class QueryAnalyzerNode:
    """LangGraph node for analyzing and rewriting queries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=QueryAnalysisResult)
        
        # Setup prompts
        self.analysis_prompt = PromptTemplate(
            template="""Analyze the following user query and optimize it for document retrieval.

Query: {query}

Your task is to:
1. Identify the user's intent (qa, summarization, comparison, or general)
2. Extract key keywords that would be useful for retrieval
3. Identify any named entities mentioned
4. Rewrite the query to improve retrieval effectiveness
5. Recommend the best search strategy

Consider:
- What type of answer is the user looking for?
- What terms would be most effective for finding relevant documents?
- How can we expand or rephrase the query to capture more relevant content?
- Should we focus on semantic similarity, exact keywords, or both?

{format_instructions}

Provide your analysis:""",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    async def analyze(self, state: RAGState) -> RAGState:
        """Analyze query and update state"""
        try:
            logger.info(f"Analyzing query: {state['query']}")
            
            # Generate query analysis using LLM
            chain = self.analysis_prompt | self.llm | self.parser
            analysis_result = await chain.ainvoke({"query": state["query"]})
            
            # Generate embedding for the rewritten query
            query_embedding = self.embedding_model.encode(
                analysis_result.rewritten_query
            ).tolist()
            
            # Update state
            state["rewritten_query"] = analysis_result.rewritten_query
            state["query_intent"] = analysis_result.intent
            state["query_embedding"] = query_embedding
            
            # Store analysis results for later use
            state["query_analysis"] = {
                "intent": analysis_result.intent,
                "keywords": analysis_result.keywords,
                "entities": analysis_result.entities,
                "rewritten_query": analysis_result.rewritten_query,
                "search_strategy": analysis_result.search_strategy,
                "original_query": state["query"]
            }
            
            logger.info(f"Query analysis complete. Intent: {analysis_result.intent}, Strategy: {analysis_result.search_strategy}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback to basic processing
            state["rewritten_query"] = state["query"]
            state["query_intent"] = "general"
            state["query_embedding"] = self.embedding_model.encode(state["query"]).tolist()
            state["query_analysis"] = {
                "intent": "general",
                "keywords": [],
                "entities": [],
                "rewritten_query": state["query"],
                "search_strategy": "vector",
                "original_query": state["query"]
            }
            return state
    
    def __call__(self, state: RAGState) -> RAGState:
        """Make the node callable for LangGraph"""
        import asyncio
        return asyncio.run(self.analyze(state))
