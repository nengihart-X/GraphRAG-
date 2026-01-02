from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import structlog
import re

from ..models.schemas import RAGState, DocumentChunk
from config.settings import settings

logger = structlog.get_logger()

class AnswerGenerationResult(BaseModel):
    """Pydantic model for answer generation output"""
    answer: str = Field(description="Generated answer to the user's query")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    citations: List[Dict[str, Any]] = Field(description="List of citations with source information")
    reasoning: str = Field(description="Brief explanation of how the answer was generated")

class AnswerGeneratorNode:
    """LangGraph node for generating context-aware answers"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3
        )
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=AnswerGenerationResult)
        
        # Setup prompts for different query intents
        self.qa_prompt = PromptTemplate(
            template="""You are a helpful assistant that answers questions based ONLY on the provided context. 

Query: {query}
Context: {context}

Instructions:
1. Answer the query using ONLY information from the provided context
2. If the context doesn't contain enough information to answer the query, say "I don't have enough information to answer this question"
3. Be accurate, concise, and helpful
4. Include specific citations to the sources you used
5. Do not make up or infer information not present in the context

{format_instructions}

Provide your answer:""",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.summarization_prompt = PromptTemplate(
            template="""You are a helpful assistant that summarizes information based ONLY on the provided context.

Query: {query}
Context: {context}

Instructions:
1. Create a comprehensive summary addressing the query using ONLY information from the provided context
2. Organize the summary logically with key points
3. Include specific citations to the sources you used
4. Do not include information not present in the context

{format_instructions}

Provide your summary:""",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.comparison_prompt = PromptTemplate(
            template="""You are a helpful assistant that compares information based ONLY on the provided context.

Query: {query}
Context: {context}

Instructions:
1. Compare and contrast information from the provided context to address the query
2. Identify similarities, differences, and key insights
3. Organize the comparison clearly
4. Include specific citations to the sources you used
5. Do not include information not present in the context

{format_instructions}

Provide your comparison:""",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.general_prompt = PromptTemplate(
            template="""You are a helpful assistant that provides information based ONLY on the provided context.

Query: {query}
Context: {context}

Instructions:
1. Address the query using ONLY information from the provided context
2. Be helpful and informative
3. Include specific citations to the sources you used
4. If the context doesn't contain relevant information, indicate this clearly
5. Do not make up or infer information not present in the context

{format_instructions}

Provide your response:""",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    async def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer based on query and retrieved documents"""
        try:
            ranked_docs = state.get("ranked_docs", [])
            if not ranked_docs:
                logger.warning("No documents available for answer generation")
                state["answer"] = "I don't have any relevant information to answer your question."
                state["confidence"] = 0.0
                state["citations"] = []
                return state
            
            logger.info(f"Generating answer using {len(ranked_docs)} documents")
            
            # Prepare context
            context = self._prepare_context(ranked_docs)
            
            # Select prompt based on query intent
            query_intent = state.get("query_intent", "general")
            if query_intent == "qa":
                prompt = self.qa_prompt
            elif query_intent == "summarization":
                prompt = self.summarization_prompt
            elif query_intent == "comparison":
                prompt = self.comparison_prompt
            else:
                prompt = self.general_prompt
            
            # Generate answer
            chain = prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "query": state["query"],
                "context": context
            })
            
            # Update state
            state["answer"] = result.answer
            state["confidence"] = result.confidence
            state["citations"] = result.citations
            state["answer_reasoning"] = result.reasoning
            
            logger.info(f"Answer generated with confidence: {result.confidence:.3f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            # Fallback response
            state["answer"] = "I encountered an error while generating your answer. Please try again."
            state["confidence"] = 0.0
            state["citations"] = []
            return state
    
    def _prepare_context(self, docs: List[DocumentChunk]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(docs):
            source = doc.get("source", "Unknown source")
            chunk_index = doc.get("chunk_index", i)
            content = doc["content"]
            
            context_part = f"[Source {i+1}: {source}, Chunk {chunk_index}]\n{content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    async def _extract_citations(self, answer: str, docs: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract and format citations from the answer"""
        citations = []
        
        # Look for citation patterns in the answer
        citation_patterns = [
            r'\[Source (\d+)\]',
            r'\[Document (\d+)\]',
            r'source (\d+)',
            r'document (\d+)'
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                source_num = int(match.group(1))
                if 1 <= source_num <= len(docs):
                    doc = docs[source_num - 1]
                    citation = {
                        "source_index": source_num,
                        "source": doc.get("source", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "text_snippet": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "score": doc.get("rerank_score", doc.get("score", 0))
                    }
                    citations.append(citation)
        
        # Remove duplicates
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation["source_index"], citation["chunk_index"])
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _calculate_confidence(self, answer: str, docs: List[DocumentChunk]) -> float:
        """Calculate confidence score for the answer"""
        if not docs or not answer:
            return 0.0
        
        # Factors for confidence calculation
        factors = []
        
        # 1. Average document relevance score
        avg_doc_score = sum(doc.get("rerank_score", doc.get("score", 0)) for doc in docs) / len(docs)
        factors.append(avg_doc_score)
        
        # 2. Number of relevant documents
        doc_count_factor = min(len(docs) / 5.0, 1.0)  # Cap at 1.0
        factors.append(doc_count_factor)
        
        # 3. Answer length (reasonable length indicates more complete answer)
        answer_length = len(answer.split())
        length_factor = min(answer_length / 100.0, 1.0)  # Cap at 1.0
        factors.append(length_factor)
        
        # 4. Presence of citations
        has_citations = 1.0 if "[Source" in answer or "[Document" in answer else 0.3
        factors.append(has_citations)
        
        # 5. Uncertainty indicators
        uncertainty_keywords = ["don't know", "not enough information", "cannot determine", "unclear"]
        uncertainty_penalty = 0.3 if any(keyword in answer.lower() for keyword in uncertainty_keywords) else 0.0
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.3]  # Total weight = 1.0
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        # Apply uncertainty penalty
        confidence = max(0.0, confidence - uncertainty_penalty)
        
        return confidence
    
    def __call__(self, state: RAGState) -> RAGState:
        """Make the node callable for LangGraph"""
        import asyncio
        return asyncio.run(self.generate_answer(state))
