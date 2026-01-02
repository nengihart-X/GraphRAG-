from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import structlog
import re

from ..models.schemas import RAGState, ValidationResult
from config.settings import settings

logger = structlog.get_logger()

class AnswerValidationResult(BaseModel):
    """Pydantic model for answer validation output"""
    is_grounded: bool = Field(description="Whether the answer is grounded in the provided context")
    hallucination_score: float = Field(description="Score indicating likelihood of hallucination (0.0 to 1.0)")
    confidence_score: float = Field(description="Confidence in the answer quality (0.0 to 1.0)")
    issues: List[str] = Field(description="List of identified issues with the answer")
    needs_regeneration: bool = Field(description="Whether the answer needs to be regenerated")

class AnswerValidatorNode:
    """LangGraph node for validating answer quality and detecting hallucinations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1
        )
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=AnswerValidationResult)
        
        # Setup validation prompts
        self.validation_prompt = PromptTemplate(
            template="""You are an expert fact-checker and content validator. Your task is to validate the quality and accuracy of an AI-generated answer.

Original Query: {query}
Generated Answer: {answer}
Context: {context}

Your validation criteria:
1. Grounding: Is the answer based ONLY on the provided context?
2. Accuracy: Does the answer correctly represent the information in the context?
3. Completeness: Does the answer fully address the user's query?
4. Consistency: Is the answer internally consistent?
5. Hallucination: Does the answer contain information not present in the context?

Look for these issues:
- Claims not supported by the context
- Contradictions with the context
- Made-up facts, figures, or sources
- Overly specific details not in context
- Uncertainty or hedging when context is clear
- Missing important relevant information

{format_instructions}

Provide your validation assessment:""",
            input_variables=["query", "answer", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.grounding_check_prompt = PromptTemplate(
            template="""Check if each statement in the answer is supported by the context.

Answer: {answer}
Context: {context}

For each statement in the answer, determine if it's directly supported by the context.
Respond with:
SUPPORTED: [statement] - if supported
UNSUPPORTED: [statement] - if not supported

Provide your analysis:""",
            input_variables=["answer", "context"]
        )
    
    async def validate_answer(self, state: RAGState) -> RAGState:
        """Validate answer quality and detect hallucinations"""
        try:
            answer = state.get("answer", "")
            if not answer:
                logger.warning("No answer to validate")
                state["validation_passed"] = False
                state["validation_result"] = {
                    "is_grounded": False,
                    "hallucination_score": 1.0,
                    "confidence_score": 0.0,
                    "issues": ["No answer provided"],
                    "needs_regeneration": True
                }
                return state
            
            ranked_docs = state.get("ranked_docs", [])
            context = self._prepare_context(ranked_docs)
            
            logger.info("Validating answer quality")
            
            # Perform comprehensive validation
            validation_result = await self._comprehensive_validation(state, context)
            
            # Update state
            state["validation_passed"] = validation_result["is_grounded"] and validation_result["hallucination_score"] < 0.5
            state["validation_result"] = validation_result
            
            logger.info(f"Validation complete. Grounded: {validation_result['is_grounded']}, "
                       f"Hallucination score: {validation_result['hallucination_score']:.3f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in answer validation: {e}")
            # Conservative fallback
            state["validation_passed"] = False
            state["validation_result"] = {
                "is_grounded": False,
                "hallucination_score": 0.8,
                "confidence_score": 0.2,
                "issues": [f"Validation error: {str(e)}"],
                "needs_regeneration": True
            }
            return state
    
    async def _comprehensive_validation(self, state: RAGState, context: str) -> Dict[str, Any]:
        """Perform comprehensive answer validation"""
        try:
            # Use LLM for detailed validation
            chain = self.validation_prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "query": state["query"],
                "answer": state["answer"],
                "context": context
            })
            
            validation = {
                "is_grounded": result.is_grounded,
                "hallucination_score": result.hallucination_score,
                "confidence_score": result.confidence_score,
                "issues": result.issues,
                "needs_regeneration": result.needs_regeneration
            }
            
            # Perform additional rule-based checks
            rule_based_issues = self._rule_based_validation(state["answer"], context)
            validation["issues"].extend(rule_based_issues)
            
            # Adjust scores based on rule-based findings
            if rule_based_issues:
                validation["hallucination_score"] = min(1.0, validation["hallucination_score"] + 0.2)
                validation["confidence_score"] = max(0.0, validation["confidence_score"] - 0.2)
                validation["is_grounded"] = False
            
            return validation
            
        except Exception as e:
            logger.error(f"Comprehensive validation error: {e}")
            return {
                "is_grounded": False,
                "hallucination_score": 0.8,
                "confidence_score": 0.2,
                "issues": [f"Validation failed: {str(e)}"],
                "needs_regeneration": True
            }
    
    def _rule_based_validation(self, answer: str, context: str) -> List[str]:
        """Perform rule-based validation checks"""
        issues = []
        
        # Check for specific hallucination indicators
        hallucination_patterns = [
            (r'\b\d{4}\b', "Specific years not found in context"),
            (r'\$\d+(?:,\d{3})*(?:\.\d{2})?', "Specific monetary amounts not verified"),
            (r'\b\d+%\b', "Specific percentages not verified"),
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', "Specific dates not verified"),
        ]
        
        for pattern, issue in hallucination_patterns:
            matches = re.findall(pattern, answer)
            if matches:
                # Check if these patterns exist in context
                context_matches = re.findall(pattern, context)
                if len(matches) > len(context_matches):
                    issues.append(issue)
        
        # Check for uncertainty vs confidence mismatch
        uncertainty_phrases = ["I think", "maybe", "perhaps", "possibly", "it seems", "it appears"]
        confidence_phrases = ["definitely", "certainly", "absolutely", "clearly", "obviously"]
        
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        has_confidence = any(phrase in answer.lower() for phrase in confidence_phrases)
        
        if has_uncertainty and has_confidence:
            issues.append("Mixed uncertainty and confidence signals")
        
        # Check for source citation issues
        citation_pattern = r'\[Source (\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        if citations:
            max_citation = max(int(c) for c in citations)
            # This would need access to the actual number of sources
            # For now, just check if citations are reasonable
            if max_citation > 10:  # Arbitrary reasonable limit
                issues.append("Unusually high citation numbers")
        
        # Check for answer length vs content quality
        if len(answer.split()) < 10:
            issues.append("Answer too short")
        elif len(answer.split()) > 500:
            issues.append("Answer unusually long - potential verbosity")
        
        return issues
    
    async def _grounding_check(self, answer: str, context: str) -> Dict[str, Any]:
        """Detailed grounding check for answer statements"""
        try:
            chain = self.grounding_check_prompt | self.llm
            result = await chain.ainvoke({
                "answer": answer,
                "context": context
            })
            
            # Parse the result to count supported vs unsupported statements
            content = result.content
            supported_count = content.count("SUPPORTED:")
            unsupported_count = content.count("UNSUPPORTED:")
            
            total_statements = supported_count + unsupported_count
            if total_statements == 0:
                return {"grounding_score": 0.5, "supported_ratio": 0.5}
            
            supported_ratio = supported_count / total_statements
            grounding_score = supported_ratio
            
            return {
                "grounding_score": grounding_score,
                "supported_ratio": supported_ratio,
                "total_statements": total_statements,
                "unsupported_statements": unsupported_count
            }
            
        except Exception as e:
            logger.error(f"Grounding check error: {e}")
            return {"grounding_score": 0.5, "supported_ratio": 0.5}
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context for validation"""
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            source = doc.get("source", f"Source {i+1}")
            context_parts.append(f"[{source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def should_regenerate(self, state: RAGState) -> bool:
        """Determine if answer should be regenerated"""
        validation_result = state.get("validation_result", {})
        
        # Check if regeneration is needed
        if validation_result.get("needs_regeneration", False):
            return True
        
        if not validation_result.get("is_grounded", False):
            return True
        
        if validation_result.get("hallucination_score", 0) > 0.6:
            return True
        
        if validation_result.get("confidence_score", 0) < 0.3:
            return True
        
        return False
    
    def __call__(self, state: RAGState) -> RAGState:
        """Make the node callable for LangGraph"""
        import asyncio
        return asyncio.run(self.validate_answer(state))
