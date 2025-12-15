"""
RAG Chain for response generation using Local Ollama
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from config import settings
import logging

logger = logging.getLogger(__name__)

class RAGChain:
    """RAG pipeline for generating responses using a local Ollama model"""
    
    def __init__(self):
        # Initialize the local Ollama model
        try:
            logger.info(f"Attempting to connect to Ollama at {settings.OLLAMA_BASE_URL}")
            logger.info(f"Loading model: {settings.OLLAMA_MODEL}")
            
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.TEMPERATURE,
                num_predict=settings.MAX_OUTPUT_TOKENS,
            )
            logger.info(f"✅ Successfully initialized Ollama model: {settings.OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
            logger.error("Please ensure Ollama is running with: ollama serve")
            raise
    
    def generate_response(self, query: str, context: str,
                         conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate response using RAG with local Ollama"""
        
        logger.info(f"Generating response for query: '{query}'")
        logger.info(f"Context length: {len(context)} characters")
        
        # Debug: Log first 500 chars of context
        if context and len(context) > 500:
            logger.info(f"Context preview: {context[:500]}...")
        else:
            logger.info(f"Full context: {context}")
        
        try:
            # Build the complete prompt
            prompt = self._build_prompt(query, context, conversation_history)
            logger.debug(f"Final prompt sent to model:\n{prompt}")
            
            # Create a simple chain - FIXED APPROACH
            messages = [
                ("system", "You are a helpful enterprise AI assistant. Answer questions STRICTLY based on the provided context."),
                ("human", prompt)
            ]
            
            prompt_template = ChatPromptTemplate.from_messages(messages)
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Invoke the chain
            response_text = chain.invoke({"input": prompt})
            
            logger.info(f"✅ Raw model response: {response_text[:200]}...")
            
            # Check for lack of information - less aggressive
            if self._check_no_information(response_text):
                logger.warning("Model indicated lack of information in response")
                # Don't override with canned response if model already said it
                if "don't have enough information" not in response_text.lower() and "not in the context" not in response_text.lower():
                    response_text = "Based on the available company documents, I cannot find specific information about this. Please consult the HR department for details."
            
            return {
                "response": response_text,
                "sources": self._extract_sources_from_context(context),
                "confidence": self._estimate_confidence(response_text, context)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please check if Ollama is running.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _build_prompt(self, query: str, context: str,
                     conversation_history: Optional[List[Dict]]) -> str:
        """Build the RAG prompt with explicit instructions"""
        
        # CRITICAL FIX: Better prompt structure
        prompt = f"""Answer the question based ONLY on the following context. 
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER BASED ONLY ON CONTEXT:"""
        
        return prompt
    
    def _check_no_information(self, response: str) -> bool:
        """Check if response indicates lack of information - LESS AGGRESSIVE"""
        # Only check if response is extremely short or contains specific phrases
        response_lower = response.lower()
        
        # If response is too short (likely empty or error)
        if len(response.strip()) < 10:
            return True
            
        # Check for explicit "I don't know" phrases from the model itself
        no_info_phrases = [
            "i cannot find",
            "not in the context",
            "no information provided",
            "context does not mention",
            "the context does not contain"
        ]
        
        for phrase in no_info_phrases:
            if phrase in response_lower:
                return True
                
        return False
    
    def _extract_sources_from_context(self, context: str) -> List[str]:
        """Extract source document names from context"""
        import re
        sources = []
        
        # Look for patterns like "Source: filename.pdf"
        source_patterns = [
            r'Source:\s*([^\|]+?\.pdf)',
            r'from ([^\|]+?\.pdf)',
            r'filename:\s*([^\|]+?\.pdf)'
        ]
        
        for pattern in source_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            sources.extend([match.strip() for match in matches])
        
        # Remove duplicates but preserve order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return unique_sources if unique_sources else ["General company documents"]
    
    def _estimate_confidence(self, response: str, context: str) -> float:
        """Estimate confidence score for the response"""
        # Simple confidence based on response length and specificity
        confidence = 0.5  # Base confidence
        
        # Increase confidence for longer, detailed responses
        word_count = len(response.split())
        if word_count > 20:
            confidence += 0.2
        elif word_count > 10:
            confidence += 0.1
            
        # Decrease confidence if response has uncertainty markers
        uncertainty_words = ["may", "might", "could", "possibly", "perhaps"]
        if any(word in response.lower() for word in uncertainty_words):
            confidence -= 0.1
            
        # Cap between 0.1 and 0.95
        return max(0.1, min(0.95, confidence))