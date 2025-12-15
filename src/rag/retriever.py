"""
Retriever for RAG pipeline
"""

from typing import List, Dict, Any, Optional
from src.embedding.embedder import SentenceTransformerEmbedder
from src.embedding.vector_store import VectorStore
from config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieve relevant documents for queries"""
    
    def __init__(self):
        logger.info("Initializing DocumentRetriever...")
        self.embedder = SentenceTransformerEmbedder()
        self.vector_store = VectorStore()
        self.similarity_threshold = 0.5  # LOWER THRESHOLD for better recall
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
    
    def retrieve(self, query: str, department: Optional[str] = None, 
                 k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if k is None:
            k = settings.TOP_K_RESULTS
            
        logger.info(f"🔍 Retrieving documents for query: '{query}'")
        logger.info(f"Requesting {k} results, department filter: {department}")
        
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed_text(query)
            logger.info(f"Generated query embedding of length: {len(query_embedding)}")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
        
        # Apply department filter if specified
        filter_dict = None
        if department and department != "All":
            filter_dict = {"department": department}
            logger.info(f"Applying department filter: {filter_dict}")
        
        # Search vector store
        try:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                filter_dict=filter_dict
            )
            logger.info(f"Vector store returned {len(results)} raw results")
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []
        
        # Filter by similarity threshold (but be more lenient)
        filtered_results = []
        for result in results:
            if result["score"] >= self.similarity_threshold:
                filtered_results.append(result)
                logger.debug(f"Accepted result with score: {result['score']:.3f}")
            else:
                logger.debug(f"Rejected result with low score: {result['score']:.3f}")
        
        logger.info(f"📊 Retrieved {len(filtered_results)} documents after filtering")
        
        # If no results pass threshold, return top 2 anyway for debugging
        if not filtered_results and results:
            logger.warning("No results passed threshold, returning top 2 for debugging")
            filtered_results = results[:2]
        
        return filtered_results
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            logger.warning("No documents retrieved for context formatting")
            return "No relevant documents found in the database."
        
        logger.info(f"Formatting context from {len(retrieved_docs)} documents")
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc["metadata"].get("filename", "Unknown Document")
            department = doc["metadata"].get("department", "Unknown Department")
            confidence = f"{doc['score']:.1%}"
            
            # Format each document clearly
            context = f"DOCUMENT {i+1} - {source} (Department: {department}, Relevance: {confidence}):\n"
            context += f"{doc['text']}\n"
            context_parts.append(context)
        
        full_context = "\n" + "="*60 + "\n".join(context_parts) + "="*60
        logger.info(f"Formatted context length: {len(full_context)} characters")
        
        return full_context