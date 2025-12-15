"""
Local Embedding generation using Sentence Transformers
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config import settings
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder:
    """Generate embeddings using a local Sentence Transformer model"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            # This downloads the model once (if not cached) and loads it into memory
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model_name = "paraphrase-MiniLM-L3-v2"
            self.dimension = 384
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Using fallback model: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # The model.encode() method returns a numpy array
            embedding = self.model.encode(text)
            # Convert to a regular Python list of floats
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts efficiently"""
        try:
            # Encode all texts at once (much faster than looping)
            embeddings = self.model.encode(texts)
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Fallback: generate embeddings one by one
            return [self.embed_text(text) for text in texts]
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate if embedding is valid"""
        if not embedding:
            return False
        if len(embedding) != self.dimension:
            return False
        if all(abs(v) < 1e-10 for v in embedding):  # Check if all values are ~zero
            return False
        return True