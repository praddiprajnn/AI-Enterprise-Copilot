"""
Intelligent chunking strategies for documents
"""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from config import settings

class SmartChunker:
    """Implement intelligent document chunking"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._tiktoken_len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _tiktoken_len(self, text: str) -> int:
        """Calculate token length using tiktoken"""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "token_count": self._tiktoken_len(chunk)
            })
            chunk_docs.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return chunk_docs
    
    def semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Alternative: Semantic chunking based on content structure"""
        # Split by major sections (assuming titles in all caps or numbered)
        sections = re.split(r'\n\s*(?=[A-Z][A-Z\s]{10,}|[0-9]+\.)', text)
        
        chunks = []
        current_chunk = ""
        chunk_metadata = metadata.copy()
        
        for section in sections:
            if section.strip():
                if len(current_chunk) + len(section) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": chunk_metadata
                        })
                    current_chunk = section
                else:
                    current_chunk += "\n" + section
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks