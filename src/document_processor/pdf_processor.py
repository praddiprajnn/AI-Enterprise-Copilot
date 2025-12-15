"""
PDF Document Processor
Handles PDF text extraction and preprocessing
"""

import PyPDF2
import pdfplumber
from typing import List, Dict, Any
import re
from pathlib import Path
import logging
from config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents for RAG pipeline"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            
            # Try pdfplumber first for better text extraction
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Fallback to PyPDF2 if pdfplumber fails
            if not text.strip():
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\r]', '', text)
        return text.strip()
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF path and content"""
        metadata = {
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "department": self._get_department_from_path(pdf_path),
            "document_type": "pdf"
        }
        
        # Extract additional metadata from PDF if available
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.metadata:
                    if reader.metadata.title:
                        metadata["title"] = reader.metadata.title
                    if reader.metadata.author:
                        metadata["author"] = reader.metadata.author
        except:
            pass
        
        return metadata
    
    def _get_department_from_path(self, pdf_path: Path) -> str:
        """Determine department from file path"""
        path_str = str(pdf_path)
        for dept in settings.DOCUMENT_CATEGORIES.keys():
            if dept in path_str.lower():
                return settings.DOCUMENT_CATEGORIES[dept]
        return "Unknown"