
'''
#before-file ai_enterprise_copilot/config/settings.py
"""
Configuration settings for the AI Enterprise Copilot
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Document categories
DOCUMENT_CATEGORIES = {
    "hr": "Human Resources",
    "it_tech": "IT & Technical",
    "operations": "Operations",
    "company_wide": "Company Wide"
}

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding settings - USING LOCAL SENTENCE-TRANSFORMERS
# Model options: 
# - "all-MiniLM-L6-v2" (384 dimensions, good balance)
# - "all-mpnet-base-v2" (768 dimensions, higher quality)
# - "paraphrase-MiniLM-L3-v2" (384 dimensions, faster)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Must match the chosen model's dimension

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

OLLAMA_MODEL = "llama3.1:8b"  # or "phi4-mini"
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.3
# Gemini model settings (for chat/generation only - NOT for embeddings)
GEMINI_MODEL = "gemini-2.5-flash"  # Using flash for speed and cost efficiency
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.3

# Rate limiting for Gemini API (to avoid quota issues)
# Free tier limits: ~10 requests per minute
GEMINI_RATE_LIMIT_REQUESTS = 8  # Stay under 10
GEMINI_RATE_LIMIT_PERIOD = 60   # seconds

# UI settings
APP_TITLE = "AI Enterprise Copilot"
APP_DESCRIPTION = "Your intelligent assistant for HR, IT, and Operations queries"
APP_ICON = "🤖"

# Ensure directories exist
for directory in [RAW_DOCS_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model info based on selection
MODEL_INFO = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Good general-purpose model",
        "size": "~90MB"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Higher quality, slower",
        "size": "~420MB"
    },
    "paraphrase-MiniLM-L3-v2": {
        "dimension": 384,
        "description": "Fast, good for paraphrasing",
        "size": "~70MB"
    }
}

# Get current model info
def get_model_info():
    """Get information about the current embedding model"""
    model = EMBEDDING_MODEL
    if model in MODEL_INFO:
        return MODEL_INFO[model]
    else:
        return {
            "dimension": EMBEDDING_DIMENSION,
            "description": "Custom model",
            "size": "Unknown"
        }
        '''
"""
Configuration settings for the AI Enterprise Copilot
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Document categories
DOCUMENT_CATEGORIES = {
    "hr": "Human Resources",
    "it_tech": "IT & Technical",
    "operations": "Operations",
    "company_wide": "Company Wide"
}

# Chunking settings - SMALLER CHUNKS for better precision
CHUNK_SIZE = 800      # Reduced from 1000
CHUNK_OVERLAP = 150   # Increased from 200

# Embedding settings - USING LOCAL SENTENCE-TRANSFORMERS
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval settings - MORE RESULTS for better coverage
TOP_K_RESULTS = 8     # Increased from 5
SIMILARITY_THRESHOLD = 0.5  # Lowered for better recall

# Ollama settings - LOCAL MODEL
OLLAMA_MODEL = "llama3.1:8b"  # Make sure you pulled this: ollama pull llama3.1:8b
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.3

# UI settings
APP_TITLE = "AI Enterprise Copilot"
APP_DESCRIPTION = "Your intelligent assistant for HR, IT, and Operations queries"
APP_ICON = "🤖"

# Ensure directories exist
for directory in [RAW_DOCS_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Add debug mode
DEBUG_MODE = True