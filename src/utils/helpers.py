"""
Utility helper functions
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import hashlib
from datetime import datetime

def get_all_pdfs(directory: Path) -> List[Path]:
    """Get all PDF files from directory recursively"""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(Path(root) / file)
    return pdf_files

def calculate_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_processing_log(document_path: Path, status: str, metadata: Dict[str, Any]):
    """Save processing log"""
    log_file = Path("data/processed/processing_log.json")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "document": str(document_path),
        "status": status,
        "metadata": metadata
    }
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

def format_response_for_ui(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Format response for UI display"""
    formatted = response_dict.copy()
    
    # Add emojis based on confidence
    confidence = formatted.get("confidence", 0.5)
    if confidence > 0.8:
        formatted["confidence_emoji"] = "✅"
    elif confidence > 0.6:
        formatted["confidence_emoji"] = "⚠️"
    else:
        formatted["confidence_emoji"] = "❓"
    
    # Format sources for display
    sources = formatted.get("sources", [])
    if sources:
        formatted["sources_display"] = "\n".join([f"• {source}" for source in sources])
    else:
        formatted["sources_display"] = "No specific sources cited"
    
    return formatted