"""
Setup script to process PDFs and create vector database
"""

import sys
from pathlib import Path
from src.document_processor.pdf_processor import PDFProcessor
from src.document_processor.chunking_strategy import SmartChunker
from src.embedding.embedder import SentenceTransformerEmbedder
from src.embedding.vector_store import VectorStore
from src.utils.helpers import get_all_pdfs, save_processing_log
from config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database():
    """Main function to setup the vector database"""
    
    print("🚀 Starting AI Enterprise Copilot Database Setup")
    print("=" * 50)
    
    # Initialize components
    pdf_processor = PDFProcessor()
    chunker = SmartChunker()
    embedder = SentenceTransformerEmbedder()
    vector_store = VectorStore()
    
    # Clear existing collection
    print("🗑️  Clearing existing vector database...")
    vector_store.clear_collection()
    
    # Get all PDF files
    pdf_files = get_all_pdfs(settings.RAW_DOCS_DIR)
    print(f"📚 Found {len(pdf_files)} PDF files to process")
    
    total_chunks = 0
    
    for pdf_file in pdf_files:
        try:
            print(f"\n📄 Processing: {pdf_file.name}")
            
            # Extract text
            text = pdf_processor.extract_text_from_pdf(pdf_file)
            if not text.strip():
                print(f"   ⚠️  No text extracted, skipping...")
                continue
            
            # Extract metadata
            metadata = pdf_processor.extract_metadata(pdf_file)
            
            # Chunk document
            chunks = chunker.chunk_document(text, metadata)
            print(f"   ✂️  Created {len(chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            print(f"   🔤 Generating embeddings...")
            embeddings = embedder.embed_batch(texts)
            
            # Add to vector store
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(chunks, embeddings):
                if embedder.validate_embedding(embedding):
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            if valid_chunks:
                vector_store.add_documents(valid_chunks, valid_embeddings)
                total_chunks += len(valid_chunks)
                
                # Log successful processing
                save_processing_log(
                    pdf_file,
                    "success",
                    {
                        "chunks_processed": len(valid_chunks),
                        "total_text_length": len(text),
                        "department": metadata.get("department", "Unknown")
                    }
                )
                
                print(f"   ✅ Added {len(valid_chunks)} chunks to vector store")
            else:
                print(f"   ❌ No valid embeddings generated")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            save_processing_log(pdf_file, "failed", {"error": str(e)})
            print(f"   ❌ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("🏁 Database Setup Complete!")
    print(f"📊 Total chunks processed: {total_chunks}")
    
    # Show collection info
    info = vector_store.get_collection_info()
    print(f"🗃️  Collection: {info['collection_name']}")
    print(f"📈 Document count: {info['document_count']}")
    print(f"💾 Storage location: {info['location']}")
    
    print("\n✅ Setup completed successfully!")
    print("\nTo start the application, run: streamlit run app.py")

if __name__ == "__main__":
    try:
        setup_database()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)