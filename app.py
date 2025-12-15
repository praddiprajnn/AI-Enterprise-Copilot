"""
Streamlit UI for AI Enterprise Copilot
"""

import streamlit as st
import sys
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress torch warnings
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.rag.retriever import DocumentRetriever
from src.rag.chain import RAGChain
from src.utils.helpers import format_response_for_ui
from config import settings
from storing_keys import validate_keys

# Page configuration
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon=settings.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .confidence-high {
        color: #059669;
        font-weight: bold;
    }
    .confidence-medium {
        color: #D97706;
        font-weight: bold;
    }
    .confidence-low {
        color: #DC2626;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
    }
    .assistant-message {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
    }
    .debug-info {
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #F59E0B;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

def initialize_components():
    """Initialize RAG components"""
    if st.session_state.retriever is None or st.session_state.rag_chain is None:
        with st.spinner("🚀 Initializing AI components..."):
            try:
                st.session_state.retriever = DocumentRetriever()
                st.info("✅ Document retriever initialized successfully")
                
                # Small delay to ensure Ollama is ready
                time.sleep(1)
                
                st.session_state.rag_chain = RAGChain()
                st.success("✅ AI model loaded and ready!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.error("Please ensure Ollama is running with: `ollama serve`")
                return False
    return True

def main():
    """Main application function"""
    
    # Header
    st.markdown(f'<h1 class="main-header">{settings.APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{settings.APP_DESCRIPTION}</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title("Settings & Controls")
        
        # Toggle debug mode
        st.session_state.debug_mode = st.checkbox("🔧 Enable Debug Mode", value=False)
        
        # Department filter
        st.subheader("Filter by Department")
        departments = ["All"] + list(settings.DOCUMENT_CATEGORIES.values())
        selected_dept = st.selectbox(
            "Select department for query",
            departments,
            help="Filter responses to specific department documents"
        )
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider(
                "Documents to retrieve",
                min_value=1,
                max_value=15,
                value=settings.TOP_K_RESULTS,
                help="More documents may provide better context"
            )
        with col2:
            similarity_thresh = st.slider(
                "Similarity threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Higher = more precise but fewer results"
            )
        
        # System info
        st.subheader("System Information")
        if st.session_state.retriever:
            try:
                info = st.session_state.retriever.vector_store.get_collection_info()
                st.info(f"""
                **Documents in DB:** {info['document_count']}
                
                **Embedding Model:** {settings.EMBEDDING_MODEL}
                
                **LLM Model:** {settings.OLLAMA_MODEL}
                """)
            except:
                st.info("Vector database info not available")
        
        # Debug Tools Section
        with st.expander("🛠️ Debug Tools", expanded=False):
            st.write("**Test Document Retrieval**")
            debug_query = st.text_input("Test query:", "casual leave policy", key="debug_query")
            
            col_debug1, col_debug2 = st.columns(2)
            with col_debug1:
                if st.button("🔍 Test Retrieval", key="test_retrieval"):
                    test_retrieval(debug_query, top_k, selected_dept if selected_dept != "All" else None)
            
            with col_debug2:
                if st.button("🤖 Test Ollama", key="test_ollama"):
                    test_ollama_connection()
            
            if st.button("📊 Show Raw Chunks", key="show_chunks"):
                show_raw_chunks()
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Rate limiting info
        st.caption(f"⏱️ Last query: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_query_time)) if st.session_state.last_query_time > 0 else 'Never'}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                st.warning("⚠️ Ollama service may not be running. Please run `ollama serve` in a terminal.")
        except:
            st.error("❌ Cannot connect to Ollama. Please ensure Ollama is running with `ollama serve`")
        
        # Initialize components
        if not initialize_components():
            st.stop()
        
        # Display conversation history
        st.subheader("💬 Conversation")
        
        if not st.session_state.conversation_history:
            st.info("💡 Start by asking a question below. Try: 'How many casual leaves can I take?' or 'What is the code of conduct?'")
        
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                    <small style="float:right; opacity:0.7;">{time.strftime('%H:%M', time.localtime(message.get('timestamp', 0)))}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                response = message.get("response", {})
                if isinstance(response, dict):
                    formatted = format_response_for_ui(response)
                    
                    # Display assistant response
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {formatted.get("response", "")}
                        <small style="float:right; opacity:0.7;">{formatted.get("confidence_emoji", "")} {formatted.get("confidence", 0)*100:.0f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources if available
                    if formatted.get("sources") and len(formatted["sources"]) > 0:
                        with st.expander(f"📚 Sources ({len(formatted['sources'])})"):
                            for source in formatted["sources"]:
                                st.markdown(f"• {source}")
                    
                    # Debug info if enabled
                    if st.session_state.debug_mode and message.get("debug_info"):
                        with st.expander("🔍 Debug Info"):
                            st.json(message["debug_info"])
        
        # Query input
        st.subheader("❓ Ask a Question")
        query = st.text_area(
            "Enter your question about HR, IT, Operations, or Company policies:",
            placeholder="e.g., How do I apply for leave? Where is the API documentation? What are the wellness benefits?",
            height=100,
            key="query_input"
        )
        
        # Submit button with rate limiting
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            submit_disabled = False
            current_time = time.time()
            time_since_last = current_time - st.session_state.last_query_time
            
            # Rate limiting: 10 seconds between queries
            if time_since_last < 10 and st.session_state.last_query_time > 0:
                submit_disabled = True
                cooldown = 10 - int(time_since_last)
                button_label = f"⏳ Wait {cooldown}s"
            else:
                button_label = "🚀 Submit Query"
            
            if st.button(button_label, disabled=submit_disabled, use_container_width=True):
                if query.strip():
                    process_query(query, selected_dept, top_k, similarity_thresh)
        with col_btn2:
            if st.button("💡 Example Questions", use_container_width=True):
                show_example_questions()
        
        # Rate limiting notice
        if submit_disabled:
            st.caption(f"⏸️ Rate limiting: Please wait {10 - int(time_since_last)} seconds before next query")
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if st.session_state.conversation_history:
            # Calculate some stats
            total_queries = len([m for m in st.session_state.conversation_history if m["role"] == "user"])
            responses_with_sources = len([
                m for m in st.session_state.conversation_history 
                if m.get("role") == "assistant" and m.get("response", {}).get("sources")
            ])
            
            st.metric("Total Queries", total_queries)
            st.metric("Sourced Responses", responses_with_sources)
            
            # Last response confidence
            if st.session_state.conversation_history and st.session_state.conversation_history[-1]["role"] == "assistant":
                last_response = st.session_state.conversation_history[-1]["response"]
                if isinstance(last_response, dict):
                    confidence = last_response.get("confidence", 0)
                    confidence_pct = f"{confidence:.0%}"
                    if confidence > 0.8:
                        st.markdown(f'<p class="confidence-high">Last Confidence: {confidence_pct}</p>', unsafe_allow_html=True)
                    elif confidence > 0.6:
                        st.markdown(f'<p class="confidence-medium">Last Confidence: {confidence_pct}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="confidence-low">Last Confidence: {confidence_pct}</p>', unsafe_allow_html=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh Components", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.rag_chain = None
            st.rerun()
        
        if st.button("📁 View All Documents", use_container_width=True):
            show_all_documents()

def process_query(query, department, top_k, similarity_thresh):
    """Process user query and generate response"""
    # Update query time
    st.session_state.last_query_time = time.time()
    
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": query,
        "timestamp": time.time()
    })
    
    # Create a placeholder for the assistant response
    response_placeholder = st.empty()
    with response_placeholder.container():
        with st.spinner("🔍 Searching documents..."):
            try:
                # Apply custom similarity threshold
                st.session_state.retriever.similarity_threshold = similarity_thresh
                
                # Retrieve relevant documents
                department_filter = None if department == "All" else department
                retrieved_docs = st.session_state.retriever.retrieve(
                    query=query,
                    department=department_filter,
                    k=top_k
                )
                
                # Format context
                context = st.session_state.retriever.format_context(retrieved_docs)
                
                # Show retrieval results in debug mode
                debug_info = {}
                if st.session_state.debug_mode:
                    debug_info = {
                        "query": query,
                        "retrieved_count": len(retrieved_docs),
                        "documents": [
                            {
                                "filename": doc["metadata"].get("filename"),
                                "score": doc["score"],
                                "preview": doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
                            }
                            for doc in retrieved_docs[:3]
                        ],
                        "context_length": len(context)
                    }
                
                # Generate response
                with st.spinner("🤖 Generating response..."):
                    response = st.session_state.rag_chain.generate_response(
                        query=query,
                        context=context,
                        conversation_history=st.session_state.conversation_history[:-1]
                    )
                
                # Add debug info to response
                if debug_info:
                    response["debug_info"] = debug_info
                
                # Add assistant response to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": query,
                    "response": response,
                    "retrieved_docs": len(retrieved_docs),
                    "timestamp": time.time(),
                    **({"debug_info": debug_info} if debug_info else {})
                })
                
                # Rerun to update UI
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                error_response = {
                    "response": f"Sorry, I encountered an error: {str(e)[:100]}...",
                    "sources": [],
                    "confidence": 0.0
                }
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": query,
                    "response": error_response,
                    "timestamp": time.time()
                })
                st.rerun()

def test_retrieval(query, top_k, department):
    """Test document retrieval directly"""
    with st.spinner("Testing retrieval..."):
        try:
            retrieved = st.session_state.retriever.retrieve(query, department, top_k)
            st.success(f"✅ Retrieved {len(retrieved)} documents")
            
            for i, doc in enumerate(retrieved):
                with st.expander(f"Document {i+1}: {doc['metadata'].get('filename', 'Unknown')} (Score: {doc['score']:.3f})"):
                    st.write(f"**Department:** {doc['metadata'].get('department', 'Unknown')}")
                    st.write(f"**Text preview:**")
                    st.text(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
        except Exception as e:
            st.error(f"❌ Retrieval test failed: {e}")

def test_ollama_connection():
    """Test Ollama connection"""
    with st.spinner("Testing Ollama connection..."):
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
                num_predict=50
            )
            response = llm.invoke("Say 'Hello' if you're working.")
            st.success(f"✅ Ollama is connected! Response: '{response.content}'")
        except Exception as e:
            st.error(f"❌ Ollama connection failed: {e}")
            st.info("Make sure Ollama is running with: `ollama serve`")

def show_raw_chunks():
    """Show raw chunks from database"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection("enterprise_docs")
        
        results = collection.get(include=["documents", "metadatas"], limit=20)
        
        st.write(f"**Total chunks in database:** {collection.count()}")
        
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            with st.expander(f"Chunk {i+1}: {meta.get('filename', 'N/A')}"):
                st.write(f"**Department:** {meta.get('department', 'N/A')}")
                st.write(f"**Text:**")
                st.text(doc[:800] + "..." if len(doc) > 800 else doc)
    except Exception as e:
        st.error(f"Failed to show chunks: {e}")

def show_all_documents():
    """Show all available documents"""
    import os
    from pathlib import Path
    
    st.subheader("📄 Available Documents")
    
    for dept in ["hr", "it_tech", "operations", "company_wide"]:
        dept_path = settings.RAW_DOCS_DIR / dept
        if dept_path.exists():
            files = list(dept_path.glob("*.pdf"))
            if files:
                st.write(f"**{dept.upper()}:**")
                for file in files:
                    st.write(f"• {file.name}")

def show_example_questions():
    """Display example questions"""
    examples = [
        "How do I apply for annual leave?",
        "What is the process for software development?",
        "Where can I find API documentation?",
        "What wellness benefits are available?",
        "How does the procurement process work?",
        "What is the code of conduct policy?",
        "How do I report a security incident?",
        "What are the learning and development opportunities?"
    ]
    
    # Store the selected example
    selected_example = st.selectbox("Choose an example question:", examples)
    
    # Update the query input with selected example
    st.session_state.query_input = selected_example
    st.rerun()

# Run the app
if __name__ == "__main__":
    main()