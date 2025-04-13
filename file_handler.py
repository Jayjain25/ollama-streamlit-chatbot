import base64
import io
import streamlit as st
from PIL import Image # For image type validation/conversion if needed
from PyPDF2 import PdfReader # For PDF text extraction
# Basic RAG - Use a simple text splitting approach for now
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Basic RAG - Use a simple list store and exact match (or basic keyword) for retrieval
# More advanced: FAISS, ChromaDB, embeddings (SentenceTransformers)
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_NUM_RETRIEVED_CHUNKS
from config import ALLOWED_IMAGE_TYPES, ALLOWED_FILE_TYPES

# --- File Processing ---

def encode_image_to_base64(image_file):
    """Encodes an uploaded image file to base64."""
    try:
        # Read bytes and encode
        img_bytes = image_file.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        return encoded_string
    except Exception as e:
        st.error(f"Error encoding image '{image_file.name}': {e}", icon="🖼️")
        return None

def extract_pdf_text(pdf_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF '{pdf_file.name}': {e}", icon="📄")
        return None

def process_uploaded_files(uploaded_files):
    """
    Processes a list of uploaded files (images and PDFs).

    Returns:
        dict: {
            "images_base64": {filename: base64_string},
            "pdf_texts": {filename: extracted_text},
            "processed_info": {filename: {"type": "image/pdf", "size": size}}
        }
    """
    images_base64 = {}
    pdf_texts = {}
    processed_info = {}

    if not uploaded_files:
        return {"images_base64": {}, "pdf_texts": {}, "processed_info": {}}

    for file in uploaded_files:
        file_type = file.type.split('/')[-1].lower()
        file_ext = file.name.split('.')[-1].lower()

        if file_ext in ALLOWED_IMAGE_TYPES or file_type in ['png', 'jpeg', 'webp']:
            base64_img = encode_image_to_base64(file)
            if base64_img:
                images_base64[file.name] = base64_img
                processed_info[file.name] = {"type": "image", "size": file.size}
        elif file_ext == 'pdf' or file_type == 'pdf':
            text = extract_pdf_text(file)
            if text:
                pdf_texts[file.name] = text
                processed_info[file.name] = {"type": "pdf", "size": file.size}
        else:
            st.warning(f"Unsupported file type skipped: {file.name} ({file.type})", icon="⚠️")

    return {
        "images_base64": images_base64,
        "pdf_texts": pdf_texts,
        "processed_info": processed_info
    }


# --- Basic RAG Implementation ---

# In-memory storage for RAG chunks (per session)
# A more robust solution would use a vector database
rag_texts_store = {} # {chat_id: {"chunks": [chunk_text], "sources": [filename]}}
rag_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP
)

def initialize_rag_store(chat_id):
    """Initializes or clears the RAG store for a given chat ID."""
    global rag_texts_store
    rag_texts_store[chat_id] = {"chunks": [], "sources": []}
    print(f"RAG store initialized for chat {chat_id}")

def add_text_to_rag(chat_id, text, source_filename):
    """Splits text and adds chunks to the in-memory RAG store."""
    global rag_texts_store
    if not text:
        return
    if chat_id not in rag_texts_store:
        initialize_rag_store(chat_id)

    chunks = rag_splitter.split_text(text)
    rag_texts_store[chat_id]["chunks"].extend(chunks)
    rag_texts_store[chat_id]["sources"].extend([source_filename] * len(chunks))
    print(f"Added {len(chunks)} chunks from '{source_filename}' to RAG store for chat {chat_id}")


def get_rag_context(chat_id, query, num_chunks=RAG_NUM_RETRIEVED_CHUNKS):
    """
    Retrieves relevant text chunks based on a simple keyword match (basic example).
    A real implementation should use embeddings and vector similarity search.
    """
    global rag_texts_store
    if chat_id not in rag_texts_store or not rag_texts_store[chat_id]["chunks"]:
        return "", [] # No context available

    all_chunks = rag_texts_store[chat_id]["chunks"]
    all_sources = rag_texts_store[chat_id]["sources"]

    # --- Basic Keyword Matching Retrieval ---
    query_words = set(query.lower().split())
    scored_chunks = []
    for i, chunk in enumerate(all_chunks):
        chunk_words = set(chunk.lower().split())
        common_words = query_words.intersection(chunk_words)
        score = len(common_words) # Simple overlap score
        if score > 0:
            scored_chunks.append({"score": score, "index": i})

    # Sort by score (descending) and take top N
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_indices = [item["index"] for item in scored_chunks[:num_chunks]]

    # Get the actual chunk text and source filenames
    retrieved_chunks_text = [all_chunks[i] for i in top_indices]
    retrieved_sources = list(set([all_sources[i] for i in top_indices])) # Unique sources

    if not retrieved_chunks_text:
        return "", []

    # Format context for the LLM prompt
    context_str = "\n\n---\nRelevant Document Context:\n---\n"
    for i, chunk in enumerate(retrieved_chunks_text):
        source = all_sources[top_indices[i]]
        context_str += f"[Source: {source}]\n{chunk}\n\n"
    context_str += "---\nEnd of Context\n---\n"

    # print(f"Retrieved RAG context from sources: {retrieved_sources}") # Debug
    return context_str.strip(), retrieved_sources