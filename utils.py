
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Multi-format document loader
try:
    from multi_format_loader import (
        load_all_documents_from_directory,
        print_supported_formats,
        get_supported_formats,
    )
    MULTI_FORMAT_AVAILABLE = True
except ImportError:
    MULTI_FORMAT_AVAILABLE = False

# --- Configuration ---
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
REGION_NAME = "us-east-1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Global vectorstore cache
_vectorstore = None
_current_pdf = None
_files_directory = "files"


def load_all_pdfs_from_directory(directory: str, bedrock_client) -> Optional[FAISS]:
    """Loads all supported documents from a directory and creates a single vector store.
    
    Now supports multiple formats: PDF, DOCX, TXT, MD, CSV, HTML, JSON, PPTX, XLSX
    (depending on installed optional packages)
    """
    global _vectorstore, _current_pdf
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"⚠️ Directory '{directory}' not found. Creating it...")
        os.makedirs(directory, exist_ok=True)
        return None
    
    # Use multi-format loader if available, otherwise fallback to PDF-only
    if MULTI_FORMAT_AVAILABLE:
        print("🎨 Multi-format document support enabled!")
        all_docs = load_all_documents_from_directory(directory)
    else:
        # Fallback to PDF-only loading
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in '{directory}/' directory")
            print(f"💡 Please add PDF files to the '{directory}/' folder and restart")
            return None
        
        print(f"📚 Found {len(pdf_files)} PDF file(s) in '{directory}/':")
        for pdf in pdf_files:
            print(f"   • {pdf}")
        
        print(f"\n🔄 Loading and processing all PDFs...")
        
        all_docs = []
        
        # Load each PDF
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            print(f"\n📄 Processing: {pdf_file}")
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file
                
                all_docs.extend(docs)
                print(f"   ✓ Loaded {len(docs)} pages")
            except Exception as e:
                print(f"   ✗ Error loading {pdf_file}: {e}")
                continue
    
    if not all_docs:
        print("❌ No documents loaded successfully")
        return None
    
    print(f"\n📊 Total document sections loaded: {len(all_docs)}")
    
    # Split text
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"   ✓ Created {len(splits)} text chunks")
    
    # Create embeddings
    print("🧠 Generating embeddings (this may take a moment)...")
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_client
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ All documents processed and indexed!\n")
    
    # Cache the vectorstore
    _vectorstore = vectorstore
    _current_pdf = directory
    
    return vectorstore


def load_and_process_pdf(file_path: str, bedrock_client) -> Optional[FAISS]:
    """Loads PDF, splits text, and creates vector store."""
    global _vectorstore, _current_pdf
    
    # Return cached vectorstore if same PDF
    if _current_pdf == file_path and _vectorstore is not None:
        return _vectorstore
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    print(f"📄 Loading and processing: {file_path}...")
    
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"   - Loaded {len(docs)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"   - Split into {len(splits)} text chunks.")

    # 3. Embeddings & Vector Store
    print("   - Generating embeddings...")
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_client
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ Processing complete!")
    
    # Cache the vectorstore
    _vectorstore = vectorstore
    _current_pdf = file_path
    
    return vectorstore


def build_qa_chain(llm, vectorstore):
    """Builds a RAG chain for question answering with source citations."""
    
    qa_prompt = ChatPromptTemplate.from_template("""
    You are a helpful study assistant. Use the following pieces of context to answer the question at the end.
    If the answer is not in the context, say "I couldn't find the answer in the notes provided."
    
    Be clear, concise, and educational in your responses.
    
    IMPORTANT: After providing your answer, you MUST cite your sources by listing:
    - Source filename(s)
    - Page number(s) if available
    
    Format your response as:
    [Your detailed answer here]
    
    📚 Sources:
    - Filename: [source_file], Page: [page number]
    (Include all relevant sources used)
    
    <context>
    {context}
    </context>

    Question: {input}
    
    Answer:""")

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


def build_quiz_chain(llm, vectorstore, num_questions: int = 5):
    """Builds a chain for generating multiple-choice questions."""
    
    quiz_prompt = ChatPromptTemplate.from_template("""
    You are a test generator. Based on the following content, create {num_questions} multiple-choice questions.
    
    For each question:
    - Write a clear question based on the content
    - Provide 4 options (A, B, C, D)
    - Mark the correct answer
    - Include a brief explanation
    
    Format each question like this:
    
    Q1: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Letter]
    Explanation: [Brief explanation]
    
    <context>
    {context}
    </context>
    
    Generate {num_questions} multiple-choice questions:""")

    # Get relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    # Build chain that properly handles the input
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke("summary overview main topics")), 
            "num_questions": lambda x: x
        }
        | quiz_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def get_available_files(vectorstore) -> list:
    """Get list of unique source files in the vectorstore."""
    try:
        # Get all documents from vectorstore
        all_docs = vectorstore.docstore._dict.values()
        
        # Extract unique source files
        source_files = set()
        for doc in all_docs:
            if "source_file" in doc.metadata:
                source_files.add(doc.metadata["source_file"])
        
        return sorted(list(source_files))
    except:
        return []


def build_quiz_chain_for_file(llm, vectorstore, source_file: str, num_questions: int = 5):
    """Builds a chain for generating multiple-choice questions from a specific file."""
    
    quiz_prompt = ChatPromptTemplate.from_template("""
    You are a test generator. Based on the following content from "{source_file}", create {num_questions} multiple-choice questions.
    
    For each question:
    - Write a clear question based on the content
    - Provide 4 options (A, B, C, D)
    - Mark the correct answer
    - Include a brief explanation
    
    Format each question like this:
    
    Q1: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Letter]
    Explanation: [Brief explanation]
    
    <context>
    {context}
    </context>
    
    Generate {num_questions} multiple-choice questions:""")

    # Create a filtered retriever for specific file
    def get_file_specific_docs(query):
        # Get more documents to ensure we have enough from the specific file
        all_results = vectorstore.similarity_search(query, k=20)
        
        # Filter for specific file
        file_docs = [doc for doc in all_results if doc.metadata.get("source_file") == source_file]
        
        # Take top 8 from the specific file
        return file_docs[:8]
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": lambda x: format_docs(get_file_specific_docs("summary overview main topics")), 
            "num_questions": lambda x: x,
            "source_file": lambda x: source_file
        }
        | quiz_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
