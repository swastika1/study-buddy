#!/usr/bin/env python3

"""
Interactive study buddy application.
Run this for a chat-based study session with your documents.
Supports multiple formats: PDF, DOCX, TXT, MD, and more!
"""

import os
import sys
import boto3
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

from utils import (
    load_all_pdfs_from_directory, 
    build_qa_chain, 
    build_quiz_chain,
    build_quiz_chain_for_file,
    get_available_files
)

# Check for multi-format support
try:
    from multi_format_loader import get_supported_formats
    MULTI_FORMAT_AVAILABLE = True
    SUPPORTED_FORMATS = get_supported_formats()
except ImportError:
    MULTI_FORMAT_AVAILABLE = False
    SUPPORTED_FORMATS = {'pdf': 'PDF documents'}

# Load environment variables from .env file
load_dotenv()


def main():
    print("=" * 60)
    print("📚 Study Buddy - Your AI Learning Assistant")
    print("=" * 60)
    
    # Show supported formats
    if MULTI_FORMAT_AVAILABLE:
        format_list = ", ".join([f".{ext}" for ext in sorted(SUPPORTED_FORMATS.keys())])
        print(f"📄 Supported formats: {format_list}")
    else:
        print("📄 Supported formats: PDF only")
        print("💡 Install optional packages for more formats (see MULTI_FORMAT_GUIDE.md)")
    print()

    # 1. Setup AWS Bedrock client
    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
    except Exception as e:
        print(f"❌ Error setting up AWS Client: {e}")
        print("\n💡 Make sure you have configured AWS credentials:")
        print("   Option 1: Run 'aws configure' to set up credentials")
        print("   Option 2: Create a .env file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)

    # 2. Ask user for document location
    print("📂 Document Source Options:")
    print("   1. Use documents from 'files/' folder (default)")
    print("   2. Specify custom path(s)")
    print()
    
    choice = input("Select option (1-2, press Enter for default): ").strip() or "1"
    
    vectorstore = None
    
    if choice == "1":
        # Use files/ directory
        print("\n📂 Looking for PDFs in 'files/' directory...\n")
        try:
            vectorstore = load_all_pdfs_from_directory("files", bedrock_client)
            if not vectorstore:
                print("\n⚠️  No PDFs found in 'files/' folder.")
                print("💡 Add your PDF files to the 'files/' folder or choose option 2")
                return
        except Exception as e:
            print(f"❌ Error processing PDFs: {e}")
            return
    
    elif choice == "2":
        # Custom path(s)
        print("\n📄 Enter PDF path(s):")
        print("   • For single file: /path/to/document.pdf")
        print("   • For directory: /path/to/folder/")
        print("   • For multiple files: /path/file1.pdf,/path/file2.pdf (comma-separated)")
        print()
        
        paths = input("PDF path(s): ").strip()
        
        if not paths:
            print("❌ No path provided. Exiting.")
            return
        
        # Check if it's a directory
        if os.path.isdir(paths):
            print(f"\n📂 Loading all PDFs from directory: {paths}\n")
            try:
                vectorstore = load_all_pdfs_from_directory(paths, bedrock_client)
            except Exception as e:
                print(f"❌ Error processing directory: {e}")
                return
        
        # Check if it's multiple comma-separated paths
        elif "," in paths:
            pdf_files = [p.strip() for p in paths.split(",")]
            print(f"\n� Loading {len(pdf_files)} PDF file(s)...\n")
            
            all_docs = []
            for pdf_path in pdf_files:
                if not os.path.exists(pdf_path):
                    print(f"⚠️  File not found: {pdf_path}")
                    continue
                
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = os.path.basename(pdf_path)
                    all_docs.extend(docs)
                    print(f"   ✓ Loaded {os.path.basename(pdf_path)}: {len(docs)} pages")
                except Exception as e:
                    print(f"   ✗ Error loading {pdf_path}: {e}")
            
            if not all_docs:
                print("❌ No documents loaded successfully")
                return
            
            # Process all docs
            print(f"\n✂️  Splitting text into chunks...")
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            print(f"   ✓ Created {len(splits)} text chunks")
            
            print("🧠 Generating embeddings...")
            from langchain_aws import BedrockEmbeddings
            from langchain_community.vectorstores import FAISS
            embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
            vectorstore = FAISS.from_documents(splits, embeddings)
            print("✅ All PDFs processed!\n")
        
        # Single file
        else:
            if not os.path.exists(paths):
                print(f"❌ File not found: {paths}")
                return
            
            print(f"\n📄 Loading PDF: {os.path.basename(paths)}\n")
            try:
                vectorstore = load_and_process_pdf(paths, bedrock_client)
            except Exception as e:
                print(f"❌ Error processing PDF: {e}")
                return
    
    else:
        print("❌ Invalid option. Exiting.")
        return
    
    if not vectorstore:
        print("❌ Failed to create knowledge base")
        return

    # 3. Setup LLM
    llm = ChatBedrock(
        model_id="amazon.nova-lite-v1:0",
        client=bedrock_client,
        model_kwargs={"temperature": 0.1, "max_tokens": 2048}
    )

    # 4. Build chains
    qa_chain = build_qa_chain(llm, vectorstore)
    quiz_chain = build_quiz_chain(llm, vectorstore)

    # 5. Interactive chat loop
    print("💬 Study session started!")
    print("\n📖 Available Commands:")
    print("  • Ask any question about your documents")
    print("  • Type 'quiz' or 'quiz N' to generate practice questions (from all files)")
    print("  • Type 'quiz file' to interactively select a file for quiz")
    print("  • Type 'quiz file <filename>' to quiz from specific file directly")
    print("  • Type 'files' to see available files")
    print("  • Type 'exit' or 'quit' to end the session")
    print("-" * 60)
    print()

    while True:
        try:
            query = input("🎓 You: ").strip()

            if not query:
                continue

            # Exit command
            if query.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Happy studying! Good luck!")
                break

            # Show available files
            if query.lower() == "files":
                available_files = get_available_files(vectorstore)
                if available_files:
                    print(f"\n📚 Available files ({len(available_files)}):")
                    for i, file in enumerate(available_files, 1):
                        print(f"   {i}. {file}")
                    print()
                else:
                    print("\n⚠️  Could not retrieve file list\n")
                continue

            # Quiz generation from specific file (with filename or interactive)
            if query.lower().startswith("quiz file"):
                available_files = get_available_files(vectorstore)
                
                if not available_files:
                    print("\n⚠️  Could not retrieve file list. Using general quiz instead.\n")
                    continue
                
                # Check if filename was provided after "quiz file"
                parts = query.split(maxsplit=2)  # Split into ["quiz", "file", "filename"]
                
                if len(parts) > 2:
                    # Filename provided directly
                    provided_filename = parts[2].strip()
                    
                    # Try to find matching file (support partial matching)
                    matched_files = [f for f in available_files if provided_filename.lower() in f.lower()]
                    
                    if len(matched_files) == 1:
                        selected_file = matched_files[0]
                        print(f"\n📄 Selected file: {selected_file}")
                    elif len(matched_files) > 1:
                        print(f"\n⚠️  Multiple files match '{provided_filename}':")
                        for i, file in enumerate(matched_files, 1):
                            print(f"   {i}. {file}")
                        print()
                        file_choice = input("Enter file number: ").strip()
                        
                        if not file_choice.isdigit() or int(file_choice) < 1 or int(file_choice) > len(matched_files):
                            print("❌ Invalid selection\n")
                            continue
                        
                        selected_file = matched_files[int(file_choice) - 1]
                    else:
                        print(f"\n❌ No file found matching '{provided_filename}'")
                        print(f"💡 Available files:")
                        for i, file in enumerate(available_files, 1):
                            print(f"   {i}. {file}")
                        print("\n💡 Try: quiz file <exact_filename> or just 'quiz file' for interactive selection\n")
                        continue
                else:
                    # Interactive selection
                    if len(available_files) == 1:
                        print(f"\n📄 Only one file available: {available_files[0]}")
                        selected_file = available_files[0]
                    else:
                        print(f"\n📚 Select a file to quiz from:")
                        for i, file in enumerate(available_files, 1):
                            print(f"   {i}. {file}")
                        print()
                        
                        file_choice = input("Enter file number: ").strip()
                        
                        if not file_choice.isdigit() or int(file_choice) < 1 or int(file_choice) > len(available_files):
                            print("❌ Invalid selection\n")
                            continue
                        
                        selected_file = available_files[int(file_choice) - 1]
                
                # Ask for number of questions
                num_input = input(f"Number of questions (press Enter for 5): ").strip()
                num_questions = int(num_input) if num_input.isdigit() else 5
                
                print(f"\n🤖 Generating {num_questions} questions from '{selected_file}'...\n")
                
                try:
                    file_quiz_chain = build_quiz_chain_for_file(llm, vectorstore, selected_file, num_questions)
                    result = file_quiz_chain.invoke(str(num_questions))
                    print(f"📝 Quiz from: {selected_file}\n")
                    print(result)
                    print()
                except Exception as e:
                    print(f"❌ Error generating quiz: {e}\n")
                
                continue

            # Quiz generation (all files)
            if query.lower().startswith("quiz"):
                parts = query.split()
                num_questions = 5  # default

                if len(parts) > 1 and parts[1].isdigit():
                    num_questions = int(parts[1])

                print(f"\n🤖 Generating {num_questions} practice questions (from all files)...\n")

                try:
                    result = quiz_chain.invoke(str(num_questions))
                    print(result)
                    print()
                except Exception as e:
                    print(f"❌ Error generating quiz: {e}\n")

                continue

            # Regular Q&A
            print("🤔 Thinking...\n")

            try:
                response = qa_chain.invoke({"input": query})
                print("🤖 Assistant:\n")
                print(response["answer"])
                print()
            except Exception as e:
                print(f"❌ Error: {e}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
