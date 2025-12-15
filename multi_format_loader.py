"""
Multi-Format Document Loader
Supports: PDF, Word (DOCX), Text (TXT/MD), CSV, JSON, HTML, and more
"""

import os
from typing import List, Optional, Dict
from pathlib import Path

from langchain_core.documents import Document

# Track available loaders
AVAILABLE_FORMATS = {
    'pdf': 'PDF documents',
    'txt': 'Plain text files',
    'md': 'Markdown files',
}

OPTIONAL_FORMATS = {}

# Try importing optional loaders
try:
    from langchain_community.document_loaders import Docx2txtLoader
    OPTIONAL_FORMATS['docx'] = 'Microsoft Word documents'
except ImportError:
    pass

try:
    from langchain_community.document_loaders import CSVLoader
    OPTIONAL_FORMATS['csv'] = 'CSV spreadsheets'
except ImportError:
    pass

try:
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    OPTIONAL_FORMATS['html'] = 'HTML web pages'
    OPTIONAL_FORMATS['htm'] = 'HTML web pages'
except ImportError:
    pass

try:
    from langchain_community.document_loaders import JSONLoader
    OPTIONAL_FORMATS['json'] = 'JSON data files'
except ImportError:
    pass

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    OPTIONAL_FORMATS['pptx'] = 'PowerPoint presentations'
except ImportError:
    pass

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    OPTIONAL_FORMATS['xlsx'] = 'Excel spreadsheets'
    OPTIONAL_FORMATS['xls'] = 'Excel spreadsheets'
except ImportError:
    pass

# Merge available formats
AVAILABLE_FORMATS.update(OPTIONAL_FORMATS)


def get_supported_formats() -> Dict[str, str]:
    """Return dictionary of supported file formats."""
    return AVAILABLE_FORMATS.copy()


def get_format_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return list(AVAILABLE_FORMATS.keys())


def load_pdf_document(file_path: str) -> List[Document]:
    """Load PDF document."""
    from langchain_community.document_loaders import PyPDFLoader
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"   ⚠️ Error loading PDF: {e}")
        return []


def load_text_document(file_path: str) -> List[Document]:
    """Load plain text or markdown document."""
    from langchain_community.document_loaders import TextLoader
    
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        return docs
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            loader = TextLoader(file_path, encoding='latin-1')
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"   ⚠️ Error loading text file: {e}")
            return []
    except Exception as e:
        print(f"   ⚠️ Error loading text file: {e}")
        return []


def load_docx_document(file_path: str) -> List[Document]:
    """Load Microsoft Word document."""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return docs
    except ImportError:
        print(f"   ⚠️ DOCX support not available. Install: pip install docx2txt")
        return []
    except Exception as e:
        print(f"   ⚠️ Error loading DOCX: {e}")
        return []


def load_csv_document(file_path: str) -> List[Document]:
    """Load CSV document."""
    try:
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file_path)
        docs = loader.load()
        return docs
    except ImportError:
        print(f"   ⚠️ CSV support available by default")
        return []
    except Exception as e:
        print(f"   ⚠️ Error loading CSV: {e}")
        return []


def load_html_document(file_path: str) -> List[Document]:
    """Load HTML document."""
    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()
        return docs
    except ImportError:
        print(f"   ⚠️ HTML support not available. Install: pip install unstructured")
        return []
    except Exception as e:
        print(f"   ⚠️ Error loading HTML: {e}")
        return []


def load_json_document(file_path: str) -> List[Document]:
    """Load JSON document."""
    try:
        from langchain_community.document_loaders import JSONLoader
        
        # JSONLoader requires jq_schema to extract text
        # Try to load entire JSON as text
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to readable text
        text_content = json.dumps(data, indent=2)
        
        doc = Document(
            page_content=text_content,
            metadata={"source": file_path, "format": "json"}
        )
        return [doc]
    except Exception as e:
        print(f"   ⚠️ Error loading JSON: {e}")
        return []


def load_pptx_document(file_path: str) -> List[Document]:
    """Load PowerPoint document."""
    try:
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(file_path)
        docs = loader.load()
        return docs
    except ImportError:
        print(f"   ⚠️ PPTX support not available. Install: pip install unstructured python-pptx")
        return []
    except Exception as e:
        print(f"   ⚠️ Error loading PPTX: {e}")
        return []


def load_excel_document(file_path: str) -> List[Document]:
    """Load Excel document."""
    try:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file_path)
        docs = loader.load()
        return docs
    except ImportError:
        print(f"   ⚠️ Excel support not available. Install: pip install unstructured openpyxl")
        return []
    except Exception as e:
        print(f"   ⚠️ Error loading Excel: {e}")
        return []


def load_document_by_extension(file_path: str) -> List[Document]:
    """
    Load document based on file extension.
    
    Args:
        file_path: Path to the document
    
    Returns:
        List of Document objects
    """
    ext = Path(file_path).suffix.lower().lstrip('.')
    
    loaders = {
        'pdf': load_pdf_document,
        'txt': load_text_document,
        'md': load_text_document,
        'markdown': load_text_document,
        'docx': load_docx_document,
        'csv': load_csv_document,
        'html': load_html_document,
        'htm': load_html_document,
        'json': load_json_document,
        'pptx': load_pptx_document,
        'xlsx': load_excel_document,
        'xls': load_excel_document,
    }
    
    loader_func = loaders.get(ext)
    
    if loader_func is None:
        print(f"   ⚠️ Unsupported file format: .{ext}")
        print(f"   💡 Supported formats: {', '.join(get_format_extensions())}")
        return []
    
    return loader_func(file_path)


def load_all_documents_from_directory(directory: str) -> List[Document]:
    """
    Load all supported document formats from a directory.
    
    Args:
        directory: Path to the directory containing documents
    
    Returns:
        List of all loaded Document objects
    """
    if not os.path.exists(directory):
        print(f"⚠️ Directory '{directory}' not found")
        return []
    
    # Get all files with supported extensions
    supported_exts = get_format_extensions()
    all_files = []
    
    for ext in supported_exts:
        pattern = f"*.{ext}"
        files = list(Path(directory).glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        print(f"⚠️ No supported documents found in '{directory}/'")
        print(f"💡 Supported formats: {', '.join(supported_exts)}")
        return []
    
    # Group files by extension for display
    files_by_type = {}
    for file in all_files:
        ext = file.suffix.lower().lstrip('.')
        if ext not in files_by_type:
            files_by_type[ext] = []
        files_by_type[ext].append(file.name)
    
    print(f"📚 Found {len(all_files)} document(s) in '{directory}/':")
    for ext, files in sorted(files_by_type.items()):
        format_name = AVAILABLE_FORMATS.get(ext, ext.upper())
        print(f"\n   📄 {format_name} ({len(files)} file{'s' if len(files) != 1 else ''}):")
        for filename in sorted(files):
            print(f"      • {filename}")
    
    print(f"\n🔄 Loading and processing all documents...")
    
    all_docs = []
    success_count = 0
    
    # Load each file
    for file_path in sorted(all_files):
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            docs = load_document_by_extension(str(file_path))
            
            if docs:
                # Add source metadata
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["file_type"] = file_path.suffix.lower().lstrip('.')
                
                all_docs.extend(docs)
                success_count += 1
                
                # Show appropriate metric based on document type
                if file_path.suffix.lower() in ['.pdf']:
                    print(f"   ✓ Loaded {len(docs)} pages")
                elif file_path.suffix.lower() in ['.csv']:
                    print(f"   ✓ Loaded {len(docs)} rows")
                else:
                    print(f"   ✓ Loaded {len(docs)} section(s)")
            else:
                print(f"   ✗ No content extracted")
                
        except Exception as e:
            print(f"   ✗ Error loading {file_path.name}: {e}")
            continue
    
    print(f"\n📊 Successfully loaded: {success_count}/{len(all_files)} files")
    print(f"📊 Total document sections: {len(all_docs)}")
    
    return all_docs


def get_installation_guide() -> str:
    """Return installation guide for optional format support."""
    guide = """
╔════════════════════════════════════════════════════════════════╗
║         Multi-Format Document Support - Installation          ║
╚════════════════════════════════════════════════════════════════╝

📦 Currently Supported (Built-in):
   ✅ PDF documents (.pdf)
   ✅ Text files (.txt)
   ✅ Markdown (.md)

📦 Optional Format Support:

1️⃣  Microsoft Office Documents:
   pip install docx2txt python-pptx openpyxl
   
   Enables:
   • Word documents (.docx)
   • PowerPoint (.pptx)
   • Excel (.xlsx, .xls)

2️⃣  Web & Structured Data:
   pip install unstructured
   
   Enables:
   • HTML files (.html, .htm)
   • Additional format parsing

3️⃣  Data Files:
   pip install pandas
   
   Enables:
   • Better CSV handling (.csv)
   • Excel support (.xlsx, .xls)

4️⃣  Complete Support (All formats):
   pip install docx2txt python-pptx openpyxl unstructured pandas

╔════════════════════════════════════════════════════════════════╗
║  After installation, restart the app to use new formats!      ║
╚════════════════════════════════════════════════════════════════╝
"""
    return guide


def print_supported_formats():
    """Print currently supported file formats."""
    formats = get_supported_formats()
    
    print("\n" + "="*60)
    print("📚 SUPPORTED DOCUMENT FORMATS")
    print("="*60)
    
    for ext, description in sorted(formats.items()):
        status = "✅" if ext in AVAILABLE_FORMATS else "❌"
        print(f"  {status} .{ext:<6} - {description}")
    
    missing_count = len([f for f in ['docx', 'pptx', 'xlsx', 'html'] if f not in AVAILABLE_FORMATS])
    
    if missing_count > 0:
        print(f"\n💡 {missing_count} additional format(s) available with optional packages")
        print("   Run: python -c \"from multi_format_loader import get_installation_guide; print(get_installation_guide())\"")
    
    print("="*60 + "\n")
