"""
ingestion.py — Document Parsing, Chunking, Embedding & ChromaDB Storage
========================================================================
Responsibilities:
  1. Validate file extensions (.pdf, .txt, .md)
  2. Parse PDFs with PyMuPDF (page-level text extraction)
  3. Parse .txt files with paragraph-based section breakdown
  4. Parse .md files with heading-based section extraction
  5. Split text into overlapping chunks via LangChain RecursiveCharacterTextSplitter
  6. Embed chunks using Gemini text-embedding-001
  7. Store in ChromaDB with metadata: doc_id, filename, location, chunk_index
  8. Support document deletion and listing

Key Design Decisions:
  - chunk_size=900, chunk_overlap=150: balances context window vs retrieval precision
  - Each chunk carries doc_id metadata so documents can be cleanly removed
  - ChromaDB PersistentClient stores data locally in ./chroma_data/
  - Embedding function wraps Gemini with a .name() method (required by ChromaDB)
  - Session isolation: collection names include session_id to prevent data bleed
  - .txt files split by double-newline paragraphs for section-aware citations
"""

import os
import uuid
from typing import List, Dict, Tuple, Optional

import pymupdf as fitz  # PyMuPDF — fast, reliable PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 900       # Max characters per chunk
CHUNK_OVERLAP = 150    # Overlap between consecutive chunks (preserves context)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")


# ---------------------------------------------------------------------------
# Embeddings — Gemini text-embedding-001
# ---------------------------------------------------------------------------

def get_embeddings_model(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Create a Gemini embeddings model instance.
    Uses models/gemini-embedding-001 (3072-dim vectors, cosine similarity).
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    ChromaDB-compatible embedding function wrapper around Gemini.
    
    ChromaDB requires embedding functions to have a __call__ and .name() method.
    This class wraps GoogleGenerativeAIEmbeddings to satisfy that interface.
    """

    def __init__(self, api_key: str):
        self._embeddings = get_embeddings_model(api_key)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of text strings into vectors."""
        return self._embeddings.embed_documents(input)

    def name(self) -> str:
        """Return the embedding model name (required by ChromaDB)."""
        return "gemini-embedding-001"


# ---------------------------------------------------------------------------
# ChromaDB Collection (Session-Isolated)
# ---------------------------------------------------------------------------

def get_chroma_collection(api_key: str, session_id: str = "default") -> chromadb.Collection:
    """
    Get or create a ChromaDB collection scoped to a session.
    
    Uses PersistentClient for local file-based storage (survives restarts).
    Collection name includes session_id to prevent data bleed between users/sessions.
    Collection uses cosine similarity for vector search.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = GeminiEmbeddingFunction(api_key)

    collection_name = f"docs_{session_id}"

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# File Validation
# ---------------------------------------------------------------------------

def validate_file(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the file extension is supported.
    Returns (is_valid, error_message).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, None


# ---------------------------------------------------------------------------
# PDF Parsing (PyMuPDF)
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF using PyMuPDF.
    
    Returns a list of dicts: {content, location, page_num}
    Empty/skipped pages are filtered out.
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text("text")  # Extract selectable text only
            if text and text.strip():
                pages.append({
                    "content": text.strip(),
                    "location": f"Page {i + 1}",
                    "page_num": i + 1,
                })
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")
    return pages


# ---------------------------------------------------------------------------
# Text/Markdown Parsing
# ---------------------------------------------------------------------------

def parse_text_md(file_path: str, filename: str) -> List[Dict]:
    """
    Parse .txt or .md files.
    - .md files: split by headings (# / ##) to preserve section context
    - .txt files: split by paragraphs (double newlines) for section-aware citations
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 for non-UTF-8 files
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

    if not content or not content.strip():
        raise ValueError("File is empty")

    if filename.endswith(".md"):
        return _split_markdown(content)
    else:
        return _split_text_by_paragraphs(content, filename)


def _split_markdown(content: str) -> List[Dict]:
    """
    Split markdown content by headings (# / ##) to create section-aware chunks.
    Each section becomes a separate page/section for citation purposes.
    """
    sections = []
    current_heading = "Introduction"
    current_text = []

    for line in content.split("\n"):
        if line.startswith("#"):
            # New heading found — save previous section
            if current_text:
                text = "\n".join(current_text).strip()
                if text:
                    sections.append({
                        "content": text,
                        "location": current_heading,
                        "page_num": None,
                    })
                current_text = []
            current_heading = line.lstrip("#").strip()
        else:
            current_text.append(line)

    # Save the last section
    if current_text:
        text = "\n".join(current_text).strip()
        if text:
            sections.append({
                "content": text,
                "location": current_heading,
                "page_num": None,
            })

    # Fallback: if no headings found, treat as single section
    if not sections and content.strip():
        sections.append({
            "content": content.strip(),
            "location": "Document",
            "page_num": None,
        })

    return sections


def _split_text_by_paragraphs(content: str, filename: str) -> List[Dict]:
    """
    Split plain text by paragraphs (double newlines).
    Each paragraph group becomes a labeled section for citation.
    
    This prevents the entire .txt file from being treated as a single location,
    enabling more precise source attribution.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        # Single paragraph or no double-newlines — treat as one section
        return [{"content": content.strip(), "location": "Section 1", "page_num": None}]

    sections = []
    for i, para in enumerate(paragraphs, 1):
        # Use first line as section label if it's short (like a heading)
        first_line = para.split("\n")[0].strip()
        if len(first_line) < 80 and len(first_line) > 3:
            location = first_line
        else:
            location = f"Section {i}"

        sections.append({
            "content": para,
            "location": location,
            "page_num": None,
        })

    return sections


# ---------------------------------------------------------------------------
# Text Chunking (LangChain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------

def chunk_text(pages: List[Dict]) -> List[Dict]:
    """
    Split page/section text into overlapping chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split on:
      \n\n → \n → ". " → " " → ""
    This preserves natural paragraph and sentence boundaries.
    
    Each chunk carries: content, location, page_num, chunk_index
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["content"])
        for i, split in enumerate(splits):
            chunks.append({
                "content": split,
                "location": page["location"],
                "page_num": page["page_num"],
                "chunk_index": i,
            })

    return chunks


# ---------------------------------------------------------------------------
# Document ID Generation
# ---------------------------------------------------------------------------

def generate_doc_id() -> str:
    """Generate a short, unique document ID (first 8 chars of UUID4)."""
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Full Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    filename: str,
    api_key: str,
    collection: chromadb.Collection,
) -> Tuple[bool, str, int]:
    """
    Ingest a single file into the vector store.
    
    Pipeline: validate → parse → chunk → embed → store in ChromaDB
    
    Returns: (success, result_or_error, chunk_count)
    """
    # Step 1: Validate extension
    valid, error = validate_file(filename)
    if not valid:
        return False, error, 0

    # Step 2: Parse file into pages/sections
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".pdf":
            pages = parse_pdf(file_path)
        else:
            pages = parse_text_md(file_path, filename)
    except ValueError as e:
        return False, str(e), 0
    except Exception as e:
        return False, f"Unexpected error parsing {filename}: {e}", 0

    if not pages:
        return False, f"File '{filename}' is empty or contains no readable text", 0

    # Step 3: Split into chunks
    chunks = chunk_text(pages)
    if not chunks:
        return False, f"File '{filename}' produced no chunks after splitting", 0

    # Step 4: Generate unique doc_id for this upload
    doc_id = generate_doc_id()

    # Step 5: Prepare ChromaDB data
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append({
            "doc_id": doc_id,
            "filename": filename,
            "location": chunk["location"],
            "chunk_index": i,
        })

    # Step 6: Batch-insert into ChromaDB (100 at a time to avoid memory spikes)
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return True, doc_id, len(chunks)


# ---------------------------------------------------------------------------
# Document Management
# ---------------------------------------------------------------------------

def remove_document(doc_id: str, collection: chromadb.Collection) -> bool:
    """
    Remove all chunks for a document from ChromaDB.
    Uses doc_id filter to delete all associated chunks.
    """
    try:
        collection.delete(where={"doc_id": doc_id})
        return True
    except Exception:
        return False


def list_documents(collection: chromadb.Collection) -> List[Dict]:
    """
    List all unique documents in the collection with their chunk counts.
    Returns: [{doc_id, filename, chunk_count}, ...]
    """
    try:
        results = collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []

        # Group chunks by doc_id and count them
        doc_map = {}
        for meta in results["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "chunk_count": 0,
                }
            doc_map[doc_id]["chunk_count"] += 1

        return list(doc_map.values())
    except Exception:
        return []
