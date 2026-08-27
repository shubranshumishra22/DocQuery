"""
retrieval.py — Query Processing, Scope Detection & Similarity Search
=====================================================================
Responsibilities:
  1. Rewrite follow-up questions into standalone form using chat history
  2. LLM-based scope detection: map natural language doc refs to exact filenames
  3. LLM-based comparison intent detection (not just keyword matching)
  4. Retrieve relevant chunks from ChromaDB
     - Default: per-doc balanced retrieval when multiple docs loaded
     - Scoped: filter to specific document
     - Comparison: explicitly retrieve top-k per document separately

Key Design Decisions:
  - Question rewriting + scope + intent all done in a SINGLE LLM call (cost-efficient)
  - Scope detection uses semantic matching, not just substring/regex
  - Multi-doc queries default to per-doc retrieval to prevent single-doc dominance
  - Relevance threshold is dynamic based on score distribution
"""

import json
import re
from typing import List, Dict, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

from prompts import REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base threshold — dynamically adjusted based on score distribution
BASE_RELEVANCE_THRESHOLD = 0.35

# Per-doc retrieval count when multiple documents are loaded
PER_DOC_K = 4

# Max total chunks to return
MAX_TOTAL_CHUNKS = 12


# ---------------------------------------------------------------------------
# Helper: Text Extraction
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """
    Extract text from LLM response, handling both string and list formats.
    Some Gemini models return content as [{type: "text", text: "..."}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
        return str(content)
    return str(content)


# ---------------------------------------------------------------------------
# Helper: Format Chat History
# ---------------------------------------------------------------------------

def _format_chat_history(history: List[Dict], max_turns: int = 3) -> str:
    """
    Format the last N turns of chat history into a readable string.
    Used by the rewrite prompt to resolve pronouns and references.
    """
    recent = history[-(max_turns * 2):] if len(history) > max_turns * 2 else history
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "(no prior conversation)"


# ---------------------------------------------------------------------------
# Helper: Format Document List for LLM
# ---------------------------------------------------------------------------

def _format_document_list(documents: List[Dict]) -> str:
    """Format document list for the LLM scope detection prompt."""
    if not documents:
        return "(no documents loaded)"
    return "\n".join(f"- {doc['filename']} ({doc['chunk_count']} chunks)" for doc in documents)


# ---------------------------------------------------------------------------
# Question Rewriting + Scope + Intent (Single LLM Call)
# ---------------------------------------------------------------------------

def analyze_and_rewrite(
    question: str,
    history: List[Dict],
    documents: List[Dict],
    llm: ChatGoogleGenerativeAI,
) -> Tuple[str, Optional[str], bool]:
    """
    Single LLM call that does THREE things:
    1. Rewrites follow-up questions into standalone form
    2. Detects document scope (maps natural language to exact filename)
    3. Detects comparison/synthesis intent

    Returns: (rewritten_question, scope_filename_or_None, is_comparison)
    """
    chat_history = _format_chat_history(history)
    doc_list = _format_document_list(documents)

    # Fill in the document list placeholder in the system prompt
    system_prompt = REWRITE_SYSTEM_PROMPT.format(document_list=doc_list)

    prompt = REWRITE_USER_TEMPLATE.format(
        document_list=doc_list,
        chat_history=chat_history,
        question=question,
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ])
        text = _extract_text(response.content)
        parsed = _parse_json_response(text)

        rewritten = parsed.get("rewritten_question", question).strip()
        if not rewritten or len(rewritten) < 5:
            rewritten = question

        # Validate scope against actual documents
        scope = parsed.get("detected_scope")
        if scope:
            valid_filenames = {doc["filename"] for doc in documents}
            if scope not in valid_filenames:
                # Try partial match
                for fname in valid_filenames:
                    if scope.lower() in fname.lower() or fname.lower().endswith(scope.lower()):
                        scope = fname
                        break
                else:
                    scope = None

        is_comparison = bool(parsed.get("is_comparison", False))

        return rewritten, scope, is_comparison

    except Exception:
        # Fallback: use original question, no scope, no comparison
        return question, None, False


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (for tests and direct calls)
# ---------------------------------------------------------------------------

def rewrite_question(
    question: str,
    history: List[Dict],
    llm: ChatGoogleGenerativeAI,
    documents: Optional[List[Dict]] = None,
) -> str:
    """
    Rewrite a follow-up question into standalone form.
    If documents are provided, also detects scope and comparison intent.
    """
    if documents is not None:
        rewritten, _, _ = analyze_and_rewrite(question, history, documents, llm)
        return rewritten

    # Fallback without document context
    chat_history = _format_chat_history(history)
    prompt = REWRITE_USER_TEMPLATE.format(
        document_list="(not provided)",
        chat_history=chat_history,
        question=question,
    )
    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a query rewriting assistant. Rewrite the question as standalone text. Return ONLY the rewritten question."},
            {"role": "user", "content": prompt},
        ])
        text = _extract_text(response.content)
        rewritten = text.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 5:
            return rewritten
    except Exception:
        pass
    return question


def detect_scope(question: str, documents: List[Dict]) -> Optional[str]:
    """
    Detect if the user scoped their question to a specific document.
    Uses substring matching as a fast pre-check (LLM does the real work).
    """
    if not documents:
        return None
    question_lower = question.lower()
    for doc in documents:
        filename = doc["filename"]
        name_parts = filename.lower().replace(".pdf", "").replace(".txt", "").replace(".md", "")
        if name_parts in question_lower or filename.lower() in question_lower:
            return filename
    return None


def detect_comparison(question: str) -> bool:
    """Detect comparison intent via keyword matching (fast fallback)."""
    keywords = {
        "compare", "vs", "versus", "difference", "differ",
        "both", "contrasting", "contrast", "differences between",
    }
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Dynamic Relevance Threshold
# ---------------------------------------------------------------------------

def compute_relevance_threshold(chunks: List[Dict]) -> float:
    """
    Compute a dynamic relevance threshold based on the score distribution.
    
    If scores are tightly clustered (all similarly relevant), raise the threshold.
    If scores are spread out, use the base threshold.
    """
    if not chunks:
        return BASE_RELEVANCE_THRESHOLD

    scores = [c["score"] for c in chunks]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    spread = max_score - min_score

    # If top score is well above average, threshold is relative to top score
    if spread > 0.2 and max_score > avg_score + 0.15:
        return max(BASE_RELEVANCE_THRESHOLD, max_score * 0.5)

    return BASE_RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Chunk Retrieval (Enhanced with per-doc balancing)
# ---------------------------------------------------------------------------

def retrieve_chunks(
    question: str,
    collection: chromadb.Collection,
    documents: List[Dict],
    scope: Optional[str] = None,
    is_comparison: bool = False,
) -> Tuple[List[Dict], float]:
    """
    Retrieve the most relevant chunks from ChromaDB for a given question.
    
    Strategies:
    - Scoped query: filter to specific document
    - Comparison or multi-doc: retrieve top-k PER document (balanced)
    - Single doc or small collection: normal pooled search
    
    Returns: (chunks_with_scores, top_score)
    """
    total_docs = len(documents)

    # Strategy 1: Scoped to specific document
    if scope:
        return _retrieve_scoped(question, collection, scope)

    # Strategy 2: Multi-doc — default to per-doc balanced retrieval
    # This prevents single-document dominance in vector search
    if total_docs > 1:
        return _retrieve_balanced(question, collection, documents)

    # Strategy 3: Single document — normal pooled search
    return _retrieve_pooled(question, collection, n_results=8)


def _retrieve_scoped(
    question: str,
    collection: chromadb.Collection,
    scope: str,
) -> Tuple[List[Dict], float]:
    """Retrieve chunks filtered to a specific document."""
    try:
        results = collection.query(
            query_texts=[question],
            n_results=min(8, collection.count() or 1),
            where={"filename": scope},
        )
        chunks = _format_results(results)
        top_score = chunks[0]["score"] if chunks else 0.0
        return chunks, top_score
    except Exception:
        return [], 0.0


def _retrieve_balanced(
    question: str,
    collection: chromadb.Collection,
    documents: List[Dict],
) -> Tuple[List[Dict], float]:
    """
    Retrieve top-k chunks from EACH document separately.
    This ensures all documents are represented in the results.
    Prevents the common failure mode where one doc dominates vector search.
    """
    all_chunks = []
    for doc in documents:
        try:
            results = collection.query(
                query_texts=[question],
                n_results=PER_DOC_K,
                where={"filename": doc["filename"]},
            )
            doc_chunks = _format_results(results)
            all_chunks.extend(doc_chunks)
        except Exception:
            continue

    # Sort by score and cap at max total
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_score = all_chunks[0]["score"] if all_chunks else 0.0

    return all_chunks[:MAX_TOTAL_CHUNKS], top_score


def _retrieve_pooled(
    question: str,
    collection: chromadb.Collection,
    n_results: int = 8,
) -> Tuple[List[Dict], float]:
    """Normal pooled similarity search across all documents."""
    try:
        results = collection.query(
            query_texts=[question],
            n_results=min(n_results, collection.count() or 1),
        )
        chunks = _format_results(results)
        top_score = chunks[0]["score"] if chunks else 0.0
        return chunks, top_score
    except Exception:
        return [], 0.0


# ---------------------------------------------------------------------------
# Result Formatting
# ---------------------------------------------------------------------------

def _format_results(results: dict) -> List[Dict]:
    """
    Convert ChromaDB query results into a clean list of chunk dicts.
    Score = max(0, 1 - distance) for cosine similarity.
    """
    chunks = []
    if not results.get("metadatas") or not results["metadatas"][0]:
        return chunks

    for i, (meta, doc, dist) in enumerate(zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0] if results.get("distances") else [0.0] * len(results["metadatas"][0]),
    )):
        score = max(0.0, 1.0 - dist)
        chunks.append({
            "content": doc,
            "filename": meta["filename"],
            "location": meta["location"],
            "chunk_index": meta["chunk_index"],
            "doc_id": meta["doc_id"],
            "score": round(score, 4),
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks
