"""
DocQuery — AI Document Q&A Knowledge Assistant
===============================================
Main Streamlit application that orchestrates the full pipeline:
  1. Sidebar: file upload, parsing, chunking, embedding, ChromaDB storage
  2. Chat: question rewriting, retrieval, grounded generation, source display

Architecture:
  app.py          → This file (UI layer, session state, Streamlit rendering)
  ingestion.py    → File parsing, text chunking, embedding, ChromaDB CRUD
  retrieval.py    → Query rewriting, scope detection, similarity search
  generation.py   → LLM invocation, JSON parsing, Pydantic schema validation
  prompts.py      → All system prompts and user-facing templates

Run:  streamlit run app.py
"""

import os
import uuid
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Internal modules ---
from ingestion import (
    get_chroma_collection,
    ingest_file,
    remove_document,
    list_documents,
    ALLOWED_EXTENSIONS,
)
from retrieval import (
    analyze_and_rewrite,
    retrieve_chunks,
    compute_relevance_threshold,
    BASE_RELEVANCE_THRESHOLD,
)
from generation import generate_answer


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """
    Initialize Streamlit session state with defaults.
    Persists across reruns within a single browser session.
    Each session gets a unique ID for ChromaDB isolation.
    """
    defaults = {
        "chat_history": [],   # List of {role, content, result?} dicts
        "documents": [],      # List of {doc_id, filename, chunk_count} dicts
        "session_id": None,   # Unique session identifier for ChromaDB isolation
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Generate unique session ID for ChromaDB collection isolation
    # This prevents data bleed between different browser sessions/users
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())[:8]


def load_api_keys() -> List[str]:
    """
    Load API keys from environment variables or Streamlit Cloud secrets.
    Returns a list of keys to try in order — first success wins.
    This provides automatic fallback when a key hits rate limits.
    """
    keys = []

    # Streamlit Cloud secrets (deployed) or .env (local)
    try:
        keys.append(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, FileNotFoundError):
        pass

    try:
        keys.append(st.secrets["GEMINI_API_KEY"])
    except (KeyError, FileNotFoundError):
        pass

    # Environment variables fallback
    primary = os.environ.get("GOOGLE_API_KEY", "")
    if primary and primary not in keys:
        keys.append(primary)
    secondary = os.environ.get("GEMINI_API_KEY", "")
    if secondary and secondary not in keys:
        keys.append(secondary)

    return [k for k in keys if k]


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """
    Create a Gemini LLM instance with the given API key.
    Uses gemini-flash-lite-latest for higher free-tier quota.
    Temperature 0.1 for deterministic, grounded responses.
    """
    return ChatGoogleGenerativeAI(
        model="models/gemini-flash-lite-latest",
        google_api_key=api_key,
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# UI Rendering Helpers
# ---------------------------------------------------------------------------

def render_sources(citations: List[Dict], label: str = "Sources"):
    """
    Render an expandable sources section showing each citation:
    filename, page/section, and the exact snippet used.
    Users can verify answers against original documents.
    """
    if not citations:
        return
    with st.expander(label, expanded=False):
        for i, cit in enumerate(citations, 1):
            st.markdown(f"**{i}.** `{cit['doc']}` — {cit['location']}")
            st.markdown(f"> {cit['snippet']}")


def render_conflict(conflicting_values: List[Dict]):
    """
    Render conflicting values side-by-side with a visible warning.
    Each value is shown with its source so the user can see the disagreement.
    """
    if not conflicting_values:
        return
    st.warning("**⚠️ Sources disagree:**")
    cols = st.columns(len(conflicting_values))
    for i, val in enumerate(conflicting_values):
        with cols[i]:
            st.markdown(f"**Value:** {val.get('value', 'N/A')}")
            st.markdown(f"**Source:** {val.get('source', 'N/A')}")


def render_answer(result: Dict, msg_index: int):
    """
    Render a single answer turn with all UI elements:
    - "Not found" message if answer doesn't exist in docs
    - Conflict warning with side-by-side values
    - The answer text itself
    - Low confidence badge if applicable
    - Expandable sources section
    """
    if result.get("not_found"):
        st.info("I couldn't find this in the uploaded documents.")
        return

    if result.get("conflict"):
        render_conflict(result.get("conflicting_values", []))

    st.markdown(result.get("answer", ""))

    if result.get("confidence") == "Low":
        st.caption("⚠️ Low confidence — the answer may be incomplete or weakly supported.")

    render_sources(
        result.get("citations", []),
        label=f"Sources (turn {msg_index})",
    )


# ---------------------------------------------------------------------------
# Sidebar — File Upload & Document Management
# ---------------------------------------------------------------------------

def handle_ingestion(api_key: str, collection):
    """
    Sidebar: file upload, validation, ingestion, and document list.
    - Accepts .pdf, .txt, .md files (multiple at once)
    - Validates extension → parses → chunks → embeds → stores in Chroma
    - Shows success/error per file
    - Lists all loaded documents with a Remove button each
    """
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files:
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    st.error(f"❌ '{file.name}': Unsupported file type '{ext}'")
                    continue

                # Write uploaded bytes to a temp file for parsing
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                # Ingest: parse → chunk → embed → store
                success, result, chunk_count = ingest_file(
                    tmp_path, file.name, api_key, collection
                )

                os.unlink(tmp_path)  # Clean up temp file

                if success:
                    st.success(f"✅ '{file.name}' ready — {chunk_count} chunks indexed (ID: {result})")
                else:
                    st.error(f"❌ '{file.name}': {result}")

            # Refresh document list after uploads
            st.session_state.documents = list_documents(collection)

        st.divider()
        st.header("Loaded Documents")
        st.session_state.documents = list_documents(collection)

        if not st.session_state.documents:
            st.info("No documents uploaded yet.")
        else:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{doc['filename']} ({doc['chunk_count']} chunks)")
                with col2:
                    if st.button("Remove", key=f"rm_{doc['doc_id']}"):
                        if remove_document(doc["doc_id"], collection):
                            st.success(f"Removed '{doc['filename']}'")
                            st.session_state.documents = list_documents(collection)
                            st.rerun()
                        else:
                            st.error("Failed to remove")


# ---------------------------------------------------------------------------
# Chat — Question Pipeline (LLM-based scope + intent detection)
# ---------------------------------------------------------------------------

def handle_chat(api_key: str, collection):
    """
    Main chat interface:
    1. Renders existing conversation history
    2. Validates new question (empty? too short? no docs?)
    3. LLM-based analysis: rewrites question, detects scope, detects comparison intent
    4. Retrieves relevant chunks from ChromaDB (per-doc balanced for multi-doc)
    5. Generates grounded answer via Gemini with retry + fallback
    6. Renders answer with sources
    """
    st.header("Ask a Question")

    # --- Render conversation history ---
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_answer(msg.get("result", {}), i)

    # --- Get new question from user ---
    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    question = question.strip()

    # --- Validate: minimum 3 words ---
    if len(question.split()) < 3:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.warning("Please ask a complete question (at least 3 words).")
        return

    # --- Validate: documents must exist ---
    if not st.session_state.documents:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.info("Please upload a document first before asking questions.")
        return

    # --- Display user message ---
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # --- Process question through the full pipeline ---
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            api_keys = load_api_keys()
            result_dict = None

            # Try each API key until one succeeds (fallback on rate limit)
            for attempt_key in api_keys:
                try:
                    llm = get_llm(attempt_key)

                    # Step 1: LLM-based analysis
                    # Single call that rewrites question + detects scope + detects comparison
                    # Uses semantic matching, not just substring/regex
                    rewritten, scope, is_comparison = analyze_and_rewrite(
                        question,
                        st.session_state.chat_history[:-1],
                        st.session_state.documents,
                        llm,
                    )

                    # Step 2: Retrieve relevant chunks from ChromaDB
                    # Multi-doc queries default to per-doc balanced retrieval
                    chunks, top_score = retrieve_chunks(
                        rewritten,
                        collection,
                        st.session_state.documents,
                        scope=scope,
                        is_comparison=is_comparison,
                    )

                    # Step 3: Force Low confidence if top result is weakly relevant
                    threshold = compute_relevance_threshold(chunks)
                    force_low = top_score < threshold if top_score > 0 else False

                    # Step 4: Generate grounded answer via Gemini
                    result = generate_answer(
                        rewritten,
                        chunks,
                        top_score,
                        llm,
                        force_low_confidence=force_low,
                    )

                    result_dict = result.model_dump()
                    break  # Success — stop trying keys

                except Exception as e:
                    # Rate limit hit → try next key
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        continue
                    raise  # Non-rate-limit error → propagate

            # All keys exhausted
            if result_dict is None:
                result_dict = {
                    "answer": "Sorry, all API keys are rate-limited. Please try again in a few minutes.",
                    "not_found": True,
                    "conflict": False,
                    "conflicting_values": [],
                    "citations": [],
                    "confidence": "Low",
                }

            # --- Render the answer ---
            render_answer(result_dict, len(st.session_state.chat_history))

            # --- Persist to chat history ---
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result_dict.get("answer", ""),
                "result": result_dict,
            })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point — configures page, loads keys, initializes services."""

    st.set_page_config(
        page_title="DocQuery - AI Document Q&A",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 DocQuery")
    st.caption("AI-powered document Q&A with source attribution")

    # Load .env file (secrets)
    load_dotenv()

    # Get API keys — fail early if none available
    api_keys = load_api_keys()
    if not api_keys:
        st.error("⚠️ No API key found. Create a .env file with GOOGLE_API_KEY=your-key")
        st.stop()

    # Use first available key for collection setup
    api_key = api_keys[0]

    # Initialize session state (generates unique session_id)
    init_session_state()

    # Initialize ChromaDB collection (session-isolated)
    collection = get_chroma_collection(api_key, session_id=st.session_state.session_id)

    # Run sidebar (upload + doc management) and chat
    handle_ingestion(api_key, collection)
    handle_chat(api_key, collection)


if __name__ == "__main__":
    main()
