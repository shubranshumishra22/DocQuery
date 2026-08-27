# DocQuery — AI Document Q&A Knowledge Assistant

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-here"

# 4. Run the app
streamlit run app.py
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| UI | Streamlit | Rapid prototyping, built-in chat UI, file upload |
| PDF Parsing | PyMuPDF (`fitz`) | Fast, reliable page-level text extraction |
| Chunking | LangChain RecursiveCharacterTextSplitter | Smart splitting with overlap, preserves context |
| Embeddings | Gemini `text-embedding-004` | High-quality, fast, Google-native |
| Vector Store | ChromaDB (persistent) | Simple local setup, supports metadata filtering |
| LLM | Gemini `gemini-2.0-flash` | Fast, capable, structured output support |

## Approach

**Chunking:** Text is split into 900-char chunks with 150-char overlap using recursive separators (`\n\n`, `\n`, `. `, ` `, `""`). Each chunk carries metadata: filename, page/section, chunk index, and a unique `doc_id`.

**Retrieval:** Embeddings are stored in ChromaDB with cosine similarity. Comparison queries retrieve top-k chunks per document separately to ensure both sides are represented. Scope detection extracts filename filters from natural language.

**Conflict Handling:** The LLM is prompted to detect when different sources give different values for the same fact. When conflicts arise, both values are shown side-by-side with their sources — the system never silently picks one. A `⚠️ Sources disagree` flag is rendered in the UI.

## Assumptions / Trade-offs

- Gemini API key must be set via environment variable (no .env file auto-loading to avoid security issues).
- ChromaDB stores data locally in `./chroma_data/`. Deleting documents removes chunks from the store.
- The relevance threshold for "Low" confidence is set at 0.35 cosine similarity. This may need tuning for different document types.
- PDF text extraction assumes selectable text (not scanned images). OCR is not included.
- The app processes files sequentially on upload; very large batches may be slow due to embedding API calls.
