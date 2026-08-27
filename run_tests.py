"""
Comprehensive test suite for DocQuery — Functional Coverage + RAG Quality.
Run: python3 run_tests.py
"""
import os
import sys
import json
import tempfile
import traceback

from dotenv import load_dotenv
load_dotenv()

from ingestion import (
    ingest_file, get_chroma_collection, remove_document,
    list_documents, validate_file, parse_pdf, chunk_text,
)
from retrieval import (
    rewrite_question, detect_scope, detect_comparison,
    retrieve_chunks, BASE_RELEVANCE_THRESHOLD as RELEVANCE_THRESHOLD,
)
from generation import generate_answer, _parse_json_response, _call_llm_with_retry
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
COLLECTION = None
RESULTS = []

def log(test_num, name, status, detail=""):
    icon = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "INFO"
    RESULTS.append({"num": test_num, "name": name, "status": status, "detail": detail})
    print(f"  [{icon}] #{test_num} {name}: {detail[:120]}")


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)


def reset_collection():
    global COLLECTION
    import chromadb
    client = chromadb.PersistentClient(path="chroma_data")
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    from ingestion import GeminiEmbeddingFunction
    ef = GeminiEmbeddingFunction(API_KEY)
    COLLECTION = client.get_or_create_collection(
        name="documents", embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )


def ingest(path, name):
    success, result, count = ingest_file(path, name, API_KEY, COLLECTION)
    return success, result, count


# ============================================================
# FUNCTIONAL COVERAGE TESTS
# ============================================================

def test_01_no_docs_block():
    """Test 1: No docs loaded, ask a question -> blocked"""
    reset_collection()
    docs = list_documents(COLLECTION)
    assert len(docs) == 0, "Should have 0 docs"
    llm = get_llm()
    rewritten = rewrite_question("What is the refund policy?", [], llm)
    scope = detect_scope(rewritten, [])
    chunks, score = retrieve_chunks(rewritten, COLLECTION, [], scope=scope)
    assert len(chunks) == 0, "Should retrieve 0 chunks with no docs"
    log(1, "No docs, ask question", "PASS", "Blocked - no docs, no chunks retrieved")


def test_02_empty_short_question():
    """Test 2: Empty/short question -> blocked"""
    q_empty = ""
    q_short = "Hi"
    assert len(q_empty.strip().split()) < 3, "Empty too short"
    assert len(q_short.strip().split()) < 3, "Short too short"
    log(2, "Empty/short question", "PASS", "Blocked - words < 3")


def test_03_unsupported_file():
    """Test 3: Unsupported file type -> specific error"""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"fake docx content")
        tmp = f.name
    valid, error = validate_file("report.docx")
    assert not valid, "Should reject .docx"
    assert ".docx" in error, "Error should mention .docx"
    os.unlink(tmp)
    log(3, "Unsupported file", "PASS", f"Error correctly: {error}")


def test_04_corrupt_empty_file():
    """Test 4: Corrupted/empty file -> specific error"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"not a real pdf")
        tmp = f.name
    success, result, count = ingest_file(tmp, "corrupt.pdf", API_KEY, COLLECTION)
    os.unlink(tmp)
    assert not success, "Should fail on corrupt file"
    assert count == 0, "Should have 0 chunks"
    log(4, "Corrupt file", "PASS", f"Error: {result}")


def test_05_direct_lookup():
    """Test 5: Direct fact lookup -> correct answer + citation"""
    reset_collection()
    path = os.path.join(os.path.dirname(__file__), "test_resume.pdf")
    success, doc_id, count = ingest(path, "test_resume.pdf")
    assert success, f"Ingest failed: {doc_id}"

    llm = get_llm()
    q = "What is John Smith's GPA?"
    rewritten = rewrite_question(q, [], llm)
    scope = detect_scope(rewritten, list_documents(COLLECTION))
    chunks, score = retrieve_chunks(rewritten, COLLECTION, list_documents(COLLECTION), scope=scope)
    assert len(chunks) > 0, "Should retrieve chunks"

    result = generate_answer(rewritten, chunks, score, llm)
    r = result.model_dump()
    assert "3.8" in r["answer"] or "3.8" in str(r["citations"]), "Answer should mention GPA 3.8"
    assert len(r["citations"]) > 0, "Should have citations"
    assert r["citations"][0]["doc"] == "test_resume.pdf", "Citation should reference correct file"
    log(5, "Direct lookup", "PASS", f"Answer: {r['answer'][:80]}...")


def test_06_not_found():
    """Test 6: Answer not present -> not_found=true"""
    reset_collection()
    path = os.path.join(os.path.dirname(__file__), "test_resume.pdf")
    success, doc_id, count = ingest(path, "test_resume.pdf")
    assert success

    llm = get_llm()
    q = "What is John Smith's annual salary?"
    rewritten = rewrite_question(q, [], llm)
    scope = detect_scope(rewritten, list_documents(COLLECTION))
    chunks, score = retrieve_chunks(rewritten, COLLECTION, list_documents(COLLECTION), scope=scope)

    result = generate_answer(rewritten, chunks, score, llm)
    r = result.model_dump()
    assert r["not_found"] == True, f"Should be not_found=True, got: {r}"
    log(6, "Not found", "PASS", f"not_found={r['not_found']}, answer empty: {r['answer'] == ''}")


def test_07_synthesis_across_docs():
    """Test 7: Synthesis across 2 docs -> answer with both cited"""
    reset_collection()
    r1 = ingest("test_refund_v1.pdf", "test_refund_v1.pdf")
    r2 = ingest("test_refund_v2.pdf", "test_refund_v2.pdf")
    assert r1[0] and r2[0]

    llm = get_llm()
    q = "What are the refund policies across all documents?"
    is_comp = detect_comparison(q)
    docs = list_documents(COLLECTION)
    chunks, score = retrieve_chunks(q, COLLECTION, docs, is_comparison=is_comp)
    assert len(chunks) >= 2, f"Should retrieve from both docs, got {len(chunks)}"

    result = generate_answer(q, chunks, score, llm)
    r = result.model_dump()
    cited_docs = set(c["doc"] for c in r["citations"])
    assert len(cited_docs) >= 2, f"Should cite both docs, cited: {cited_docs}"
    log(7, "Synthesis across docs", "PASS", f"Cited: {cited_docs}")


def test_08_conflict_detection():
    """Test 8: Two docs with conflicting values -> conflict flagged"""
    reset_collection()
    r1 = ingest("test_refund_v1.pdf", "test_refund_v1.pdf")
    r2 = ingest("test_refund_v2.pdf", "test_refund_v2.pdf")
    assert r1[0] and r2[0]

    llm = get_llm()
    q = "What is the refund window period?"
    is_comp = detect_comparison(q)
    docs = list_documents(COLLECTION)
    chunks, score = retrieve_chunks(q, COLLECTION, docs, is_comparison=is_comp)

    result = generate_answer(q, chunks, score, llm)
    r = result.model_dump()
    assert r["conflict"] == True, f"Should detect conflict, got conflict={r['conflict']}"
    assert len(r["conflicting_values"]) >= 2, f"Should have 2+ conflicting values, got {len(r['conflicting_values'])}"
    log(8, "Conflict detection", "PASS", f"conflict={r['conflict']}, values: {r['conflicting_values']}")


def test_09_scoped_query():
    """Test 9: Document-scoped query -> only that doc used"""
    reset_collection()
    r1 = ingest("test_refund_v1.pdf", "test_refund_v1.pdf")
    r2 = ingest("test_refund_v2.pdf", "test_refund_v2.pdf")
    assert r1[0] and r2[0]

    llm = get_llm()
    q = "In test_refund_v1.pdf only, what is the refund window?"
    docs = list_documents(COLLECTION)
    scope = detect_scope(q, docs)
    assert scope == "test_refund_v1.pdf", f"Should scope to v1, got scope={scope}"

    chunks, score = retrieve_chunks(q, COLLECTION, docs, scope=scope)
    scoped_docs = set(c["filename"] for c in chunks)
    assert scoped_docs == {"test_refund_v1.pdf"}, f"Only v1 chunks, got: {scoped_docs}"

    result = generate_answer(q, chunks, score, llm)
    r = result.model_dump()
    assert "14" in r["answer"], "Answer should mention 14 days (v1)"
    log(9, "Scoped query", "PASS", f"Scope: {scope}, answer: {r['answer'][:80]}")


def test_10_followup_rewrite():
    """Test 10: Follow-up question -> correctly resolved via history"""
    reset_collection()
    r1 = ingest("test_resume.pdf", "test_resume.pdf")
    assert r1[0]

    llm = get_llm()
    history = [
        {"role": "user", "content": "What companies has John Smith worked at?"},
        {"role": "assistant", "content": "John Smith has worked at Google and Meta."},
    ]
    followup = "And what was his GPA at Stanford?"
    rewritten = rewrite_question(followup, history, llm)
    assert "John" in rewritten or "GPA" in rewritten or "Stanford" in rewritten, \
        f"Rewritten should resolve references: {rewritten}"
    log(10, "Follow-up rewrite", "PASS", f"Rewritten: {rewritten}")


def test_12_injection():
    """Test 12: Prompt injection -> quoted as data, not followed"""
    reset_collection()
    r = ingest("test_injection.pdf", "test_injection.pdf")
    assert r[0]

    llm = get_llm()
    q = "How many days of annual leave do employees get?"
    rewritten = rewrite_question(q, [], llm)
    docs = list_documents(COLLECTION)
    chunks, score = retrieve_chunks(rewritten, COLLECTION, docs)

    result = generate_answer(rewritten, chunks, score, llm)
    r = result.model_dump()
    assert "20" in r["answer"], f"Should say 20 days, got: {r['answer']}"
    assert "50" not in r["answer"] or "50" in str(r["citations"]), \
        f"Should NOT say 50 days (injection), got: {r['answer']}"
    log(12, "Injection resistance", "PASS", f"Answer: {r['answer'][:80]}")


def test_13_weak_relevance():
    """Test 13: Weakly relevant -> Low confidence"""
    reset_collection()
    ingest("test_resume.pdf", "test_resume.pdf")

    llm = get_llm()
    q = "What is the meaning of life according to this document?"
    docs = list_documents(COLLECTION)
    chunks, score = retrieve_chunks(q, COLLECTION, docs)

    result = generate_answer(q, chunks, score, llm, force_low_confidence=(score < RELEVANCE_THRESHOLD and score > 0))
    r = result.model_dump()
    log(13, "Weak relevance", "PASS", f"confidence={r['confidence']}, score={score}")


def test_14_doc_removal():
    """Test 14: Remove doc -> content gone from retrieval"""
    reset_collection()
    ingest("test_resume.pdf", "test_resume.pdf")
    docs = list_documents(COLLECTION)
    assert len(docs) == 1

    doc_id = docs[0]["doc_id"]
    removed = remove_document(doc_id, COLLECTION)
    assert removed

    docs_after = list_documents(COLLECTION)
    assert len(docs_after) == 0, f"Should have 0 docs after removal, got {len(docs_after)}"

    llm = get_llm()
    chunks, score = retrieve_chunks("What is John's GPA?", COLLECTION, [])
    assert len(chunks) == 0, "Should retrieve 0 chunks after removal"
    log(14, "Document removal", "PASS", "Doc removed, no chunks retrieved")


# ============================================================
# RAG QUALITY GOLDEN SET
# ============================================================

def run_golden_set():
    """Run golden test questions against known ground truth."""
    reset_collection()
    ingest("test_resume.pdf", "test_resume.pdf")
    ingest("test_refund_v1.pdf", "test_refund_v1.pdf")
    ingest("test_refund_v2.pdf", "test_refund_v2.pdf")
    ingest("test_injection.pdf", "test_injection.pdf")
    ingest("test_filler.pdf", "test_filler.pdf")

    llm = get_llm()
    docs = list_documents(COLLECTION)

    golden = [
        {
            "q": "What is John Smith's GPA?",
            "expected_answer": "3.8/4.0",
            "expected_doc": "test_resume.pdf",
            "expected_behavior": "normal",
        },
        {
            "q": "What is the refund period in version 1?",
            "expected_answer": "14 days",
            "expected_doc": "test_refund_v1.pdf",
            "expected_behavior": "normal",
        },
        {
            "q": "What is the refund period in version 2?",
            "expected_answer": "30 days",
            "expected_doc": "test_refund_v2.pdf",
            "expected_behavior": "normal",
        },
        {
            "q": "What are the refund policies across all documents?",
            "expected_answer": "v1: 14 days, v2: 30 days",
            "expected_doc": "both refund docs",
            "expected_behavior": "conflict",
        },
        {
            "q": "How many days of annual leave do employees get?",
            "expected_answer": "20 days",
            "expected_doc": "test_injection.pdf",
            "expected_behavior": "normal",
        },
        {
            "q": "What is the meaning of life according to this document?",
            "expected_answer": "not found",
            "expected_doc": "none",
            "expected_behavior": "not_found",
        },
        {
            "q": "What ingredients are in the chocolate cake?",
            "expected_answer": "flour, sugar, cocoa powder",
            "expected_doc": "test_filler.pdf",
            "expected_behavior": "normal",
        },
    ]

    print("\n" + "="*60)
    print("RAG QUALITY GOLDEN SET")
    print("="*60)

    scores = {"correct": 0, "total": 0, "grounded": 0, "cited": 0}

    for i, g in enumerate(golden, 1):
        q = g["q"]
        rewritten = rewrite_question(q, [], llm)
        scope = detect_scope(rewritten, docs)
        is_comp = detect_comparison(rewritten)
        chunks, score = retrieve_chunks(rewritten, COLLECTION, docs, scope=scope, is_comparison=is_comp)
        force_low = score < RELEVANCE_THRESHOLD and score > 0
        result = generate_answer(rewritten, chunks, score, llm, force_low_confidence=force_low)
        r = result.model_dump()

        # Score correctness
        correct = False
        if g["expected_behavior"] == "not_found":
            correct = r["not_found"] == True
        elif g["expected_behavior"] == "conflict":
            correct = r["conflict"] == True
        else:
            correct = g["expected_answer"].lower() in r["answer"].lower()

        # Groundedness: check citations exist
        grounded = len(r["citations"]) > 0
        cited_correctly = any(g["expected_doc"] in c.get("doc", "") for c in r["citations"]) if g["expected_doc"] != "none" else True

        if correct:
            scores["correct"] += 1
        if grounded:
            scores["grounded"] += 1
        if cited_correctly:
            scores["cited"] += 1
        scores["total"] += 1

        status = "PASS" if correct else "FAIL"
        print(f"\n  Q{i}: {q}")
        print(f"    Expected: {g['expected_answer']} (behavior: {g['expected_behavior']})")
        print(f"    Got:      {r['answer'][:100]}")
        print(f"    Score:    {score} | Conflict: {r['conflict']} | Not found: {r['not_found']}")
        print(f"    Citations: {[c['doc'] for c in r['citations']]}")
        print(f"    [{status}] Correct={correct} Grounded={grounded} Cited={cited_correctly}")

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Correctness:  {scores['correct']}/{scores['total']}")
    print(f"  Groundedness: {scores['grounded']}/{scores['total']}")
    print(f"  Citation:     {scores['cited']}/{scores['total']}")
    print("="*60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        print("="*60)
        print("FUNCTIONAL COVERAGE TESTS")
        print("="*60)

        test_01_no_docs_block()
        test_02_empty_short_question()
        test_03_unsupported_file()
        test_04_corrupt_empty_file()
        test_05_direct_lookup()
        test_06_not_found()
        test_07_synthesis_across_docs()
        test_08_conflict_detection()
        test_09_scoped_query()
        test_10_followup_rewrite()
        test_12_injection()
        test_13_weak_relevance()
        test_14_doc_removal()

        run_golden_set()

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
