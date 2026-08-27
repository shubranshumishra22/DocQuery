"""
generation.py — Grounded Answer Generation via Gemini LLM
==========================================================
Responsibilities:
  1. Format retrieved chunks into labeled context for the LLM
  2. Invoke Gemini with a system prompt that enforces grounded generation
  3. Parse the LLM's JSON output into a Pydantic schema
  4. Handle parse failures with retry + strict fallback
  5. Force Low confidence when relevance score is weak

Key Design Decisions:
  - System prompt explicitly forbids outside knowledge and fabrication
  - Prompt injection in documents is quoted as data, never executed
  - Pydantic schema enforces structured output (answer, not_found, conflict, citations)
  - Retry on JSON parse failure with stricter instructions
  - Fallback returns not_found=True if all retries fail
"""

import json
from typing import List, Dict, Literal
from enum import Enum

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import GENERATION_SYSTEM_PROMPT, GENERATION_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Pydantic Schema — Enforces structured LLM output
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single source citation linking an answer claim to a document chunk."""
    doc: str = Field(description="Filename of the source document")
    location: str = Field(description="Page number or section heading")
    snippet: str = Field(description="Exact text snippet from the source")


class ConflictingValue(BaseModel):
    """A single conflicting value with its source."""
    value: str = Field(description="The conflicting value")
    source: str = Field(description="Source of this value")


class QAOutput(BaseModel):
    """
    The structured output schema for every Q&A response.
    
    Fields:
      - answer: The answer text (empty if not_found)
      - not_found: True if answer is not in the provided context
      - conflict: True if sources disagree on a fact
      - conflicting_values: List of differing values with sources
      - citations: Source references for the answer
      - confidence: High/Medium/Low based on retrieval relevance
    """
    answer: str = Field(default="", description="The answer, or empty if not_found")
    not_found: bool = Field(default=False, description="True if answer not in context")
    conflict: bool = Field(default=False, description="True if sources disagree")
    conflicting_values: List[ConflictingValue] = Field(
        default_factory=list,
        description="List of conflicting values with sources"
    )
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: Literal["High", "Medium", "Low"] = Field(
        default="Low",
        description="Confidence level in the answer"
    )


# ---------------------------------------------------------------------------
# Text Extraction (handles list/string responses)
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
# Context Formatting
# ---------------------------------------------------------------------------

def _format_chunks_for_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into labeled context for the LLM.
    
    Each chunk is labeled with its source:
      [Source: filename, page/section] chunk_text
    """
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Source: {chunk['filename']}, {chunk['location']}] {chunk['content']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks and extra text.
    
    Steps:
    1. Strip markdown code block markers (```json ... ```)
    2. Find the first { and last } to extract the JSON object
    3. Parse with json.loads
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Extract JSON object (may be surrounded by extra text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    chunks: List[Dict],
    top_score: float,
    llm: ChatGoogleGenerativeAI,
    force_low_confidence: bool = False,
) -> QAOutput:
    """
    Generate a grounded answer using the LLM with retrieved context.
    
    Pipeline:
    1. Format chunks into labeled context
    2. Call LLM with system prompt enforcing grounded generation
    3. Parse JSON response into Pydantic schema
    4. Force Low confidence if top_score is below threshold
    5. Add fallback citations if LLM didn't provide any
    
    Returns a validated QAOutput instance.
    """
    context = _format_chunks_for_context(chunks)
    prompt = GENERATION_USER_TEMPLATE.format(context=context, question=question)

    # Call LLM with retry on parse failure
    response = _call_llm_with_retry(llm, prompt)

    # Force Low confidence if retrieval score was weak
    if force_low_confidence:
        response["confidence"] = "Low"

    # Fallback: add citations from top chunks if LLM didn't provide any
    if not response.get("citations"):
        response["citations"] = [
            {
                "doc": c["filename"],
                "location": c["location"],
                "snippet": c["content"][:200],
            }
            for c in chunks[:3]
        ]

    # Normalize conflicting_values (LLM may return nested dicts)
    normalized = []
    for cv in response.get("conflicting_values", []):
        if isinstance(cv, dict):
            val = cv.get("value", "")
            src = cv.get("source", "")
            if isinstance(val, dict):
                val = str(val)
            if isinstance(src, dict):
                src = str(src)
            normalized.append({"value": val, "source": src})
        else:
            normalized.append({"value": str(cv), "source": ""})
    response["conflicting_values"] = normalized

    return QAOutput(**response)


# ---------------------------------------------------------------------------
# LLM Invocation with Retry
# ---------------------------------------------------------------------------

def _call_llm_with_retry(llm: ChatGoogleGenerativeAI, prompt: str) -> dict:
    """
    Call the LLM with automatic retry on JSON parse failure.
    
    Strategy:
    1. Try normal call
    2. On failure: retry with stricter "return only JSON" instruction
    3. On second failure: return fallback (not_found=True)
    """
    try:
        result = _call_llm(llm, prompt)
        return result
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Retry with stricter instruction
    strict_prompt = (
        prompt
        + "\n\nIMPORTANT: Return ONLY valid JSON matching the schema. "
        "No extra text, no markdown, no explanation — just the raw JSON object."
    )
    try:
        result = _call_llm(llm, strict_prompt)
        return result
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # All retries failed — return safe fallback
    return {
        "answer": "",
        "not_found": True,
        "conflict": False,
        "conflicting_values": [],
        "citations": [],
        "confidence": "Low",
    }


def _call_llm(llm: ChatGoogleGenerativeAI, prompt: str) -> dict:
    """
    Single LLM call: invoke Gemini, parse JSON, validate schema.
    
    Raises ValueError if required keys are missing.
    """
    response = llm.invoke([
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    raw = _extract_text(response.content)
    parsed = _parse_json_response(raw)

    # Validate required keys exist
    required_keys = {"answer", "not_found", "conflict", "citations", "confidence"}
    if not required_keys.issubset(parsed.keys()):
        raise ValueError(f"Missing required keys: {required_keys - parsed.keys()}")

    # Set defaults for optional fields
    parsed.setdefault("conflicting_values", [])
    parsed.setdefault("citations", [])

    # Validate confidence value
    valid_confidence = {"High", "Medium", "Low"}
    if parsed.get("confidence") not in valid_confidence:
        parsed["confidence"] = "Low"

    return parsed
