"""
Full RAG pipeline: retrieve top-5 chunks → Groq LLM → validated output.
"""

import os
import json
import re
import sys

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY not set in .env")

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=GROQ_API_KEY,
        )
    return _llm


def _build_prompt(query: str, candidates: list[dict]) -> str:
    context_lines = []
    for c in candidates:
        snippet = c["text"][:300].replace("\n", " ").strip()
        context_lines.append(
            f"- {c['standard_id']} | {c['title']}\n  Summary: {snippet}"
        )
    context = "\n".join(context_lines)

    return f"""You are a BIS (Bureau of Indian Standards) compliance expert helping Indian MSEs.

Given the product description below, select the most relevant standards ONLY from the retrieved list.

Product Description:
{query}

Retrieved Standards:
{context}

Instructions:
- Return ONLY a valid JSON array. No text before or after the JSON.
- Each object must have exactly two keys: "standard_id" and "rationale".
- "standard_id" must be copied exactly as shown above (e.g. "IS 269: 1989").
- "rationale" must be one short sentence. Do NOT use double quotes inside the rationale.
- Include 3 to 5 standards. Only include standards that genuinely apply.
- Do NOT invent or modify any standard ID. Use only IDs from the list above.

Respond with ONLY the JSON array, nothing else:
["""


def _extract_json(text: str) -> list:
    """Extract JSON array from LLM response with aggressive repair."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = match.group(0) if match else text
    # Remove trailing commas before ] or }
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: extract individual objects with regex
        objects = []
        for m in re.finditer(r'\{[^{}]+\}', candidate, re.DOTALL):
            try:
                obj = json.loads(re.sub(r",\s*}", "}", m.group(0)))
                if "standard_id" in obj:
                    objects.append(obj)
            except Exception:
                continue
        if objects:
            return objects
        raise


def run(query: str, top_k: int = 5) -> list[dict]:
    """
    Returns list of {standard_id, title, rationale, score} dicts, ranked by relevance.
    Always returns at least top_k standard_ids even if LLM fails.
    """
    from src.retriever import retrieve  # lazy import to allow index loading once

    candidates = retrieve(query, top_k=top_k)
    valid_ids = {c["standard_id"] for c in candidates}
    id_to_candidate = {c["standard_id"]: c for c in candidates}

    prompt = _build_prompt(query, candidates)
    llm = _get_llm()

    try:
        response = llm.invoke(prompt)
        raw = response.content
        # If prompt ended with '[', prepend it back
        if not raw.strip().startswith('['):
            raw = '[' + raw
        parsed = _extract_json(raw)

        results = []
        for item in parsed:
            sid = item.get("standard_id", "").strip()
            rationale = item.get("rationale", "").strip()
            if sid in valid_ids:
                results.append({
                    "standard_id": sid,
                    "title": id_to_candidate[sid]["title"],
                    "rationale": rationale,
                    "score": id_to_candidate[sid]["score"],
                })

        # Pad with remaining candidates if LLM returned fewer than 3 valid ones
        if len(results) < 3:
            seen = {r["standard_id"] for r in results}
            for c in candidates:
                if c["standard_id"] not in seen:
                    results.append({
                        "standard_id": c["standard_id"],
                        "title": c["title"],
                        "rationale": "Retrieved as a relevant standard based on semantic similarity.",
                        "score": c["score"],
                    })
                if len(results) >= 3:
                    break

        return results[:top_k]

    except Exception as e:
        # Fallback: return top candidates with no LLM rationale
        print(f"LLM error (falling back to retrieval only): {e}")
        return [
            {
                "standard_id": c["standard_id"],
                "title": c["title"],
                "rationale": "Retrieved as a relevant standard based on semantic similarity.",
                "score": c["score"],
            }
            for c in candidates[:top_k]
        ]
