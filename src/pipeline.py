"""
Full RAG pipeline: retrieve top-5 chunks → Groq LLM → validated output.
Uses structured output with Pydantic for robustness and supports async.
"""

import os
import logging
import sys
import asyncio
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.config import GROQ_API_KEY, LLM_MODEL_NAME, TOP_K

load_dotenv()

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not set in .env")
    sys.exit(1)

# ── Pydantic Models for Structured Output ──
class Recommendation(BaseModel):
    standard_id: str = Field(description="The BIS standard ID, e.g., IS 269: 1989")
    rationale: str = Field(description="A short sentence explaining why this standard is relevant to the product.")

class RecommendationList(BaseModel):
    recommendations: List[Recommendation]

_llm = None
_structured_llm = None

def _get_llm():
    global _llm, _structured_llm
    if _llm is None:
        logger.info(f"Initializing LLM: {LLM_MODEL_NAME}")
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model=LLM_MODEL_NAME,
            temperature=0,
            api_key=GROQ_API_KEY,
        )
        _structured_llm = _llm.with_structured_output(RecommendationList)
    return _structured_llm


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
- Select 3 to 5 standards that genuinely apply to the product.
- Use only IDs from the list above.
- Provide a short, one-sentence rationale for each.
"""


def _normalize_id(sid: str) -> str:
    """Normalize ID for comparison (remove spaces, lowercase)."""
    return sid.replace(" ", "").replace(":", "").lower()


def _process_response(response: RecommendationList, candidates: list[dict], top_k: int) -> list[dict]:
    valid_ids_norm = {_normalize_id(c["standard_id"]): c["standard_id"] for c in candidates}
    id_to_candidate = {c["standard_id"]: c for c in candidates}
    
    results = []
    for item in response.recommendations:
        sid_raw = item.standard_id.strip()
        sid_norm = _normalize_id(sid_raw)
        rationale = item.rationale.strip()
        
        if sid_norm in valid_ids_norm:
            actual_sid = valid_ids_norm[sid_norm]
            results.append({
                "standard_id": actual_sid,
                "title": id_to_candidate[actual_sid]["title"],
                "rationale": rationale,
                "score": id_to_candidate[actual_sid]["score"],
            })
        else:
            logger.warning(f"LLM returned invalid standard_id: {sid_raw}")

    # Pad with remaining candidates if LLM returned fewer than 3 valid ones
    if len(results) < 3:
        logger.info("Padding results with next-best retrieval candidates")
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


async def arun(query: str, top_k: int = TOP_K) -> list[dict]:
    """Async version of the pipeline."""
    from src.retriever import retrieve
    # retriever is synchronous (numpy/sentence-transformers)
    candidates = retrieve(query, top_k=top_k)
    prompt = _build_prompt(query, candidates)
    structured_llm = _get_llm()

    try:
        logger.info(f"Invoking LLM (async) for query: {query[:50]}...")
        response = await structured_llm.ainvoke(prompt)
        return _process_response(response, candidates, top_k)
    except Exception as e:
        logger.error(f"LLM error in arun: {e}")
        return [
            {
                "standard_id": c["standard_id"],
                "title": c["title"],
                "rationale": "Retrieved as a relevant standard based on semantic similarity.",
                "score": c["score"],
            }
            for c in candidates[:top_k]
        ]


def run(query: str, top_k: int = TOP_K) -> list[dict]:
    """Synchronous version of the pipeline."""
    from src.retriever import retrieve
    candidates = retrieve(query, top_k=top_k)
    prompt = _build_prompt(query, candidates)
    structured_llm = _get_llm()

    try:
        logger.info(f"Invoking LLM (sync) for query: {query[:50]}...")
        response = structured_llm.invoke(prompt)
        return _process_response(response, candidates, top_k)
    except Exception as e:
        logger.error(f"LLM error in run: {e}")
        return [
            {
                "standard_id": c["standard_id"],
                "title": c["title"],
                "rationale": "Retrieved as a relevant standard based on semantic similarity.",
                "score": c["score"],
            }
            for c in candidates[:top_k]
        ]
