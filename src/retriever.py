"""
Loads index/chunks.pkl and retrieves top-K chunks via hybrid search.
Combines semantic similarity (all-MiniLM-L6-v2) with keyword matching.
"""

import os
import pickle
import sys
import logging
import re

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from src.config import INDEX_PATH, EMBED_MODEL_NAME, TOP_K, ALPHA

logger = logging.getLogger(__name__)

_chunks = None
_embeddings = None  # shape (N, D), L2-normalized
_model = None


def _load_index():
    global _chunks, _embeddings, _model

    if _chunks is not None:
        return

    if not os.path.exists(INDEX_PATH):
        logger.error(f"Index not found at {INDEX_PATH}. Run: python src/ingest.py")
        sys.exit(1)

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    _chunks = data
    _embeddings = np.array([c["embedding"] for c in data], dtype=np.float32)

    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info(f"Loaded index with {len(_chunks)} chunks")


def get_keyword_score(query: str, chunk: dict) -> float:
    """Simple keyword overlap score between query and standard ID/title."""
    query_norm = query.lower()
    # Check for exact standard ID match (highest boost)
    if chunk["standard_id"].lower() in query_norm:
        return 1.0
    
    # Check for word overlap in title and ID
    q_words = set(re.findall(r"\w+", query_norm))
    if not q_words:
        return 0.0
    
    target_text = f"{chunk['standard_id']} {chunk['title']}".lower()
    t_words = set(re.findall(r"\w+", target_text))
    
    overlap = q_words.intersection(t_words)
    return len(overlap) / len(q_words)


def retrieve(query: str, top_k: int = TOP_K) -> list:
    global _chunks, _embeddings, _model

    if _chunks is None:
        _load_index()

    # 1. Semantic Search
    q_vec = _model.encode([query], normalize_embeddings=True)[0]
    semantic_scores = _embeddings @ q_vec  # cosine similarity (already normalized)

    # 2. Hybrid Combining
    final_results = []
    for i, chunk in enumerate(_chunks):
        s_score = float(semantic_scores[i])
        k_score = get_keyword_score(query, chunk)
        
        # Combine scores
        combined = (ALPHA * s_score) + ((1 - ALPHA) * k_score)
        
        final_results.append({
            "standard_id": chunk["standard_id"],
            "title": chunk["title"],
            "text": chunk["text"],
            "page_start": chunk["page_start"],
            "score": combined,
            "semantic_score": s_score,
            "keyword_score": k_score
        })

    # Sort by combined score
    final_results.sort(key=lambda x: (-x["score"], x["standard_id"]))
    
    return final_results[:top_k]
