"""
Loads index/chunks.pkl and retrieves top-K chunks via cosine similarity.
Embeddings are already L2-normalized at ingest time.
"""

import os
import pickle
import sys

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "index", "chunks.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"

_chunks = None
_embeddings = None  # shape (N, D), L2-normalized
_model = None


def _load_index():
    global _chunks, _embeddings, _model

    if not os.path.exists(INDEX_PATH):
        sys.exit(f"ERROR: Index not found at {INDEX_PATH}. Run: python src/ingest.py")

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    _chunks = data
    _embeddings = np.array([c["embedding"] for c in data], dtype=np.float32)
    _model = SentenceTransformer(EMBED_MODEL)


def retrieve(query: str, top_k: int = 5) -> list:
    global _chunks, _embeddings, _model

    if _chunks is None:
        _load_index()

    q_vec = _model.encode([query], normalize_embeddings=True)[0]
    scores = _embeddings @ q_vec  # cosine similarity (already normalized)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        c = _chunks[i]
        results.append({
            "standard_id": c["standard_id"],
            "title": c["title"],
            "text": c["text"],
            "page_start": c["page_start"],
            "score": float(scores[i]),
        })

    results.sort(key=lambda x: (-x["score"], x["standard_id"]))
    return results
