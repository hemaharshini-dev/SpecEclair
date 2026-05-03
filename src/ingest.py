"""
Parses dataset.pdf, chunks by standard, embeds with sentence-transformers, saves index/chunks.pkl.
Run once: python src/ingest.py
"""

import os
import re
import pickle
import sys
import logging

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import fitz
from sentence_transformers import SentenceTransformer
from src.config import PDF_PATH, INDEX_PATH, EMBED_MODEL_NAME

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HEADER_RE = re.compile(r"SUMMARY\s+OF", re.IGNORECASE)
IS_ID_RE = re.compile(r"IS\s+(\d+(?:\s*\(Part\s*\d+\))?)\s*[:\s]\s*(\d{4})", re.IGNORECASE)


def fix_part_case(s: str) -> str:
    """Ensure (Part N) is title-cased, not upper."""
    return re.sub(r"\(PART\s*(\d+)\)", lambda m: f"(Part {m.group(1)})", s)


def normalize_id(raw: str) -> str:
    # Remove extra whitespace
    raw = re.sub(r"\s+", " ", raw.strip())
    # Handle different separators between number and year
    raw = re.sub(r"[:\-]\s*", ": ", raw)
    
    m = IS_ID_RE.search(raw)
    if not m:
        return raw.strip()
    
    number = m.group(1).strip()
    # Ensure space before (Part ...)
    number = re.sub(r"(\d+)\s*\(", r"\1 (", number)
    # Fix PART -> Part
    number = re.sub(r"\(PART\s*(\d+)\)", lambda m2: f"(Part {m2.group(1)})", number, flags=re.IGNORECASE)
    
    year = m.group(2).strip()
    return f"IS {number}: {year}"


def extract_title(text: str) -> str:
    # Remove standard ID prefix if present
    text = re.sub(r"^IS\s+\d+.*?:\s*\d{4}\s*", "", text, flags=re.IGNORECASE).strip()
    
    title = re.sub(r"\(.*?revision.*?\)", "", text, flags=re.IGNORECASE).strip()
    # Take only first line
    title = title.split("\n")[0].strip()
    # Remove any leading dashes or colons
    title = re.sub(r"^[ \-:]+", "", title).strip()
    return title if title else "BIS Standard"


def parse_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    logger.info(f"PDF loaded: {doc.page_count} pages")

    # Build full text from page 13 onward (skip front matter)
    full_text = ""
    page_offsets = []
    for i in range(12, doc.page_count):
        t = doc[i].get_text()
        page_offsets.append((len(full_text), i + 1))
        full_text += t + "\n"
    doc.close()

    boundaries = [m.start() for m in re.finditer(HEADER_RE, full_text)]
    if not boundaries:
        logger.error("No 'SUMMARY OF' headers found. Check PDF path.")
        sys.exit(1)

    logger.info(f"Found {len(boundaries)} standard boundaries")

    chunks = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(full_text)
        block = full_text[start:end]

        lines = block.split("\n")
        # Grab first 4 lines to find IS id
        header_text = " ".join(lines[:4])
        id_match = IS_ID_RE.search(header_text)
        if not id_match:
            continue

        standard_id = normalize_id(id_match.group(0))
        title = extract_title(header_text[id_match.end():])
        body = "\n".join(lines[2:]).strip()

        # Determine page start
        page_start = 1
        for offset, pnum in reversed(page_offsets):
            if offset <= start:
                page_start = pnum
                break

        chunks.append({
            "standard_id": standard_id,
            "title": title,
            "text": body,
            "page_start": page_start,
        })

    logger.info(f"Parsed {len(chunks)} standard chunks")
    return chunks


def embed_chunks(chunks: list) -> list:
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [
        f"{c['standard_id']} {c['title']}\n{c['text'][:2000]}"
        for c in chunks
    ]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    logger.info("Embedding complete.")
    return chunks


def main():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    chunks = parse_pdf(PDF_PATH)
    chunks = embed_chunks(chunks)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"Index saved to {INDEX_PATH} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
