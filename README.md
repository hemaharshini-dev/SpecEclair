# BIS Standards Recommendation Engine (SpecEclair)

AI-powered RAG system that maps product descriptions to relevant Bureau of Indian Standards (BIS) for Building Materials.

Built for the **BIS x Sigma Squad AI Hackathon** — Track: RAG / AI.

---

## Evaluation Results (Public Test Set)

| Metric | Target | Our Score |
|---|---|---|
| Hit Rate @3 | >80% | **100%** ✅ |
| MRR @5 | >0.7 | **0.95** ✅ |
| Avg Latency | <5s | **0.91s** ✅ |

---

## Key Improvements (v2.0)

- **Hybrid Search Strategy:** Combines semantic vector similarity (AI) with keyword matching to ensure exact standard IDs are never missed.
- **Anti-Hallucination Guardrails:** Uses Pydantic-based structured outputs to force the LLM to only recommend standards present in the retrieved context.
- **Async Inference Pipeline:** Process multiple queries in parallel for significantly faster evaluation on large datasets.
- **Robust ID Normalization:** Advanced parsing logic to handle various formats of BIS Standard IDs (spaces, colons, parts, etc.).
- **Centralized Configuration:** All system parameters and model settings are unified in `src/config.py`.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
Edit `.env`:
```env
GROQ_API_KEY=your_groq_key
```

### 3. Build the index (run once)
```bash
python src/ingest.py
```
This parses `dataset.pdf`, chunks by BIS standard, embeds with `all-MiniLM-L6-v2` (local), and saves `index/chunks.pkl`.

---

## Usage

### Run the UI
```bash
streamlit run src/app.py
```

### Run Evaluation
```bash
# Generate results with async inference
python inference.py --input public_test_set.json --output data/public_test_results.json

# Score with eval script
python eval_script.py --results data/public_test_results.json
```

---

## Architecture

```
Product Description
       │
       ▼
Hybrid Retrieval (Semantic + Keyword)
over index/chunks.pkl (559 standards)
       │  (top-5 BIS standard chunks)
       ▼
Groq llama-3.3-70b-versatile
       │  (Pydantic-structured rationale)
       ▼
Validated Results
(standard_id + rationale)
```

**Chunking strategy:** One chunk per BIS standard summary. 559 standards indexed. 

**Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` (Local CPU).

**LLM Logic:** Uses `langchain-groq` with `with_structured_output` for deterministic and valid JSON responses.

---

## Project Structure
```
├── src/
│   ├── config.py       # Centralized settings & paths
│   ├── ingest.py       # PDF -> chunks -> embeddings -> index
│   ├── retriever.py    # Hybrid search logic
│   ├── pipeline.py     # RAG pipeline (Async/Sync)
│   └── app.py          # Enhanced Streamlit UI
├── data/
│   └── public_test_results.json
├── tests/
│   └── test_utils.py   # Unit tests for parsing logic
├── index/              # chunks.pkl (generated index)
├── inference.py        # Async judge entry point
├── eval_script.py      # Organizer evaluation script
├── dataset.pdf         # BIS SP 21 (2005)
├── improvements.md     # Design document for enhancements
├── requirements.txt
└── .env                # API keys
```
