# Suggested Improvements for SpecEclair

Based on the analysis of the current codebase, here are several areas where the BIS Standards Recommendation Engine can be improved for better performance, robustness, and maintainability.

## 1. Retrieval Strategy Enhancements

### Hybrid Search (BM25 + Semantic)
*   **Current State:** Uses only semantic search with `all-MiniLM-L6-v2`.
*   **Improvement:** Implement hybrid search by combining vector similarity scores with BM25 keyword matching. This is particularly effective for matching exact standard numbers (e.g., "IS 269") or specific technical terms that might be "diluted" in a dense vector representation.
*   **Benefit:** Increases Hit Rate @3 and MRR @5.

### Cross-Encoder Re-ranking
*   **Current State:** Returns top-K based solely on cosine similarity.
*   **Improvement:** Use a Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the top 10-20 candidates retrieved by the Bi-Encoder. Cross-encoders are much more accurate as they process the query and document together.
*   **Benefit:** Significantly improves MRR @5 and ensures the most relevant results are at the top.

### Sub-chunking for Long Standards
*   **Current State:** One chunk per standard summary.
*   **Improvement:** For standards with very long summaries, implement sub-chunking (e.g., 500-1000 tokens with overlap) while retaining the parent standard ID as metadata.
*   **Benefit:** Improves the granularity of semantic matching and prevents truncation of important details.

## 2. LLM & Pipeline Robustness

### Structured Outputs with Pydantic
*   **Current State:** Manual regex-based JSON extraction and repair.
*   **Improvement:** Use LangChain's `with_structured_output` or a similar library with Pydantic models to enforce the output schema. This leverages LLM features (like tool calling or JSON mode) for guaranteed valid JSON.
*   **Benefit:** Eliminates JSON parsing errors and the need for fragile regex "repairs".

### Async Inference
*   **Current State:** Synchronous processing in `inference.py`.
*   **Improvement:** Use `asyncio` and async LLM clients to process multiple queries in parallel (where rate limits allow). 
*   **Benefit:** Reduces total execution time for large datasets, though individual query latency remains similar.

### Enhanced Prompt Engineering
*   **Current State:** Basic prompt with manual constraints (e.g., "no double quotes").
*   **Improvement:** Use Few-Shot prompting with high-quality examples of query-rationale pairs. Implement a systematic prompt evaluation framework (like LangSmith or Phoenix) to iterate on prompt quality.
*   **Benefit:** Improves the quality and relevance of the generated rationales.

## 3. Code Quality & Engineering

### Configuration Management
*   **Current State:** Hardcoded paths and model names in multiple files.
*   **Improvement:** Centralize configuration using a `config.yaml` file or a dedicated `config.py` module. Use Pydantic-Settings for environment variable validation.
*   **Benefit:** Makes the system easier to configure and deploy in different environments.

### Comprehensive Logging
*   **Current State:** Replace `print` with `logging`.
*   **Improvement:** Replace `print` statements with the Python `logging` module. Configure different levels (INFO, DEBUG, ERROR) and log to both console and file.
*   **Benefit:** Better traceability and easier debugging in production/evaluation.

### Unit & Integration Testing
*   **Current State:** No formal tests.
*   **Improvement:** Add a `tests/` directory with:
    *   **Unit tests** for `normalize_id`, `extract_title`, and JSON parsing logic.
    *   **Integration tests** for the retriever and full pipeline using a small mock dataset.
*   **Benefit:** Ensures regressions are not introduced during refactoring or when updating the index.

## 4. UI/UX Improvements

### Advanced Result Visualization
*   **Current State:** Basic cards with ID, Title, and Rationale.
*   **Improvement:** 
    *   Highlight matching keywords in the summary text.
    *   Provide links to the full standard PDF (if available) or official BIS site.
    *   Add a "Copy to Clipboard" button for the standard ID and rationale.
*   **Benefit:** Enhances usability for MSE owners who need to use these results for official documentation.

### Query Refinement / Clarification
*   **Current State:** Direct search on raw input.
*   **Improvement:** Use the LLM to "rewrite" or "expand" the user's query before retrieval (e.g., expanding abbreviations like "PPC" to "Portland Pozzolana Cement").
*   **Benefit:** Improves retrieval performance for short or vague queries.

## 5. Maintenance

### Automated Indexing Pipeline
*   **Current State:** Manual `ingest.py` run.
*   **Improvement:** Create a script that watches for changes in the `dataset.pdf` (or a folder of PDFs) and automatically rebuilds the index if needed.
*   **Benefit:** Ensures the system stays up-to-date with the latest BIS standards effortlessly.

