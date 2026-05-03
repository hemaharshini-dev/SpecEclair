"""
Judge entry point.
Usage: python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import argparse
import json
import os
import sys
import time
import logging
import asyncio


async def process_item(item, run_fn, top_k, logger):
    query = item.get("query", "")
    item_id = item.get("id", "")

    t_start = time.time()
    try:
        pipeline_out = await run_fn(query, top_k=top_k)
        retrieved_standards = [r["standard_id"] for r in pipeline_out]
    except Exception as e:
        logger.error(f"Error on query {item_id}: {e}")
        retrieved_standards = []
    latency = round(time.time() - t_start, 4)

    result = {
        "id": item_id,
        "query": query,
        "retrieved_standards": retrieved_standards,
        "latency_seconds": latency,
    }
    # Pass through expected_standards if present (needed by eval_script.py)
    if "expected_standards" in item:
        result["expected_standards"] = item["expected_standards"]
    
    logger.info(f"[{item_id}] {latency:.2f}s -> {retrieved_standards[:3]}")
    return result


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not parse input JSON: {e}")
        sys.exit(1)

    # Import pipeline here so index loads once before the loop
    from src.pipeline import arun
    from src.config import TOP_K
    
    # Warm up: run a dummy query so model is loaded before timing starts
    logger.info("Warming up pipeline...")
    try:
        await arun("warmup", top_k=1)
    except Exception:
        pass
    logger.info("Pipeline ready.")

    # Process all items in parallel (respecting rate limits implicitly by using async)
    # For very large datasets, you might want to use a semaphore to limit concurrency
    tasks = [process_item(item, arun, TOP_K, logger) for item in data]
    results = await asyncio.gather(*tasks)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main_async())
