"""
Judge entry point.
Usage: python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not parse input JSON: {e}")
        sys.exit(1)

    # Import pipeline here so index loads once before the loop
    from src.pipeline import run
    # Warm up: run a dummy query so model is loaded before timing starts
    print("Warming up pipeline...")
    try:
        run("warmup", top_k=1)
    except Exception:
        pass
    print("Pipeline ready.")

    results = []
    for item in data:
        query = item.get("query", "")
        item_id = item.get("id", "")

        t_start = time.time()
        try:
            pipeline_out = run(query, top_k=5)
            retrieved_standards = [r["standard_id"] for r in pipeline_out]
        except Exception as e:
            print(f"ERROR on query {item_id}: {e}")
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

        results.append(result)
        print(f"[{item_id}] {latency:.2f}s -> {retrieved_standards[:3]}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
