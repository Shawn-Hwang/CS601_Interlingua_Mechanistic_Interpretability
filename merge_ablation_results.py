"""
One-off script: merge per-layer intermediate ablation JSON files for
relational and factual_inference back into ablation_results.json,
which currently only contains pattern_completion results.

Usage:
    python merge_ablation_results.py
"""

import json
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent / "results" / "ablation"
OUTPUT_PATH = ABLATION_DIR / "ablation_results.json"

# Task types whose intermediate files need merging
TASK_TYPES_TO_MERGE = ["relational", "factual_inference"]


def collect_intermediate_results(task_type: str) -> list[dict]:
    """Load all ablation_{task_type}_L{layer}.json files and concatenate."""
    results = []
    layer = 0
    while True:
        path = ABLATION_DIR / f"ablation_{task_type}_L{layer}.json"
        if not path.exists():
            break
        with open(path, "r") as f:
            layer_results = json.load(f)
        results.extend(layer_results)
        layer += 1
    return results


def main():
    # Load current ablation_results.json (pattern_completion only)
    with open(OUTPUT_PATH, "r") as f:
        existing = json.load(f)

    existing_tasks = set(r["task_type"] for r in existing)
    print(f"Current ablation_results.json has: {existing_tasks}")
    print(f"  Records: {len(existing)}")

    # Collect intermediate results for the missing task types
    merged = []
    for task_type in TASK_TYPES_TO_MERGE:
        results = collect_intermediate_results(task_type)
        if not results:
            print(f"  WARNING: no intermediate files found for {task_type}")
            continue
        layers = set(r["layer"] for r in results)
        print(f"  Loaded {task_type}: {len(results)} records across {len(layers)} layers")
        merged.extend(results)

    # Combine: intermediate (relational, factual_inference) + existing (pattern_completion)
    all_results = merged + existing

    # Sort by task_type, then layer, then condition for readability
    condition_order = [
        "en_ablated_es", "en_ablated_en", "random_ablated_es",
        "baseline_es", "baseline_en", "high_var_ablated_es",
    ]
    all_results.sort(key=lambda r: (
        r["task_type"],
        r["layer"],
        condition_order.index(r["condition"]) if r["condition"] in condition_order else 99,
    ))

    # Write back
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    final_tasks = set(r["task_type"] for r in all_results)
    print(f"\nMerged ablation_results.json now has: {final_tasks}")
    print(f"  Total records: {len(all_results)}")


if __name__ == "__main__":
    main()
