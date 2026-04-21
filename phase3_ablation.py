"""
Phase 3: Causal Ablation Experiments.

Applies directional mean ablation of the English direction at each layer
and measures impact on reasoning accuracy. Runs four experimental conditions:
    (a) English direction ablated, Spanish reasoning
    (b) English direction ablated, English reasoning
    (c) Random direction (matched norm) ablated, Spanish reasoning
    (d) No ablation — baseline

Also runs an additional high-variance control.

Usage:
    python phase3_ablation.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config
import utils


def load_directions(lang: str = "es") -> dict[int, torch.Tensor]:
    """Load saved English direction vectors for each layer."""
    method = config.DIRECTION_METHOD
    directions = {}
    layer = 0
    while True:
        path = config.DIRECTIONS_DIR / f"english_direction_{lang}_L{layer}_{method}.pt"
        if not path.exists():
            break
        directions[layer] = utils.load_tensor(path)
        layer += 1
    print(f"Loaded {len(directions)} direction vectors ({method}, {lang})")
    return directions


def load_reasoning_dataset(task_type: str) -> list[dict]:
    """Load a reasoning dataset from Phase 2."""
    path = config.DATA_DIR / f"{task_type}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples for task: {task_type}")
    return data


def compute_mean_projections(
    model, directions: dict[int, torch.Tensor], texts: list[str]
) -> dict[int, float]:
    """
    Compute the mean projection of activations onto the English direction
    at each layer, over a set of reference texts. This becomes the mu
    constant in the ablation formula.
    """
    activations = utils.get_residual_activations(model, texts)
    max_seq = activations[0].shape[1]
    mask = utils.get_content_mask(model, texts, max_seq)

    mean_projections = {}
    for L, d in directions.items():
        avg_acts = utils.average_over_positions(activations[L], mask)
        # Mean projection: average of (each sample's mean activation dotted with direction)
        projections = torch.mv(avg_acts, d)  # [n_samples]
        mean_projections[L] = projections.mean().item()

    return mean_projections


def compute_high_variance_direction(
    model, texts: list[str], english_direction: torch.Tensor, layer: int
) -> torch.Tensor:
    """
    Find a high-variance direction at the given layer that has low cosine
    similarity with the English direction. Used as a specificity control.
    """
    activations = utils.get_residual_activations(model, texts)
    max_seq = activations[layer].shape[1]
    mask = utils.get_content_mask(model, texts, max_seq)
    avg_acts = utils.average_over_positions(activations[layer], mask)

    # PCA: find top principal components
    centered = avg_acts - avg_acts.mean(dim=0)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Find first PC that has low cosine similarity with English direction
    for i in range(min(10, Vh.shape[0])):
        pc = Vh[i]
        cos_sim = torch.nn.functional.cosine_similarity(
            pc.unsqueeze(0), english_direction.unsqueeze(0)
        ).item()
        if abs(cos_sim) < 0.3:
            return pc / pc.norm()

    # Fallback: orthogonalize the top PC against the English direction
    pc = Vh[0]
    pc = pc - (pc @ english_direction) * english_direction
    return pc / pc.norm()


def run_condition(
    model,
    examples: list[dict],
    prompt_key: str,
    answer_key: str,
    fwd_hooks: list | None,
    condition_name: str,
) -> dict:
    """
    Evaluate model accuracy under a specific condition.

    Returns:
        {"condition": str, "n_correct": int, "n_total": int, "accuracy": float}
    """
    n_correct = 0
    n_total = len(examples)
    
    print("Iterating thru examples...")
    for ex in tqdm(examples):
        correct, _ = utils.evaluate_completion(
            model, ex[prompt_key], ex[answer_key], fwd_hooks=fwd_hooks
        )
        if correct:
            n_correct += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return {
        "condition": condition_name,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
    }


def run_ablation_experiment(
    model,
    task_type: str,
    examples: list[dict],
    directions: dict[int, torch.Tensor],
    mean_projections: dict[int, float],
    reference_texts: list[str],
) -> list[dict]:
    """
    Run all ablation conditions for one task type across all layers.
    """
    n_layers = len(directions)
    d_model = model.cfg.d_model
    results = []

    # Pre-generate random directions (shared across layers and examples)
    random_dirs = [utils.random_direction(d_model) for _ in range(config.N_RANDOM_DIRECTIONS)]

    print(f'Ablation [{task_type}]')
    for layer in range(n_layers):
        print(f"Running layer {layer}")
        d_hat = directions[layer]
        mu = mean_projections[layer]
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Condition (a): English direction ablated + Spanish reasoning
        print(f"Running condition (a)...")
        ablation_hook = utils.make_ablation_hook(d_hat, mu)
        result_a = run_condition(
            model, examples, "prompt_es", "answer_es",
            fwd_hooks=[(hook_name, ablation_hook)],
            condition_name="en_ablated_es",
        )
        result_a.update({"task_type": task_type, "layer": layer})
        results.append(result_a)

        # Condition (b): English direction ablated + English reasoning
        print(f"Running condition (b)...")
        result_b = run_condition(
            model, examples, "prompt_en", "answer_en",
            fwd_hooks=[(hook_name, ablation_hook)],
            condition_name="en_ablated_en",
        )
        result_b.update({"task_type": task_type, "layer": layer})
        results.append(result_b)

        # Condition (c): Random direction ablated + Spanish reasoning
        # Average accuracy over N_RANDOM_DIRECTIONS random directions
        print(f"Running condition (c)...")
        random_accuracies = []
        for rd in random_dirs:
            rand_hook = utils.make_ablation_hook(rd, 0.0)
            r = run_condition(
                model, examples, "prompt_es", "answer_es",
                fwd_hooks=[(hook_name, rand_hook)],
                condition_name="random_ablated_es",
            )
            random_accuracies.append(r["accuracy"])

        result_c = {
            "condition": "random_ablated_es",
            "task_type": task_type,
            "layer": layer,
            "accuracy": float(np.mean(random_accuracies)),
            "accuracy_std": float(np.std(random_accuracies)),
            "n_correct": -1,  # Averaged, not meaningful as single int
            "n_total": len(examples),
        }
        results.append(result_c)

        # Condition (d): No ablation — baseline
        print(f"Running condition (d)...")
        result_d = run_condition(
            model, examples, "prompt_es", "answer_es",
            fwd_hooks=None,
            condition_name="baseline_es",
        )
        result_d.update({"task_type": task_type, "layer": layer})
        results.append(result_d)

        result_d_en = run_condition(
            model, examples, "prompt_en", "answer_en",
            fwd_hooks=None,
            condition_name="baseline_en",
        )
        result_d_en.update({"task_type": task_type, "layer": layer})
        results.append(result_d_en)

        # Additional control: high-variance non-English-aligned direction
        print(f"Running condition (f)...")
        hv_dir = compute_high_variance_direction(
            model, reference_texts, d_hat, layer
        )
        hv_hook = utils.make_ablation_hook(hv_dir, 0.0)
        result_hv = run_condition(
            model, examples, "prompt_es", "answer_es",
            fwd_hooks=[(hook_name, hv_hook)],
            condition_name="high_var_ablated_es",
        )
        result_hv.update({"task_type": task_type, "layer": layer})
        results.append(result_hv)

        layer_path = config.ABLATION_DIR / f"ablation_{task_type}_L{layer}.json"
        layer_results = [r for r in results if r["layer"] == layer]
        with open(layer_path, "w") as f:
            json.dump(layer_results, f, indent=2)

    return results


def main():
    config.ensure_dirs()
    model = utils.load_model()

    # Load English directions (computed against Spanish in Phase 1)
    directions = load_directions("es")

    all_results = []
    # task_types = ["relational", "factual_inference", "pattern_completion"]
    task_types = ["pattern_completion"]

    for task_type in task_types:
        print(f"\n{'='*60}")
        print(f"Task: {task_type}")
        print(f"{'='*60}")

        examples = load_reasoning_dataset(task_type)

        # Doing only 10 examples per task
        examples = examples[:10]

        # Compute mean projections using Spanish prompts as reference
        es_texts = [ex["prompt_es"] for ex in examples]
        print("Computing mean projections...")
        mean_projections = compute_mean_projections(model, directions, es_texts)

        # Run all conditions
        results = run_ablation_experiment(
            model, task_type, examples, directions, mean_projections, es_texts
        )
        all_results.extend(results)

        # Print summary for this task
        print(f"\nSummary for {task_type}:")
        for layer in range(len(directions)):
            layer_results = [r for r in results if r["layer"] == layer]
            for r in layer_results:
                print(
                    f"  L{layer:2d} | {r['condition']:25s} | "
                    f"acc={r['accuracy']:.2f}"
                )

    # Save all results
    output_path = config.ABLATION_DIR / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPhase 3 complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
