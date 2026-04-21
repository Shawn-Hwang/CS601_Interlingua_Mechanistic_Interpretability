"""
Phase 1: Finding the English Direction.

Extracts the "English direction" from the residual stream at each layer
using parallel sentence pairs from FLORES-200. Computes via mean difference
and logistic regression, then checks convergence between the two methods.
Extends to French and Chinese for generalization testing.

Usage:
    python phase1_extract_direction.py
"""

import json
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import config
import utils


def load_flores_pairs(flores_config: str) -> tuple[list[str], list[str]]:
    """
    Load parallel sentence pairs from FLORES-200.

    Args:
        flores_config: e.g. "eng_Latn-spa_Latn"

    Returns:
        (english_sentences, other_language_sentences)
    """
    local_path = config.DATA_DIR / "flores" / flores_config.replace("/", "_")
    try:
        dataset = load_from_disk(str(local_path))
        print(f"Loaded {flores_config} from local disk.")
    except Exception:
        print(f"Downloading {flores_config} from HuggingFace...")
        dataset = load_dataset(
            config.FLORES_DATASET, flores_config, split=config.FLORES_SPLIT
        )
        dataset.save_to_disk(str(local_path))

    # Column names follow the pattern: sentence_{lang_code}
    lang_codes = flores_config.split("-")
    en_col = f"sentence_{lang_codes[0]}"
    other_col = f"sentence_{lang_codes[1]}"

    n = min(config.N_SAMPLES_DIRECTION, len(dataset))
    en_texts = [dataset[i][en_col] for i in range(n)]
    other_texts = [dataset[i][other_col] for i in range(n)]

    print(f"Loaded {n} parallel pairs from {flores_config}")
    return en_texts, other_texts


def extract_directions_for_pair(
    model, en_texts: list[str], other_texts: list[str], lang_label: str
) -> dict:
    """
    Extract English directions at each layer for one language pair.

    Returns:
        Dict with keys:
            - "mean_diff": {layer: direction_tensor}
            - "logreg": {layer: direction_tensor}
            - "cosine_sim": {layer: float}
    """
    n_layers = model.cfg.n_layers

    print(f"\n--- Extracting activations for English sentences ({lang_label}) ---")
    en_activations = utils.get_residual_activations(model, en_texts)

    print(f"--- Extracting activations for {lang_label} sentences ---")
    other_activations = utils.get_residual_activations(model, other_texts)

    # Build content masks
    max_seq_en = en_activations[0].shape[1]
    max_seq_other = other_activations[0].shape[1]
    en_mask = utils.get_content_mask(model, en_texts, max_seq_en)
    other_mask = utils.get_content_mask(model, other_texts, max_seq_other)

    directions_md = {}
    directions_lr = {}
    cosine_sims = {}

    print(f"--- Computing directions at each layer ---")
    for L in tqdm(range(n_layers), desc="Layers"):
        # Average over content-word positions
        en_avg = utils.average_over_positions(en_activations[L], en_mask)
        other_avg = utils.average_over_positions(other_activations[L], other_mask)

        # Mean difference direction
        d_md = utils.compute_direction_mean_diff(en_avg, other_avg)
        directions_md[L] = d_md

        # Logistic regression direction
        d_lr = utils.compute_direction_logreg(en_avg, other_avg)
        directions_lr[L] = d_lr

        # Cosine similarity between the two methods
        cos = torch.nn.functional.cosine_similarity(
            d_md.unsqueeze(0), d_lr.unsqueeze(0)
        ).item()
        cosine_sims[L] = cos

    return {
        "mean_diff": directions_md,
        "logreg": directions_lr,
        "cosine_sim": cosine_sims,
    }


def save_directions(
    directions: dict, cosine_sims: dict, lang_label: str
) -> None:
    """Save direction vectors and convergence data to disk."""
    # Save direction vectors
    for method_name, layer_dirs in [
        ("mean_diff", directions["mean_diff"]),
        ("logreg", directions["logreg"]),
    ]:
        for L, d in layer_dirs.items():
            filename = f"english_direction_{lang_label}_L{L}_{method_name}.pt"
            utils.save_tensor(d, config.DIRECTIONS_DIR / filename)

    # Save convergence cosine similarities
    convergence_path = config.DIRECTIONS_DIR / f"convergence_{lang_label}.json"
    with open(convergence_path, "w") as f:
        json.dump(
            {str(k): v for k, v in cosine_sims.items()},
            f,
            indent=2,
        )
    print(f"Saved directions and convergence data for {lang_label}")


def main():
    config.ensure_dirs()
    model = utils.load_model()
    n_layers = model.cfg.n_layers

    # Process each language pair
    for lang_label, flores_cfg in config.FLORES_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Processing language pair: English vs {lang_label.upper()}")
        print(f"{'='*60}")

        en_texts, other_texts = load_flores_pairs(flores_cfg)
        results = extract_directions_for_pair(
            model, en_texts, other_texts, lang_label
        )
        save_directions(results, results["cosine_sim"], lang_label)

        # Print convergence summary
        print(f"\nConvergence (cosine sim between mean_diff and logreg):")
        for L in range(n_layers):
            cos = results["cosine_sim"][L]
            marker = " <-- peak" if cos == max(results["cosine_sim"].values()) else ""
            print(f"  Layer {L:2d}: {cos:.4f}{marker}")

    print("\nPhase 1 complete. Directions saved to:", config.DIRECTIONS_DIR)


if __name__ == "__main__":
    main()
