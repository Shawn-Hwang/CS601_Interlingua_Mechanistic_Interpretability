"""
Phase 1: Finding the English Direction.

Extracts the "English direction" from the residual stream at each layer
using parallel sentence pairs from FLORES-200. Computes via three methods:
  - Mean difference
  - Logistic regression
  - PCA on paired differences (RepE/LAT standard)

Includes mean-centering for anisotropy correction, held-out classification
accuracy, pairwise method convergence, and cross-language consistency analysis.

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

    Pipeline:
        1. Extract residual stream activations (once per language)
        2. Pool over token positions (mean or last_token)
        3. Mean-center for anisotropy correction
        4. Train/test split (75%/25%, deterministic)
        5. Compute directions via 3 methods on train set
        6. Pairwise method convergence (cosine similarity)
        7. Held-out classification accuracy on test set

    Returns:
        Dict with keys:
            - "mean_diff": {layer: direction_tensor}
            - "logreg": {layer: direction_tensor}
            - "pca": {layer: direction_tensor}
            - "convergence": {layer: {pair_name: cosine_sim}}
            - "classification_accuracy": {layer: {method: accuracy}}
    """
    n_layers = model.cfg.n_layers
    pooling = config.POOLING_STRATEGY

    # Step 1: Extract activations (expensive — done ONCE per language)
    print(f"\n--- Extracting activations for English sentences ({lang_label}) ---")
    en_activations = utils.get_residual_activations(model, en_texts)

    print(f"--- Extracting activations for {lang_label} sentences ---")
    other_activations = utils.get_residual_activations(model, other_texts)

    # Build content masks
    max_seq_en = en_activations[0].shape[1]
    max_seq_other = other_activations[0].shape[1]
    en_mask = utils.get_content_mask(model, en_texts, max_seq_en)
    other_mask = utils.get_content_mask(model, other_texts, max_seq_other)

    # Determine train/test split indices
    n_samples = len(en_texts)
    n_train = int(n_samples * 0.75)
    train_idx = slice(0, n_train)
    test_idx = slice(n_train, n_samples)
    print(f"Train/test split: {n_train} train, {n_samples - n_train} test")

    directions_md = {}
    directions_lr = {}
    directions_pca = {}
    convergence = {}
    classification_accuracy = {}

    print(f"--- Computing directions at each layer (pooling={pooling}) ---")
    for L in tqdm(range(n_layers), desc="Layers"):
        # Step 2: Pool activations over token positions
        en_avg = utils.pool_activations(en_activations[L], en_mask, pooling)
        other_avg = utils.pool_activations(
            other_activations[L], other_mask, pooling
        )

        # Step 3: Mean-center (anisotropy correction)
        global_mean = torch.cat([en_avg, other_avg], dim=0).mean(dim=0)
        en_centered = en_avg - global_mean
        other_centered = other_avg - global_mean

        # Step 4: Train/test split on pooled+centered activations
        en_train = en_centered[train_idx]
        en_test = en_centered[test_idx]
        other_train = other_centered[train_idx]
        other_test = other_centered[test_idx]

        # Step 5: Compute directions on train set
        d_md = utils.compute_direction_mean_diff(en_train, other_train)
        d_lr = utils.compute_direction_logreg(en_train, other_train)
        d_pca = utils.compute_direction_pca(en_train, other_train)

        directions_md[L] = d_md
        directions_lr[L] = d_lr
        directions_pca[L] = d_pca

        # Step 6: Pairwise cosine similarities between all 3 methods
        method_dirs = [("mean_diff", d_md), ("logreg", d_lr), ("pca", d_pca)]
        layer_sims = {}
        for i, (name_a, d_a) in enumerate(method_dirs):
            for name_b, d_b in method_dirs[i + 1 :]:
                cos = torch.nn.functional.cosine_similarity(
                    d_a.unsqueeze(0), d_b.unsqueeze(0)
                ).item()
                layer_sims[f"{name_a}_vs_{name_b}"] = cos
        convergence[L] = layer_sims

        # Step 7: Held-out classification accuracy
        layer_acc = {}
        for method_name, d in method_dirs:
            en_proj = en_test @ d
            other_proj = other_test @ d
            # Midpoint threshold from train-set class means
            en_train_mean = (en_train @ d).mean().item()
            other_train_mean = (other_train @ d).mean().item()
            threshold = (en_train_mean + other_train_mean) / 2.0
            # Classify: above threshold = English
            en_correct = (en_proj > threshold).sum().item()
            other_correct = (other_proj <= threshold).sum().item()
            total = len(en_proj) + len(other_proj)
            layer_acc[method_name] = (en_correct + other_correct) / total

        classification_accuracy[L] = layer_acc

    return {
        "mean_diff": directions_md,
        "logreg": directions_lr,
        "pca": directions_pca,
        "convergence": convergence,
        "classification_accuracy": classification_accuracy,
    }


def save_directions(directions: dict, lang_label: str) -> None:
    """Save direction vectors, convergence data, and classification accuracy."""
    # Save direction vectors for all three methods
    for method_name in ["mean_diff", "logreg", "pca"]:
        layer_dirs = directions[method_name]
        for L, d in layer_dirs.items():
            filename = f"english_direction_{lang_label}_L{L}_{method_name}.pt"
            utils.save_tensor(d, config.DIRECTIONS_DIR / filename)

    # Save convergence: pairwise cosine similarities for all 3 methods
    convergence_path = config.DIRECTIONS_DIR / f"convergence_{lang_label}.json"
    convergence_data = {
        str(L): sims for L, sims in directions["convergence"].items()
    }
    with open(convergence_path, "w") as f:
        json.dump(convergence_data, f, indent=2)

    # Save classification accuracy
    acc_path = (
        config.DIRECTIONS_DIR / f"classification_accuracy_{lang_label}.json"
    )
    acc_data = {
        str(L): accs
        for L, accs in directions["classification_accuracy"].items()
    }
    with open(acc_path, "w") as f:
        json.dump(acc_data, f, indent=2)

    print(f"Saved directions, convergence, and accuracy for {lang_label}")


def compute_cross_language_consistency(n_layers: int) -> dict:
    """
    Compute pairwise cosine similarity between English directions
    extracted from different language pairs, at each layer, for each method.

    Returns:
        {method: {layer_str: {lang_pair: cosine_sim}}}
    """
    langs = list(config.FLORES_CONFIGS.keys())
    methods = ["mean_diff", "logreg", "pca"]
    results = {}

    for method in methods:
        results[method] = {}
        for L in range(n_layers):
            dirs_by_lang = {}
            for lang in langs:
                path = (
                    config.DIRECTIONS_DIR
                    / f"english_direction_{lang}_L{L}_{method}.pt"
                )
                if path.exists():
                    dirs_by_lang[lang] = utils.load_tensor(path)

            pair_sims = {}
            lang_list = list(dirs_by_lang.keys())
            for i, lang_a in enumerate(lang_list):
                for lang_b in lang_list[i + 1 :]:
                    cos = torch.nn.functional.cosine_similarity(
                        dirs_by_lang[lang_a].unsqueeze(0),
                        dirs_by_lang[lang_b].unsqueeze(0),
                    ).item()
                    pair_sims[f"{lang_a}_vs_{lang_b}"] = cos

            results[method][str(L)] = pair_sims

    return results


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
        save_directions(results, lang_label)

        # Print convergence summary
        print(f"\nConvergence (pairwise cosine sim between methods):")
        for L in range(n_layers):
            sims = results["convergence"][L]
            sim_str = ", ".join(f"{k}: {v:.4f}" for k, v in sims.items())
            print(f"  Layer {L:2d}: {sim_str}")

        # Print classification accuracy summary
        print(f"\nClassification accuracy (held-out 25%):")
        for L in range(n_layers):
            accs = results["classification_accuracy"][L]
            acc_str = ", ".join(f"{k}: {v:.3f}" for k, v in accs.items())
            print(f"  Layer {L:2d}: {acc_str}")

    # Cross-language consistency analysis
    print(f"\n{'='*60}")
    print("Cross-language direction consistency analysis")
    print(f"{'='*60}")

    cross_lang = compute_cross_language_consistency(n_layers)

    cross_lang_path = config.DIRECTIONS_DIR / "cross_language_consistency.json"
    with open(cross_lang_path, "w") as f:
        json.dump(cross_lang, f, indent=2)
    print(f"Saved cross-language consistency to: {cross_lang_path}")

    # Print summary: average cross-language cosine per layer per method
    for method in ["mean_diff", "logreg", "pca"]:
        print(f"\n  Method: {method}")
        for L in range(n_layers):
            sims = cross_lang[method][str(L)]
            if sims:
                avg_sim = sum(sims.values()) / len(sims)
                print(f"    Layer {L:2d}: avg cross-lang cosine = {avg_sim:.4f}")

    print("\nPhase 1 complete. Directions saved to:", config.DIRECTIONS_DIR)


if __name__ == "__main__":
    main()
