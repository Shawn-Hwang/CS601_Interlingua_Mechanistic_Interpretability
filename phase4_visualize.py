"""
Phase 4: Visualization and Extensions.

Produces:
1. Activation colormaps showing the projection of residual stream activations
   onto the English direction at each (layer, token) position for Spanish input.
2. Layer-by-layer accuracy curves from Phase 3 ablation results.
3. Convergence plots (pairwise cosine similarity between direction methods).
4. Classification accuracy plots (held-out accuracy per method per layer).
5. Cross-language consistency plots (direction similarity across language pairs).

Usage:
    python phase4_visualize.py
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
import utils


# ── Style setup ──────────────────────────────────────────────────────────────

def setup_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


# ── Activation colormaps ─────────────────────────────────────────────────────

def plot_english_activation_colormap(
    model, directions: dict[int, torch.Tensor], prompt: str,
    title: str, save_path: str,
):
    """
    Plot the "English activation" at each (layer, token) position.

    X-axis: token positions (labeled with subword tokens)
    Y-axis: layer index
    Color: projection magnitude onto English direction (diverging colormap)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=True)
    n_layers = len(directions)
    seq_len = tokens.shape[1]

    # Extract activations at all layers
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "resid_post" in name,
        )

    # Compute projections: [n_layers, seq_len]
    projections = torch.zeros(n_layers, seq_len)
    for L in range(n_layers):
        resid = cache["resid_post", L][0].cpu()  # [seq_len, d_model]
        d = directions[L]
        projections[L] = resid @ d  # [seq_len]

    del cache

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), max(6, n_layers * 0.25)))

    # Center colormap at zero
    vmax = projections.abs().max().item()
    sns.heatmap(
        projections.numpy(),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=str_tokens,
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cbar_kws={"label": "Projection onto English direction"},
    )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved colormap: {save_path}")


def generate_colormaps(model, directions: dict[int, torch.Tensor]):
    """Generate activation colormaps for selected Spanish examples."""
    # Load some Spanish reasoning examples
    task_types = ["relational", "factual_inference", "pattern_completion"]

    for task_type in task_types:
        data_path = config.DATA_DIR / f"{task_type}.json"
        if not data_path.exists():
            print(f"  Skipping {task_type}: dataset not found")
            continue

        with open(data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        # Plot first 3 examples per task type
        for i, ex in enumerate(examples[:3]):
            plot_english_activation_colormap(
                model=model,
                directions=directions,
                prompt=ex["prompt_es"],
                title=f"English Activation — {task_type} (Example {i})",
                save_path=str(
                    config.PLOTS_DIR / f"colormap_{task_type}_{i}.png"
                ),
            )


# ── Accuracy curves ──────────────────────────────────────────────────────────

def plot_accuracy_curves():
    """
    Plot layer-by-layer accuracy under each ablation condition.
    One subplot per task type.
    """
    results_path = config.ABLATION_DIR / "ablation_results.json"
    if not results_path.exists():
        print("  Skipping accuracy curves: ablation results not found")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    task_types = sorted(set(r["task_type"] for r in results))
    conditions = [
        ("en_ablated_es", "English dir ablated (ES)", "red", "-"),
        ("en_ablated_en", "English dir ablated (EN)", "blue", "-"),
        ("random_ablated_es", "Random dir ablated (ES)", "gray", "--"),
        ("baseline_es", "Baseline (ES)", "green", ":"),
        ("baseline_en", "Baseline (EN)", "darkgreen", ":"),
        ("high_var_ablated_es", "High-var dir ablated (ES)", "orange", "--"),
    ]

    fig, axes = plt.subplots(1, len(task_types), figsize=(6 * len(task_types), 5))
    if len(task_types) == 1:
        axes = [axes]

    for ax, task_type in zip(axes, task_types):
        task_results = [r for r in results if r["task_type"] == task_type]
        layers = sorted(set(r["layer"] for r in task_results))

        for cond_name, label, color, linestyle in conditions:
            accs = []
            for layer in layers:
                r = [
                    x for x in task_results
                    if x["layer"] == layer and x["condition"] == cond_name
                ]
                if r:
                    accs.append(r[0]["accuracy"])
                else:
                    accs.append(None)

            # Filter out None values
            valid = [(l, a) for l, a in zip(layers, accs) if a is not None]
            if valid:
                ls, acs = zip(*valid)
                ax.plot(ls, acs, label=label, color=color, linestyle=linestyle, marker="o", markersize=3)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Ablation Results — {task_type}")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.PLOTS_DIR / "accuracy_curves.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved accuracy curves: {save_path}")


# ── Convergence plots ────────────────────────────────────────────────────────

def plot_convergence():
    """
    Plot pairwise cosine similarity between direction methods at each layer.
    Handles both old format ({layer: float}) and new format ({layer: {pair: float}}).
    """
    lang_labels = {"es": "ES", "fr": "FR", "zh": "ZH"}
    pair_colors = {
        "mean_diff_vs_logreg": "tab:blue",
        "mean_diff_vs_pca": "tab:orange",
        "logreg_vs_pca": "tab:green",
    }

    # Detect format from first available file
    available_langs = []
    for lang in lang_labels:
        conv_path = config.DIRECTIONS_DIR / f"convergence_{lang}.json"
        if conv_path.exists():
            available_langs.append(lang)

    if not available_langs:
        print("  Skipping convergence plot: no data found")
        return

    with open(config.DIRECTIONS_DIR / f"convergence_{available_langs[0]}.json") as f:
        sample = json.load(f)
    sample_val = list(sample.values())[0]
    is_new_format = isinstance(sample_val, dict)

    if is_new_format:
        # New format: one subplot per language, 3 method-pair lines each
        n_langs = len(available_langs)
        fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 5))
        if n_langs == 1:
            axes = [axes]

        for ax, lang in zip(axes, available_langs):
            conv_path = config.DIRECTIONS_DIR / f"convergence_{lang}.json"
            with open(conv_path, "r") as f:
                data = json.load(f)

            layers = sorted(int(k) for k in data.keys())

            for pair_name, color in pair_colors.items():
                sims = [data[str(l)].get(pair_name, 0.0) for l in layers]
                ax.plot(
                    layers, sims, label=pair_name.replace("_vs_", " vs "),
                    color=color, marker="o", markersize=3,
                )

            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Method Convergence — EN vs {lang_labels[lang]}")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
    else:
        # Old format: single plot, one line per language
        fig, ax = plt.subplots(figsize=(8, 5))
        old_colors = {"es": "red", "fr": "blue", "zh": "green"}
        for lang in available_langs:
            conv_path = config.DIRECTIONS_DIR / f"convergence_{lang}.json"
            with open(conv_path, "r") as f:
                data = json.load(f)
            layers = sorted(int(k) for k in data.keys())
            sims = [data[str(l)] for l in layers]
            ax.plot(
                layers, sims, label=f"EN vs {lang_labels[lang]}",
                color=old_colors.get(lang, "gray"), marker="o", markersize=4,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity (mean_diff vs logreg)")
        ax.set_title("Direction Extraction Convergence")
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.PLOTS_DIR / "convergence.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved convergence plot: {save_path}")


# ── Mean projection profile ─────────────────────────────────────────────────

def plot_mean_projection_profile(model, directions: dict[int, torch.Tensor]):
    """
    Plot the mean English-direction projection across layers for
    sample Spanish inputs, showing where the model "converts" to
    English-like representations.
    """
    data_path = config.DATA_DIR / "relational.json"
    if not data_path.exists():
        print("  Skipping projection profile: relational dataset not found")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    es_texts = [ex["prompt_es"] for ex in examples[:20]]
    activations = utils.get_residual_activations(model, es_texts)
    n_layers = len(directions)
    max_seq = activations[0].shape[1]
    mask = utils.get_content_mask(model, es_texts, max_seq)

    mean_projections = []
    for L in range(n_layers):
        avg_acts = utils.average_over_positions(activations[L], mask)
        proj = (avg_acts @ directions[L]).mean().item()
        mean_projections.append(proj)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(n_layers), mean_projections, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean projection onto English direction")
    ax.set_title("English Activation Profile Across Layers (Spanish input)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = config.PLOTS_DIR / "mean_projection_profile.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved projection profile: {save_path}")


# ── Classification accuracy plots ────────────────────────────────────────────

def plot_classification_accuracy():
    """
    Plot held-out classification accuracy per method per layer.
    One subplot per language pair.
    """
    lang_labels = {"es": "ES", "fr": "FR", "zh": "ZH"}
    method_styles = {
        "mean_diff": ("-", "tab:blue"),
        "logreg": ("--", "tab:orange"),
        "pca": (":", "tab:green"),
    }

    available_langs = []
    for lang in lang_labels:
        path = config.DIRECTIONS_DIR / f"classification_accuracy_{lang}.json"
        if path.exists():
            available_langs.append(lang)

    if not available_langs:
        print("  Skipping classification accuracy plot: no data found")
        return

    n_langs = len(available_langs)
    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 5))
    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, available_langs):
        acc_path = config.DIRECTIONS_DIR / f"classification_accuracy_{lang}.json"
        with open(acc_path, "r") as f:
            acc_data = json.load(f)

        layers = sorted(int(k) for k in acc_data.keys())

        for method, (linestyle, color) in method_styles.items():
            accs = [acc_data[str(l)].get(method, 0.0) for l in layers]
            ax.plot(
                layers, accs, label=method,
                linestyle=linestyle, color=color, marker="o", markersize=3,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Classification Accuracy")
        ax.set_title(f"Held-out Accuracy — EN vs {lang_labels[lang]}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = config.PLOTS_DIR / "classification_accuracy.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved classification accuracy plot: {save_path}")


# ── Cross-language consistency plots ─────────────────────────────────────────

def plot_cross_language_consistency():
    """
    Plot cross-language direction similarity per method.
    One subplot per method, with lines for each language pair.
    """
    cross_path = config.DIRECTIONS_DIR / "cross_language_consistency.json"
    if not cross_path.exists():
        print("  Skipping cross-language plot: data not found")
        return

    with open(cross_path, "r") as f:
        data = json.load(f)

    methods = list(data.keys())
    pair_colors = {
        "es_vs_fr": "tab:purple",
        "es_vs_zh": "tab:orange",
        "fr_vs_zh": "tab:cyan",
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        layers = sorted(int(k) for k in data[method].keys())

        for pair_name, color in pair_colors.items():
            sims = [data[method][str(l)].get(pair_name, 0.0) for l in layers]
            ax.plot(
                layers, sims,
                label=pair_name.replace("_vs_", " vs ").upper(),
                color=color, marker="o", markersize=3,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Cross-Language Consistency — {method}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.PLOTS_DIR / "cross_language_consistency.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-language consistency plot: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config.ensure_dirs()
    setup_style()

    # Load directions
    directions = {}
    layer = 0
    method = config.DIRECTION_METHOD
    while True:
        path = config.DIRECTIONS_DIR / f"english_direction_es_L{layer}_{method}.pt"
        if not path.exists():
            break
        directions[layer] = utils.load_tensor(path)
        layer += 1

    if not directions:
        print("No direction vectors found. Run Phase 1 first.")
        return

    print(f"Loaded {len(directions)} direction vectors")

    # Load model for colormaps and projection profile
    model = utils.load_model()

    print("\n--- Generating activation colormaps ---")
    generate_colormaps(model, directions)

    print("\n--- Plotting accuracy curves ---")
    plot_accuracy_curves()

    print("\n--- Plotting convergence ---")
    plot_convergence()

    print("\n--- Plotting classification accuracy ---")
    plot_classification_accuracy()

    print("\n--- Plotting cross-language consistency ---")
    plot_cross_language_consistency()

    print("\n--- Plotting mean projection profile ---")
    plot_mean_projection_profile(model, directions)

    print(f"\nPhase 4 complete. Plots saved to: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
