"""
Central configuration for the English Interlingua Causal Ablation project.

Change MODEL_NAME to swap models. Everything else (layer count, d_model)
is derived at runtime from the loaded model's config.
"""

import torch
from pathlib import Path

# ── Model ────────────────────────────────────────────────────────────────────
# Change this to swap models (must be supported by TransformerLens).
# Examples: "gemma-2-2b", "Qwen/Qwen1.5-1.8B", "meta-llama/Llama-2-7b-hf"
MODEL_NAME: str = "gemma-3-1b-it"

DEVICE: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── FLORES parallel corpus ───────────────────────────────────────────────────
FLORES_DATASET: str = "facebook/flores"
FLORES_CONFIGS: dict[str, str] = {
    "es": "eng_Latn-spa_Latn",
    "fr": "eng_Latn-fra_Latn",
    "zh": "eng_Latn-zho_Hans",
}
FLORES_SPLIT: str = "dev"  # 997 sentences; devtest reserved for validation

# ── Phase 1: Direction extraction ────────────────────────────────────────────
N_SAMPLES_DIRECTION: int = 10       # number of sentence pairs to use
BATCH_SIZE: int = 8                  # batch size for activation extraction

# ── Phase 2: Reasoning dataset ───────────────────────────────────────────────
N_EXAMPLES_PER_TASK: int = 5
CANDIDATE_MULTIPLIER: int = 3        # generate 3x candidates to hit target after filtering
BASELINE_ACCURACY_THRESHOLD: float = 0.75
MAX_NEW_TOKENS: int = 32             # max tokens to generate for evaluation

# ── Phase 3: Ablation ───────────────────────────────────────────────────────
N_RANDOM_DIRECTIONS: int = 2        # number of random control directions to average
DIRECTION_METHOD: str = "mean_diff"  # primary method: "mean_diff" or "logreg"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_DIR / "data" / "reasoning"
RESULTS_DIR: Path = PROJECT_DIR / "results"
DIRECTIONS_DIR: Path = RESULTS_DIR / "directions"
ABLATION_DIR: Path = RESULTS_DIR / "ablation"
PLOTS_DIR: Path = RESULTS_DIR / "plots"


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, DIRECTIONS_DIR, ABLATION_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
