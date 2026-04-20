"""
Phase 0: Download and verify the model.

Downloads the model from HuggingFace (via TransformerLens), verifies it loads
correctly, and runs a sample generation to confirm the model is functional
before running the main experiment phases.

Usage:
    python phase0_download_model.py
"""

import torch
import config
import utils


SAMPLE_PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "What is machine learning?",
]


def verify_model_config(model) -> None:
    """Print key model config attributes to confirm successful load."""
    cfg = model.cfg
    print("\n--- Model configuration ---")
    print(f"  Model name  : {cfg.model_name}")
    print(f"  n_layers    : {cfg.n_layers}")
    print(f"  d_model     : {cfg.d_model}")
    print(f"  n_heads     : {cfg.n_heads}")
    print(f"  d_vocab     : {cfg.d_vocab}")
    print(f"  Device      : {next(model.parameters()).device}")


def run_sample_generations(model) -> None:
    """Run a few short generations to confirm the model produces output."""
    print("\n--- Sample generations ---")
    for prompt in SAMPLE_PROMPTS:
        output = utils.generate_text(model, prompt, max_new_tokens=10)
        print(f"  Prompt : {prompt!r}")
        print(f"  Output : {output!r}")
        print()


def main():
    print(f"Downloading / loading model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print("(If not already cached, this will download from HuggingFace.)\n")

    model = utils.load_model()

    verify_model_config(model)

    run_sample_generations(model)

    print("Phase 0 complete. Model is ready for use in subsequent phases.")


if __name__ == "__main__":
    main()
