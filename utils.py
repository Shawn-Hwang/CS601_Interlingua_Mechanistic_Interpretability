"""
Shared utilities for the English Interlingua Causal Ablation project.

All TransformerLens interaction logic lives here so phase scripts
stay focused on experiment logic.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from functools import partial
from typing import Callable

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str | None = None) -> HookedTransformer:
    """Load a TransformerLens model in eval mode on the configured device."""
    name = model_name or config.MODEL_NAME
    print(f"Loading model: {name} on {config.DEVICE}")
    try:
        model = HookedTransformer.from_pretrained(name, device=config.DEVICE, local_files_only=True)
        print("Loaded from local cache.")
    except Exception:
        print("Local cache not found, downloading from HuggingFace...")
        model = HookedTransformer.from_pretrained(name, device=config.DEVICE)
    model.eval()
    return model


# ── Activation extraction ────────────────────────────────────────────────────

def get_residual_activations(
    model: HookedTransformer,
    texts: list[str],
    batch_size: int = config.BATCH_SIZE,
) -> dict[int, torch.Tensor]:
    """
    Extract residual stream activations (post each layer) for a list of texts.

    Returns:
        Dict mapping layer_index -> tensor of shape [n_texts, max_seq_len, d_model].
        Sequences are right-padded to the max length within each batch, then
        results are concatenated. Each text's activations retain their original
        sequence length (padded positions are masked by get_content_mask).
    """
    n_layers = model.cfg.n_layers
    all_activations: dict[int, list[torch.Tensor]] = {L: [] for L in range(n_layers)}

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[start : start + batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
            )

        for L in range(n_layers):
            # Shape: [batch, seq_len, d_model]
            acts = cache["resid_post", L].cpu()
            all_activations[L].append(acts)

        del cache
        torch.cuda.empty_cache() if config.DEVICE == "cuda" else None

    # Concatenate batches (may have different seq lengths, so pad)
    result = {}
    for L in range(n_layers):
        result[L] = _pad_and_concat(all_activations[L])
    return result


def _pad_and_concat(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Pad tensors along seq_len dimension and concatenate along batch."""
    max_seq = max(t.shape[1] for t in tensor_list)
    padded = []
    for t in tensor_list:
        if t.shape[1] < max_seq:
            pad = torch.zeros(t.shape[0], max_seq - t.shape[1], t.shape[2])
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def get_content_mask(
    model: HookedTransformer, texts: list[str], max_seq_len: int
) -> torch.Tensor:
    """
    Build a boolean mask of shape [n_texts, max_seq_len] that is True
    for content-word token positions (excludes BOS, EOS, PAD).
    """
    masks = []
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True).squeeze(0)
        seq_len = tokens.shape[0]
        mask = torch.ones(max_seq_len, dtype=torch.bool)
        # Exclude BOS (position 0)
        mask[0] = False
        # Exclude padding beyond actual sequence
        if seq_len < max_seq_len:
            mask[seq_len:] = False
        # Exclude special tokens (EOS, PAD, etc.)
        special_ids = set(model.tokenizer.all_special_ids)
        for i in range(seq_len):
            if tokens[i].item() in special_ids:
                mask[i] = False
        masks.append(mask)
    return torch.stack(masks)


def average_over_positions(
    activations: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean-pool activations over masked positions.

    Args:
        activations: [batch, seq_len, d_model]
        mask: [batch, seq_len] boolean, True = include

    Returns:
        [batch, d_model]
    """
    mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
    masked_sum = (activations * mask_expanded).sum(dim=1)  # [batch, d_model]
    counts = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
    return masked_sum / counts


# ── Direction computation ────────────────────────────────────────────────────

def compute_direction_mean_diff(
    en_acts: torch.Tensor, other_acts: torch.Tensor
) -> torch.Tensor:
    """
    Compute English direction via mean difference.

    Args:
        en_acts: [n_en, d_model] position-averaged English activations
        other_acts: [n_other, d_model] position-averaged other-language activations

    Returns:
        Unit-normalized direction vector [d_model]
    """
    d = en_acts.mean(dim=0) - other_acts.mean(dim=0)
    return d / d.norm()


def compute_direction_logreg(
    en_acts: torch.Tensor, other_acts: torch.Tensor
) -> torch.Tensor:
    """
    Compute English direction via logistic regression weight vector.

    Args:
        en_acts: [n_en, d_model]
        other_acts: [n_other, d_model]

    Returns:
        Unit-normalized direction vector [d_model]
    """
    X = torch.cat([en_acts, other_acts], dim=0).numpy()
    y = np.concatenate([np.ones(len(en_acts)), np.zeros(len(other_acts))])

    clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf.fit(X, y)

    w = torch.tensor(clf.coef_[0], dtype=en_acts.dtype)
    return w / w.norm()


# ── Ablation hooks ───────────────────────────────────────────────────────────

def make_ablation_hook(
    direction: torch.Tensor, mean_projection: float
) -> Callable:
    """
    Create a hook that performs directional mean ablation.

    For each token's residual stream vector r_i:
        r_i' = r_i - (r_i . d_hat) * d_hat + mu * d_hat

    This removes the per-token variation along the direction and
    replaces it with the population mean projection.

    Args:
        direction: Unit-normalized direction vector [d_model]
        mean_projection: Scalar mu, mean projection over reference data
    """
    d = direction.clone()

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # activation shape: [batch, seq_len, d_model]
        d_dev = d.to(activation.device)
        proj = torch.einsum("bsd, d -> bs", activation, d_dev)  # [batch, seq]
        # Remove projection and add back mean
        activation = activation - proj.unsqueeze(-1) * d_dev
        activation = activation + mean_projection * d_dev
        return activation

    return hook_fn


def random_direction(d_model: int) -> torch.Tensor:
    """Sample a random unit vector from the d_model-dimensional unit sphere."""
    v = torch.randn(d_model)
    return v / v.norm()


# ── Generation with hooks ────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_hooks(
    model: HookedTransformer,
    prompt: str,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> str:
    """
    Autoregressive generation with hooks applied at each forward pass.

    TransformerLens's run_with_hooks only does a single forward pass,
    so we manually loop for generation.

    Returns:
        Generated text (excluding the prompt).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)  # [1, seq_len]

    for _ in range(max_new_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        # logits shape: [1, seq_len, vocab_size]
        next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)  # [1]

        # Stop if EOS
        if next_token.item() == model.tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    # Decode only the generated portion
    prompt_len = model.to_tokens(prompt, prepend_bos=True).shape[1]
    generated_tokens = tokens[0, prompt_len:]
    return model.tokenizer.decode(generated_tokens, skip_special_tokens=True)


@torch.no_grad()
def generate_text(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> str:
    """Generate text without hooks (baseline generation)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)

    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == model.tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    prompt_len = model.to_tokens(prompt, prepend_bos=True).shape[1]
    generated_tokens = tokens[0, prompt_len:]
    return model.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def evaluate_completion(
    model: HookedTransformer,
    prompt: str,
    expected_answer: str,
    fwd_hooks: list[tuple[str, Callable]] | None = None,
) -> tuple[bool, str]:
    """
    Generate completion and check if expected answer appears.

    Returns:
        (is_correct, generated_text)
    """
    if fwd_hooks:
        generated = generate_with_hooks(model, prompt, fwd_hooks)
    else:
        generated = generate_text(model, prompt)

    is_correct = expected_answer.lower().strip() in generated.lower()
    return is_correct, generated


# ── Tensor I/O ───────────────────────────────────────────────────────────────

def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor to disk."""
    torch.save(tensor, path)


def load_tensor(path: Path) -> torch.Tensor:
    """Load a tensor from disk."""
    return torch.load(path, map_location="cpu", weights_only=True)
