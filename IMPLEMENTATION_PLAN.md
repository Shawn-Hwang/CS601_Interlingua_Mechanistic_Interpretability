# Plan: Strengthen Phase 1 Direction Extraction (Priority 1 + 2)

## Context

The current Phase 1 extracts English directions using only mean-diff and logistic regression on 200 FLORES samples with position-averaged pooling. This plan adds PCA-based extraction (the RepE/LAT standard), cross-language consistency analysis, held-out classification validation, mean-centering for anisotropy correction, more samples, and configurable pooling — all without breaking Phase 3/4 compatibility.

---

## Files to Modify (in order)

### 1. `config.py` — 3 small edits
- Change `N_SAMPLES_DIRECTION` from `200` to `500`
- Add `POOLING_STRATEGY: str = "mean"` after BATCH_SIZE (supports `"mean"` | `"last_token"`)
- Update `DIRECTION_METHOD` comment to include `"pca"` as valid option

### 2. `utils.py` — 3 new functions, 0 modifications
All additions go in the "Direction computation" section.

**a. `last_token_activation(activations, mask) -> [batch, d_model]`**
- Extracts the activation at the last content (non-special) token position
- Located after `average_over_positions` (~line 139)

**b. `pool_activations(activations, mask, strategy) -> [batch, d_model]`**
- Dispatcher: calls `average_over_positions` or `last_token_activation` based on strategy string
- Located after `last_token_activation`

**c. `compute_direction_pca(en_acts, other_acts) -> [d_model]`**
- Computes per-sample diffs `en[i] - other[i]`, centers them, runs SVD, takes first PC
- Sign-corrects PC so it points toward English (dot product with mean-diff > 0)
- Requires paired inputs (same count of EN and Other samples)
- Located after `compute_direction_logreg` (~line 183)

### 3. `phase1_extract_direction.py` — major restructure

**a. Rewrite `extract_directions_for_pair()`** — new pipeline:
1. Extract activations (unchanged — single call per language)
2. Build content masks (unchanged)
3. Pool via `utils.pool_activations(strategy=config.POOLING_STRATEGY)` instead of direct `average_over_positions`
4. **Mean-center**: compute `global_mean = concat(en_avg, other_avg).mean(dim=0)`, subtract from both
5. **Train/test split**: deterministic 75%/25% on the pooled+centered `[n_samples, d_model]` tensors
6. Compute all 3 methods (mean_diff, logreg, pca) on **train set only**
7. Compute pairwise cosine similarities between all 3 directions (convergence)
8. Compute held-out classification accuracy: project test samples onto each direction, threshold at midpoint between train-set class means, report accuracy per method

Returns dict with keys: `mean_diff`, `logreg`, `pca`, `convergence`, `classification_accuracy`

**b. Rewrite `save_directions()`** — saves:
- `.pt` files for all 3 methods (follows existing `english_direction_{lang}_L{layer}_{method}.pt` pattern)
- `convergence_{lang}.json` — new format: `{layer: {pair_name: cosine_sim}}` (was flat `{layer: float}`)
- `classification_accuracy_{lang}.json` — `{layer: {method: accuracy}}`

**c. Add cross-language consistency block** at end of `main()`:
- After all 3 language pairs are processed, load all saved directions
- Compute pairwise cosine similarity between EN-ES, EN-FR, EN-ZH directions at each layer, for each method
- Save `cross_language_consistency.json`: `{method: {layer: {lang_pair: cosine_sim}}}`
- Print summary to console

**d. Update console output** to show 3-method convergence and accuracy

### 4. `phase4_visualize.py` — 1 modification, 2 additions

**a. Modify `plot_convergence()`**:
- Detect old format (flat float) vs new format (dict of pairs) via `isinstance` check
- If new format: 3 subplots (one per lang), each showing 3 method-pair lines
- If old format: backward-compatible single-line plot

**b. Add `plot_classification_accuracy()`**:
- 3 subplots (one per lang), each with 3 method lines showing accuracy across layers
- Reads from `classification_accuracy_{lang}.json`

**c. Add `plot_cross_language_consistency()`**:
- 3 subplots (one per method), each with 3 lang-pair lines
- Reads from `cross_language_consistency.json`

**d. Update `main()`** to call the 2 new plotting functions

### 5. `phase3_ablation.py` — NO changes needed
The `load_directions()` function already reads by `config.DIRECTION_METHOD` + filename pattern. Setting `DIRECTION_METHOD = "pca"` will load PCA directions. The saved vectors are unit-normalized `[d_model]` with English-pointing sign convention, identical to what `make_ablation_hook` expects.

---

## New Output Files

| File | Description |
|------|-------------|
| `english_direction_{lang}_L{layer}_pca.pt` | PCA direction per layer (78 new files: 26 layers x 3 langs) |
| `convergence_{lang}.json` | **Format change**: now `{layer: {pair: cosine_sim}}` |
| `classification_accuracy_{lang}.json` | Held-out accuracy per layer per method |
| `cross_language_consistency.json` | Cross-lang direction similarity per method |
| `classification_accuracy.png` | Plot |
| `cross_language_consistency.png` | Plot |

---

## Key Design Decisions

1. **Mean-centering in phase1, not in utils** — keeps utility functions backward-compatible; centering is applied in `extract_directions_for_pair` before calling any `compute_direction_*` function
2. **PCA sign convention** — PC1 is oriented so `dot(pc1, mean_diff) > 0`, ensuring the direction points toward English for Phase 3 ablation compatibility
3. **Train/test split after pooling** — avoids running the model twice; `get_residual_activations` is called once, then pooled tensors are sliced
4. **Classification threshold** — midpoint of train-set per-class mean projections (simplest unbiased threshold for the linear classifier)
5. **Memory with 500 samples** — ~10 GB RAM for 26 layers of `[500, ~128, 1536]` on CPU. The code already uses `.cpu()` caching. If OOM, reduce BATCH_SIZE.

---

## Verification

1. Run `python phase1_extract_direction.py` — should complete without errors, producing all `.pt`, `.json` files
2. Check `classification_accuracy_es.json` — accuracy should be >0.8 at mid layers for all 3 methods
3. Check `cross_language_consistency.json` — ES/FR/ZH directions should show high cosine sim at mid layers
4. Run `python phase4_visualize.py` — should produce all plots including new ones
5. Set `DIRECTION_METHOD = "pca"` in config and verify Phase 3 loads correctly: `python -c "from phase3_ablation import load_directions; d = load_directions('es'); print(len(d))"`
