# English Interlingua Causal Ablation — Project Guide

This project tests whether the "English direction" in multilingual LLMs is causally necessary for non-English reasoning, using directional ablation via TransformerLens.

---

## Setup

### 1. Environment

Python 3.10+ is required. Create a virtual environment and install dependencies:

```bash
cd final_project
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU strongly recommended.** The project runs on CPU but Phase 3 will take many hours without a GPU. If you have CUDA, `torch` will auto-detect it. For Apple Silicon, MPS is auto-detected.

### 2. HuggingFace Access

Phase 1 downloads the FLORES-200 dataset from HuggingFace. No authentication is needed for `facebook/flores`. If you hit rate limits:

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Model Selection

Open `config.py` and set `MODEL_NAME` to any TransformerLens-supported model:

```python
MODEL_NAME = "gemma-3-1b-it"          # default, ~3GB VRAM
MODEL_NAME = "gemma-2-2b"             # alternative, ~5GB VRAM
MODEL_NAME = "Qwen/Qwen1.5-1.8B"     # alternative, lighter
MODEL_NAME = "gpt2-medium"            # for quick testing (~1.5GB)
```

Everything else (layer count, embedding dimension, tokenizer) adapts automatically at runtime. The only requirement is TransformerLens support — check the [model table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html).

### 4. Quick Smoke Test

To verify the setup works before running full experiments, temporarily reduce sample sizes in `config.py`:

```python
N_SAMPLES_DIRECTION = 10   # was 500
N_EXAMPLES_PER_TASK = 5    # was 50
N_RANDOM_DIRECTIONS = 2    # was 10
```

Then run Phase 1. If it completes without errors, restore the original values.

---

## Phase-by-Phase Guide

### Phase 1: Finding the English Direction

**What it does:** Extracts a linear "English direction" from the residual stream at each layer using parallel EN/ES/FR/ZH sentence pairs from FLORES-200. Computes via three methods with mean-centering, train/test validation, and cross-language consistency analysis.

**Run:**
```bash
python phase1_extract_direction.py
```

**What happens step by step:**
1. Loads the model (first run downloads weights from HuggingFace)
2. Loads 500 FLORES-200 parallel sentences for each language pair (EN-ES, EN-FR, EN-ZH)
3. For each pair:
   - Runs all English sentences through the model, caches residual stream activations at every layer
   - Does the same for the other-language sentences
   - Pools activations over token positions using the configured strategy (`"mean"` averages over content-word positions; `"last_token"` takes the final content token only)
   - **Mean-centers** all activations (subtracts the global mean across both EN and Other) to correct for anisotropy
   - Splits into **75% train / 25% test** (deterministic, first 375 train, last 125 test)
   - Computes the English direction at each layer via three methods on the **train set**:
     - **Mean difference**: `d = mean(EN) - mean(Other)`, normalized
     - **Logistic regression**: weight vector of an EN-vs-Other classifier, normalized
     - **PCA on paired differences**: for each sample pair `i`, computes `diff[i] = EN[i] - Other[i]`, then takes the first principal component (RepE/LAT standard method). Sign-corrected so the direction points toward English.
   - Computes **pairwise cosine similarity** between all three methods at each layer (convergence check)
   - Computes **held-out classification accuracy** on the test set: projects test samples onto each direction, classifies by midpoint threshold, reports accuracy per method
4. After all three language pairs are processed, runs a **cross-language consistency analysis**: computes pairwise cosine similarity between directions extracted from EN-ES, EN-FR, EN-ZH at each layer, for each method

**Outputs (in `results/directions/`):**
- `english_direction_{lang}_L{layer}_{method}.pt` — direction vector per layer per method (`mean_diff`, `logreg`, `pca`)
- `convergence_{lang}.json` — pairwise cosine similarities between all three methods at each layer
- `classification_accuracy_{lang}.json` — held-out classification accuracy per method per layer
- `cross_language_consistency.json` — cross-language direction similarity per method (measures whether EN-ES, EN-FR, EN-ZH directions converge to a single "English direction")

**What to check:**
- Cosine similarity between methods should be high (>0.5) at middle layers — indicates a well-defined direction
- Classification accuracy should be high (>0.8) at middle layers — validates that the direction separates languages
- Cross-language consistency: if EN-ES, EN-FR, EN-ZH directions are similar (high cosine), the direction captures "English vs. non-English" rather than "English vs. one specific language"
- If all metrics are low everywhere, the direction may not be well-defined for this model

**Expected runtime:** 45–90 min on GPU (longer than before due to 500 samples and 3 methods), longer on CPU.

---

### Phase 2: Reasoning Dataset Construction

**What it does:** Generates parallel EN/ES reasoning tasks and filters to keep only examples the model solves correctly in both languages.

**Run:**
```bash
python phase2_build_dataset.py
```

**What happens step by step:**
1. For each of 3 task types, generates candidate examples from templates:
   - **Relational reasoning**: "If Ana is taller than Maria, and Maria is taller than Lucia, who is the least tall?" — uses randomized proper nouns and comparison relations
   - **Factual inference**: "All of Carlos's cats are black. Carlos has a cat named Luna. What color is Luna?" — deductions from stated premises
   - **Pattern completion**: "What comes next: 2, 4, 6, 8, 10, ?" — arithmetic/geometric sequences
2. Generates ~3x the target (150 candidates per task) to account for filtering losses
3. For each candidate, tests if the model answers correctly in both EN and ES
4. Keeps the first 50 that pass in both languages
5. Logs the pass rate — if below 75%, warns that templates may need redesign

**Outputs (in `data/reasoning/`):**
- `relational.json` — 50 filtered examples
- `factual_inference.json` — 50 filtered examples
- `pattern_completion.json` — 50 filtered examples

Each JSON entry contains: `prompt_en`, `prompt_es`, `answer_en`, `answer_es`, `task_type`, `id`.

**What to check:**
- Pass rate per task type (printed to console). If very low (<30%), the task may be too hard for the model
- Manually inspect a few examples in the JSON to verify template quality
- If you need different templates, edit the generator functions in `phase2_build_dataset.py`

**Expected runtime:** 60–120 min on GPU (generation-heavy).

---

### Phase 3: Causal Ablation Experiments

**What it does:** Ablates the English direction at each layer independently and measures reasoning accuracy under 4+ conditions.

**Prerequisites:** Phase 1 and Phase 2 must be complete.

**Run:**
```bash
python phase3_ablation.py
```

**Direction method selection:** Phase 3 loads directions using `DIRECTION_METHOD` from `config.py`. Set this to `"mean_diff"`, `"logreg"`, or `"pca"` to select which extraction method's directions to use for ablation. All three sets are produced by Phase 1.

**What happens step by step:**
1. Loads saved English directions from Phase 1 and reasoning datasets from Phase 2
2. For each task type, computes the mean projection μ (the average activation projected onto the English direction) over Spanish prompts — this is the constant that replaces the ablated component
3. For each layer independently, runs 6 conditions:
   - **(a) `en_ablated_es`**: Ablate English direction, test Spanish reasoning — **the key condition**
   - **(b) `en_ablated_en`**: Ablate English direction, test English reasoning — within-language control
   - **(c) `random_ablated_es`**: Ablate a random direction (averaged over 10), test Spanish — null control
   - **(d) `baseline_es`**: No ablation, Spanish — baseline
   - **(e) `baseline_en`**: No ablation, English — baseline
   - **(f) `high_var_ablated_es`**: Ablate a high-variance but non-English-aligned direction — specificity control
4. The ablation formula at each layer: `r' = r - (r · d̂)d̂ + μd̂`

**The ablation interpretation:**
- If condition (a) shows **much worse** accuracy than (b), (c), and (d): the English direction is causally necessary for Spanish reasoning → supports the interlingua hypothesis
- If (a) shows **similar** degradation to (b): the direction is important for reasoning in general, not specifically for cross-lingual transfer
- If (a) and (c) are **similar**: the degradation is from generic disruption, not English-specific

**Outputs (in `results/ablation/`):**
- `ablation_results.json` — one record per (task_type, layer, condition) with accuracy

**What to check:**
- Look for layers where `en_ablated_es` accuracy drops significantly below `baseline_es`
- Compare the drop against `en_ablated_en` and `random_ablated_es`
- The effect should peak at middle layers (based on prior work by Wendler et al.)

**Expected runtime:** 3–6 hours on GPU. This is the computational bottleneck.

---

### Phase 4: Visualization

**What it does:** Generates all figures for the paper/presentation.

**Prerequisites:** Phase 1 required. Phase 2 required for colormaps. Phase 3 required for accuracy curves.

**Run:**
```bash
python phase4_visualize.py
```

**Outputs (in `results/plots/`):**

1. **`colormap_{task}_{i}.png`** — English activation heatmaps
   - X-axis: token positions (subword tokens from Spanish input)
   - Y-axis: layers (0 to N)
   - Color: projection onto English direction (red = English-like, blue = non-English-like)
   - Shows where/when the model converts Spanish input into English-like representations

2. **`accuracy_curves.png`** — Layer-by-layer accuracy under each ablation condition
   - One subplot per task type
   - All 6 conditions overlaid
   - Key plot for the paper — shows the causal profile

3. **`convergence.png`** — Pairwise cosine similarity between the three direction extraction methods across layers
   - One subplot per language pair (ES, FR, ZH)
   - Three lines per subplot: mean_diff vs logreg, mean_diff vs pca, logreg vs pca
   - Validates that the extraction methods agree (high similarity = robust direction)

4. **`classification_accuracy.png`** — Held-out classification accuracy per method per layer
   - One subplot per language pair
   - Three lines per subplot: mean_diff, logreg, pca
   - Validates that the extracted direction genuinely separates English from non-English activations

5. **`cross_language_consistency.png`** — Cross-language direction similarity per method
   - One subplot per method (mean_diff, logreg, pca)
   - Three lines per subplot: ES vs FR, ES vs ZH, FR vs ZH
   - High similarity indicates a universal "English direction" rather than language-pair-specific artifacts

6. **`mean_projection_profile.png`** — Average English-direction projection at each layer for Spanish input
   - Bar chart showing the "English activation" profile
   - Should peak in middle layers

**Expected runtime:** 10–20 min on GPU.

---

## Project Structure Reference

```
final_project/
├── config.py                       # All configuration — MODEL_NAME goes here
├── requirements.txt                # pip dependencies
├── utils.py                        # Shared utilities (model loading, hooks, generation, pooling)
├── phase0_download_model.py        # Optional: pre-download model weights
├── phase1_extract_direction.py     # Phase 1: direction extraction (3 methods + validation)
├── phase2_build_dataset.py         # Phase 2: dataset construction
├── phase3_ablation.py              # Phase 3: ablation experiments
├── phase4_visualize.py             # Phase 4: visualization (6 plot types)
├── GUIDE.md                        # This file
├── IMPLEMENTATION_PLAN.md          # Design document for direction extraction improvements
├── data/reasoning/                 # Phase 2 outputs (JSON datasets)
│   └── flores/                     # Cached FLORES-200 parallel corpus
└── results/
    ├── directions/                 # Phase 1 outputs (.pt vectors, .json metrics)
    ├── ablation/                   # Phase 3 outputs (ablation_results.json)
    └── plots/                      # Phase 4 outputs (figures .png)
```

**Data flow:**
- Phases 1 and 2 are independent — run in either order
- Phase 3 reads from both Phase 1 (`results/directions/`) and Phase 2 (`data/reasoning/`)
- Phase 4 reads from Phase 1 + Phase 2, and optionally Phase 3

---

## Direction Extraction Methods

Phase 1 extracts the English direction using three complementary methods. Understanding their differences helps interpret results:

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Mean difference** | `d = mean(EN) - mean(Other)`, normalized | Simple, most causally implicated (Marks & Tegmark 2024) | Sensitive to outliers |
| **Logistic regression** | Weight vector of a trained EN-vs-Other linear classifier | Highest detection accuracy | May find directions good for classification but less causally relevant |
| **PCA on paired differences** | First PC of `EN[i] - Other[i]` per-sample differences (RepE/LAT standard) | Captures dominant variance axis of language contrast; standard in representation engineering | Sign-ambiguous (corrected automatically); assumes language is the dominant variance source |

All three methods are applied to **mean-centered** activations (global mean of EN + Other subtracted) to correct for the anisotropy of LLM activation spaces. Directions are extracted on a **75% train split** and validated on a **25% held-out test set** via classification accuracy.

The **cross-language consistency** metric measures whether directions extracted from EN-ES, EN-FR, and EN-ZH converge. High cross-language cosine similarity indicates a universal "English direction" — strong evidence for the interlingua hypothesis.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OutOfMemoryError` on GPU | Reduce `BATCH_SIZE` in `config.py` (try 4 or 2) |
| `OutOfMemoryError` on CPU (500 samples) | Reduce `N_SAMPLES_DIRECTION` to 300, or reduce `BATCH_SIZE` |
| Model not found in TransformerLens | Check the [model table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html) for exact names |
| FLORES download fails | Run `huggingface-cli login` or set `HF_TOKEN` env var |
| Phase 1 classification accuracy low everywhere | The direction may not be well-defined; try a different model or check data quality |
| Phase 1 cross-language consistency low | Directions may be language-pair-specific rather than universal; consider using more samples |
| Phase 2 pass rate very low | The model may be too weak for the task — try simpler templates or a larger model |
| Phase 3 takes too long | Reduce `N_RANDOM_DIRECTIONS` to 3, or test on fewer layers first |
| Plots look wrong | Check that Phase 1/2/3 outputs exist in the expected directories |

## Key Config Parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `MODEL_NAME` | `"gemma-3-1b-it"` | Which model to use |
| `N_SAMPLES_DIRECTION` | `500` | Sentence pairs for direction extraction (FLORES dev has 997) |
| `BATCH_SIZE` | `8` | Batch size for activation extraction (reduce if OOM) |
| `POOLING_STRATEGY` | `"mean"` | How to aggregate token-level activations: `"mean"` (average over content tokens) or `"last_token"` (final content token only) |
| `N_EXAMPLES_PER_TASK` | `50` | Target examples per reasoning task |
| `MAX_NEW_TOKENS` | `32` | Max tokens generated per evaluation |
| `N_RANDOM_DIRECTIONS` | `10` | Random control directions in Phase 3 |
| `DIRECTION_METHOD` | `"mean_diff"` | Primary direction method for Phase 3 ablation: `"mean_diff"`, `"logreg"`, or `"pca"` |
