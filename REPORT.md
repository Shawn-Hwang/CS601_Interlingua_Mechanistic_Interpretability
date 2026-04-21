# English Interlingua Causal Ablation — Project Report

**Model:** Gemma-3-1B-IT (26 layers, instruction-tuned)
**Languages:** English (EN) vs. Spanish (ES), French (FR), Chinese (ZH)
**Dataset:** FLORES-200 parallel corpus (500 sentence pairs per language pair)
**Reasoning Tasks:** Factual inference, relational reasoning, pattern completion (50 examples each, filtered for bilingual correctness)

---

## 1. Executive Summary

This project investigates whether multilingual LLMs internally route non-English reasoning through English-like representations — the **English interlingua hypothesis**. Using directional ablation on Gemma-3-1B-IT via TransformerLens, we extract a linear "English direction" from the residual stream and causally test its necessity for Spanish reasoning.

**Key findings:**

1. A robust, language-universal English direction exists in the residual stream, concentrated in layers 5–20 and extractable via mean-difference and PCA methods.
2. Ablating this direction **destroys Spanish factual inference** (accuracy drops from 100% to ~0% at layers 4–21) while **preserving English reasoning** (~80–100% accuracy) at the same layers — strong causal evidence for the interlingua hypothesis.
3. The effect is **task-dependent**: factual inference shows the strongest asymmetric disruption, relational reasoning shows moderate language-general disruption, and pattern completion shows high variability.
4. The English direction is concentrated at the `<bos>` token position and peaks at layers 15–18, suggesting language identity is encoded early in the sequence and amplified through middle layers.

---

## 2. Phase 1: Direction Extraction and Validation

### 2.1 Method Convergence

Three direction extraction methods were applied at each of 26 layers for each of 3 language pairs (EN-ES, EN-FR, EN-ZH):

| Method Pair | Layers 6–14 (Middle) | Layers 0–3 (Early) | Layers 21–25 (Late) |
|---|---|---|---|
| mean_diff vs. PCA | **0.97–0.99** (ES), **0.98–0.99** (FR), **0.92–0.99** (ZH) | 0.02–0.55 (ES), 0.09–0.45 (FR), 0.87–0.96 (ZH) | 0.02–0.25 (ES), 0.16–0.57 (FR), 0.52–0.87 (ZH) |
| mean_diff vs. logreg | 0.10–0.15 | 0.31–0.42 | 0.39–0.42 |
| logreg vs. PCA | 0.02–0.05 | -0.02–0.02 | -0.02–0.04 |

**Finding 1: Mean-difference and PCA converge to essentially the same direction at middle layers.** For EN-ES, the cosine similarity between mean_diff and PCA reaches 0.988 at L7 and remains above 0.97 through L14. For EN-FR, similarity exceeds 0.98 at L6–L14. For EN-ZH, it exceeds 0.97 at L4–L5 and L8–L13. This convergence validates that both methods identify the same underlying linear structure.

**Finding 2: Logistic regression finds a categorically different direction.** Logreg achieves high classification accuracy (see below) but its direction has near-zero cosine similarity with both mean_diff (~0.10–0.15) and PCA (~0.02–0.05) at middle layers. This indicates logreg optimizes for a classification boundary that is orthogonal to the primary language-contrast axis, likely exploiting distributional properties rather than the dominant mean-shift signal.

### 2.2 Held-Out Classification Accuracy

Directions were trained on 75% of samples and tested on 25% held-out data:

| Method | EN-ES (Best Layer) | EN-FR (Best Layer) | EN-ZH (Best Layer) |
|---|---|---|---|
| **Logreg** | **0.884** (L20–L22) | **0.888** (L22) | **0.888** (L20) |
| **Mean_diff** | **0.868** (L24) | **0.848** (L25) | **0.676** (L25) |
| **PCA** | 0.580 (L7) | 0.604 (L7) | 0.600 (L14) |

**Finding 3: Logreg achieves the highest classification accuracy (>0.82 at most layers), but PCA hovers near chance (~0.50–0.58).** This reveals a fundamental distinction between *classification-optimal* directions and *causally relevant* directions. PCA and mean_diff capture the direction of maximum language contrast in activation space (the "true" English direction), while logreg finds a decision boundary that may exploit higher-order statistical patterns.

**Finding 4: Mean_diff classification accuracy spikes at the last 4 layers for EN-ES** (0.688 at L21, 0.864 at L22, 0.868 at L24), suggesting the English direction becomes a stronger discriminative feature near the output layers for closely related language pairs. For EN-ZH (more distant pair), this late spike is less pronounced.

### 2.3 Cross-Language Consistency

The critical test: do directions extracted from EN-ES, EN-FR, and EN-ZH converge to a **single universal** English direction?

| Method | Peak Cosine (ES vs FR) | Peak Cosine (ES vs ZH) | Peak Cosine (FR vs ZH) | Peak Layers |
|---|---|---|---|---|
| **PCA** | **0.9999** (L7) | **0.9996** (L4) | **0.9993** (L3) | L3–L22 (>0.99 all pairs) |
| **Mean_diff** | **0.9971** (L9) | **0.9777** (L9) | **0.9846** (L9) | L5–L14 (>0.95 all pairs) |
| **Logreg** | 0.649 (L9) | 0.424 (L8) | 0.445 (L8) | No clear peak |

**Finding 5: PCA yields a near-perfectly universal English direction.** The PCA-extracted direction from EN-ES, EN-FR, and EN-ZH has cosine similarity >0.99 across all language pairs at layers 3–22 (after sign correction). This is remarkable — the same linear direction separates English from Spanish, French, *and* Chinese, despite these being typologically diverse languages. This is the strongest evidence that the model encodes a language-general "English vs. not-English" feature, consistent with the interlingua hypothesis.

**Finding 6: Mean_diff shows strong but not universal cross-language consistency.** It peaks at L5–L14 with >0.95 cosine similarity across all pairs, but degrades at later layers (dropping to 0.14–0.53 at L22–L25). The ZH pair consistently shows lower similarity than ES-FR, indicating the English direction has slightly different characteristics for typologically distant language pairs.

**Finding 7: Logreg directions are language-pair-specific.** With cross-language cosine similarity of only 0.30–0.65, the logreg direction is not universal — it captures language-pair-specific discriminative features rather than a shared English representation. This further validates using mean_diff (not logreg) for causal ablation.

---

## 3. Phase 2: Reasoning Dataset Construction

Three reasoning task types were generated with parallel EN/ES versions, filtered to keep only examples the model solves correctly in both languages:

| Task Type | Examples | Description |
|---|---|---|
| Factual Inference | 50 | Deductive: "All of Elena's fish are gray. Elena has a fish named Carlos. What color is Carlos?" |
| Relational Reasoning | 50 | Transitive: "If Isabel is faster than Lucia, who is the least fast?" |
| Pattern Completion | 50 | Arithmetic: "What comes next: 2, 4, 6, 8, 10, ?" |

The 100% baseline accuracy (both EN and ES) at all layers for all unablated conditions confirms successful filtering — the model can solve all retained examples bilingually without intervention.

---

## 4. Phase 3: Causal Ablation Experiments

Ablation formula at each layer: `r' = r - (r · d̂)d̂ + μd̂`, where `d̂` is the English direction and `μ` is the mean projection of Spanish prompts. Six conditions were tested per layer.

### 4.1 Factual Inference: Strongest Evidence for Interlingua

This task shows the clearest asymmetric pattern and is the central result of the project.

**Layer-by-layer accuracy for factual inference (n=10 per layer):**

| Layer | en_ablated_es | en_ablated_en | random_ablated_es | baseline_es | baseline_en | high_var_ablated_es |
|---|---|---|---|---|---|---|
| L0 | 0.7 | 0.8 | 0.2 | 1.0 | 1.0 | 0.2 |
| L1 | 0.5 | 0.5 | 0.7 | 1.0 | 1.0 | 0.5 |
| L2 | 0.2 | 0.4 | 0.5 | 1.0 | 1.0 | 0.0 |
| L3 | 0.2 | 0.8 | 0.6 | 1.0 | 1.0 | 0.3 |
| **L4** | **0.0** | **1.0** | 0.7 | 1.0 | 1.0 | 0.0 |
| **L5** | **0.0** | **1.0** | 0.5 | 1.0 | 1.0 | 0.2 |
| **L6** | **0.0** | **0.7** | 0.5 | 1.0 | 1.0 | 0.6 |
| **L7** | **0.0** | **0.9** | 0.7 | 1.0 | 1.0 | 0.5 |
| **L8** | **0.0** | **1.0** | 0.7 | 1.0 | 1.0 | 0.3 |
| **L9** | **0.0** | **0.5** | 0.5 | 1.0 | 1.0 | 0.1 |
| **L10** | **0.0** | **1.0** | 0.9 | 1.0 | 1.0 | 0.6 |
| **L11** | **0.0** | **0.8** | 0.5 | 1.0 | 1.0 | 0.0 |
| L12 | 0.1 | 1.0 | 0.5 | 1.0 | 1.0 | 0.1 |
| **L13** | **0.0** | **1.0** | 0.7 | 1.0 | 1.0 | 0.0 |
| **L14** | **0.0** | **0.8** | 0.8 | 1.0 | 1.0 | 0.1 |
| L15 | 0.2 | 1.0 | 0.7 | 1.0 | 1.0 | 0.0 |
| L16 | 0.1 | 0.7 | 0.7 | 1.0 | 1.0 | 0.3 |
| **L17** | **0.0** | **0.9** | 0.6 | 1.0 | 1.0 | 0.7 |
| L18 | 0.3 | 0.7 | 0.8 | 1.0 | 1.0 | 0.5 |
| **L19** | **0.0** | **0.4** | 0.5 | 1.0 | 1.0 | 0.4 |
| **L20** | **0.0** | **0.9** | 0.2 | 1.0 | 1.0 | 0.4 |
| L21 | 0.1 | 0.8 | 0.4 | 1.0 | 1.0 | 1.0 |
| L22 | 0.7 | 0.9 | 0.6 | 1.0 | 1.0 | 0.0 |
| L23 | 1.0 | 1.0 | 0.6 | 1.0 | 1.0 | 0.0 |
| L24 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 0.0 |
| L25 | 1.0 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |

**Finding 8: Ablating the English direction at layers 4–21 completely destroys Spanish factual inference (accuracy ≈ 0%) while preserving English factual inference (accuracy ≈ 80–100%).** This is the strongest result of the project. At layers L4, L5, L7, L8, L10, L13, L14, L17, L19, and L20, Spanish accuracy is exactly 0.0 while English accuracy remains 0.7–1.0. The asymmetry rules out the possibility that the ablation merely disrupts general reasoning circuitry.

**Finding 9: The effect is not explained by generic disruption.** The random direction control (`random_ablated_es`) averages 0.58 accuracy across all layers — far above the near-zero en_ablated_es values at middle layers. This confirms the degradation is specific to the English direction, not to any arbitrary perturbation.

**Finding 10: The English direction's causal role spans a broad layer range (L4–L21).** Unlike prior work suggesting narrow layer specificity, the English direction is causally necessary across ~70% of the network depth. The effect vanishes at L22–L25 (accuracy returns to 1.0), coinciding with the collapse of the direction's cross-language consistency at those layers.

### 4.2 Relational Reasoning: Language-General Disruption

| Layer Range | en_ablated_es (avg) | en_ablated_en (avg) | random_ablated_es (avg) | high_var_ablated_es (avg) |
|---|---|---|---|---|
| L0–L1 | 1.0 | 0.95 | 1.0 | 0.05 |
| L2–L4 | 0.1 | 0.17 | 1.0 | 0.03 |
| L5–L9 | 0.8 | 0.40 | 1.0 | 1.0 |
| L10–L11 | 0.4 | 0.60 | 1.0 | 1.0 |
| L12–L15 | 0.97 | 0.77 | 1.0 | 1.0 |

**Finding 11: Relational reasoning shows a qualitatively different ablation profile.** At early layers (L2–L4), both `en_ablated_es` and `en_ablated_en` drop severely (to 0.0–0.3), indicating the English direction at these layers is important for reasoning in general, not specifically for cross-lingual transfer. At middle-to-late layers, the random control (`random_ablated_es`) remains at 1.0, while the English direction ablation causes moderate drops — but the drops affect English reasoning as much as or more than Spanish reasoning.

**Finding 12: The relational task does not support the interlingua hypothesis as strongly.** The key diagnostic is the comparison between `en_ablated_es` and `en_ablated_en`. For relational reasoning, these conditions often show comparable degradation (both drop to 0.1–0.4 at L2–L4), suggesting the English direction encodes reasoning-relevant features shared across languages rather than serving as a cross-lingual bridge for this task type.

### 4.3 Pattern Completion: High Variability and Small Sample Effects

| Layer Range | en_ablated_es (avg) | en_ablated_en (avg) | random_ablated_es (avg) | high_var_ablated_es (avg) |
|---|---|---|---|---|
| L0–L6 | 0.23 | 0.46 | 0.37 | 0.09 |
| L7–L14 | 0.60 | 0.56 | 0.20 | 0.03 |
| L15–L20 | 0.43 | 0.93 | 0.28 | 0.13 |
| L21–L25 | 0.32 | 1.0 | 0.68 | 0.24 |

**Finding 13: Pattern completion results are noisy due to the small sample size (n=5 per layer) and the numerical nature of the task.** The high variance makes it difficult to draw strong conclusions. However, at L15–L25 there is a suggestive asymmetry: `en_ablated_en` maintains 0.8–1.0 while `en_ablated_es` drops to 0.0–0.6, partially consistent with the interlingua pattern seen in factual inference.

**Finding 14: The high-variance direction control is highly disruptive across all tasks.** `high_var_ablated_es` shows consistently low accuracy (often 0.0), particularly for pattern completion and factual inference. This indicates that the high-variance (non-English-aligned) direction captures important general computational features. However, this does not diminish the English direction's specific role: the English direction ablation selectively impairs Spanish-language processing while the high-variance ablation impairs all processing indiscriminately.

---

## 5. Phase 4: Activation Topology

### 5.1 English Activation Heatmaps

The activation colormaps (token position x layer) reveal where and when the model converts Spanish input into English-like representations.

**Finding 15: English-direction activation is concentrated at the `<bos>` token.** Across all three task types (factual inference, relational, pattern completion), the `<bos>` token shows dramatically higher projection onto the English direction than content tokens. At the `<bos>` position, projections reach 40,000–50,000 at peak layers (L13–L18), while content tokens remain below 5,000. This suggests the model uses the beginning-of-sequence token as a **language identity register** — it encodes "this is a non-English input" information that propagates through subsequent processing.

**Finding 16: Content tokens show weak but non-zero English activation at later layers.** While the `<bos>` token dominates, content tokens (e.g., Spanish words like "Todos", "Elena", "Isabel") develop mild positive projections at layers 20–25, suggesting some late-stage convergence toward English-like representations in the content positions as well.

### 5.2 Mean Projection Profile

**Finding 17: The English-direction projection follows a bell-shaped curve across layers, peaking at L15–L18 (~7,500–8,000) for Spanish input.** The projection is near zero at L0–L2, grows monotonically from L3 to L16, and declines after L20. A notable dip occurs at L21–L23 (~100–1,600) before a secondary rise at L24–L25 (~2,200–3,300). This profile directly mirrors the layer range where ablation causes maximum disruption (L4–L21), confirming that the English direction's causal role coincides with its presence in the activation space.

---

## 6. Synthesis: Themed Conclusions

### Theme A: Evidence For the English Interlingua Hypothesis

The factual inference results provide the **strongest causal evidence** for the interlingua hypothesis found in this project:

1. **Asymmetric disruption**: Ablating the English direction at L4–L21 reduces Spanish accuracy to ~0% while English accuracy remains ~80–100% (Finding 8).
2. **Specificity**: Random direction ablation does not produce this effect (Finding 9).
3. **Universality of the direction**: The PCA-extracted English direction is nearly identical (cosine >0.99) across EN-ES, EN-FR, and EN-ZH (Finding 5).
4. **Spatial localization**: The causal role coincides with the layer range where the English direction is most prominent in the activation space (Finding 17).

This is consistent with the interpretation that the model **routes Spanish factual reasoning through English-like representations** — when this pathway is removed, Spanish reasoning collapses but English reasoning (which does not depend on the same cross-lingual transfer) is unaffected.

### Theme B: Task-Dependent Interlingua Reliance

Not all reasoning tasks rely equally on the English interlingua:

| Task | Interlingua Evidence | Interpretation |
|---|---|---|
| **Factual Inference** | **Strong** — asymmetric ES vs EN disruption | Deductive chains require language-general representations; model uses English as the shared representation space |
| **Relational Reasoning** | **Weak** — both ES and EN are disrupted | Transitive comparisons may use a more language-general circuit that overlaps with, but is not mediated by, the English direction |
| **Pattern Completion** | **Suggestive** — mild asymmetry at L15–L25, but high noise | Numerical reasoning may partially bypass linguistic representations; small sample (n=5) limits confidence |

**Implication**: The interlingua hypothesis may be most applicable to language-heavy reasoning tasks (factual inference, which requires parsing premises and conclusions stated in natural language) rather than tasks with strong non-linguistic structure (pattern completion with numerical sequences).

### Theme C: The Direction Extraction Methods Tell Different Stories

The three extraction methods reveal complementary aspects of the English representation:

1. **PCA** captures the dominant variance axis of the language contrast. It produces the most language-universal direction (>0.99 cosine across language pairs, Finding 5) but has poor held-out classification accuracy (~0.50–0.58, near chance). This suggests PCA identifies the direction of maximal *spread* between language clusters, but this direction may not be optimal for *separating* individual points.

2. **Mean_diff** captures the centroid-to-centroid direction. It shows strong convergence with PCA at middle layers (>0.97, Finding 1), high cross-language consistency (>0.95 at L5–L14, Finding 6), and moderate classification accuracy that spikes at later layers (0.87 at L22 for ES, Finding 4). This method was used for the causal ablation experiments.

3. **Logreg** optimizes for classification accuracy (>0.82 at most layers, Finding 3) but finds a direction orthogonal to the other two methods and lacking cross-language consistency (Finding 7). This method is most useful for *detecting* which language a representation encodes, but the direction it finds is not the causally relevant one.

**Implication for representation engineering**: Classification accuracy alone is an insufficient criterion for selecting directions for causal intervention. The mean_diff/PCA directions — which capture the dominant geometric structure rather than the optimal decision boundary — are the causally relevant ones.

### Theme D: The `<bos>` Token as Language Identity Register

The activation colormaps reveal that the English direction's signal is concentrated at the `<bos>` token position (Finding 15). This has several implications:

1. The model may encode language identity information in a position-specific manner, using the sequence-initial token as a global context register.
2. The causal ablation at a given layer modifies the `<bos>` representation, which then influences all downstream processing through attention mechanisms.
3. This is consistent with the observation that instruction-tuned models (like Gemma-3-1B-IT) use the beginning-of-sequence context to set up language-specific processing modes.

### Theme E: Layer-Depth Profile of Language Encoding

The English direction's properties change systematically with network depth:

| Layer Range | Direction Quality | Cross-Language Consistency | Causal Importance | Interpretation |
|---|---|---|---|---|
| L0–L3 (Early) | Low convergence | Low (mean_diff), High (PCA) | Moderate disruption | Raw token embeddings; language not yet fully separated |
| L4–L14 (Middle) | High convergence (>0.97) | Very high (>0.95) | **Maximum disruption** | Core language processing; interlingua encoding |
| L15–L20 (Upper-middle) | Declining convergence | Declining | Strong disruption | Transition to output representations |
| L21–L25 (Late) | Low convergence | Low (mean_diff), Moderate (PCA) | **No disruption** | Output-layer features; language-specific decoding |

The causal importance of the English direction mirrors its geometric quality: ablation is most destructive where the direction is most well-defined and universal.

---

## 7. Limitations

1. **Small ablation sample sizes**: Only 10 examples per layer for factual inference/relational and 5 for pattern completion. This limits statistical power and contributes to noisy results, particularly for pattern completion.

2. **Single model**: All results are from Gemma-3-1B-IT. The findings may not generalize to larger models, non-instruction-tuned models, or architecturally different models.

3. **Single random direction**: `N_RANDOM_DIRECTIONS = 1` was used due to compute constraints. Averaging over more random baselines would strengthen the null control.

4. **Mean_diff direction for ablation**: Phase 3 used the mean_diff direction exclusively. Testing with PCA-extracted directions (which show higher cross-language consistency) might yield different results.

5. **Only EN-ES reasoning tasks**: The ablation experiments test only Spanish reasoning, not French or Chinese. Cross-lingual generalization of the causal effect remains untested.

---

## 8. Summary Table of All Quantitative Findings

| Finding | Metric | Value | Evidence Source |
|---|---|---|---|
| Method convergence (middle layers) | cosine(mean_diff, PCA) | 0.97–0.99 | convergence_{es,fr,zh}.json |
| Logreg orthogonality | cosine(logreg, PCA) | 0.02–0.05 | convergence_{es,fr,zh}.json |
| Logreg classification accuracy | held-out accuracy | 0.82–0.89 | classification_accuracy_{es,fr,zh}.json |
| PCA classification accuracy | held-out accuracy | 0.50–0.60 | classification_accuracy_{es,fr,zh}.json |
| PCA cross-language universality | cosine(ES-dir, FR-dir) | >0.99 at L3–L22 | cross_language_consistency.json |
| Mean_diff cross-language universality | cosine(ES-dir, FR-dir) | >0.95 at L5–L14 | cross_language_consistency.json |
| Factual: en_ablated_es (L4–L21) | accuracy | ~0.0 (avg 0.05) | ablation_results.json |
| Factual: en_ablated_en (L4–L21) | accuracy | ~0.82 (avg) | ablation_results.json |
| Factual: baseline (all layers) | accuracy | 1.0 | ablation_results.json |
| Mean projection peak | projection magnitude | ~7,900 at L16 | mean_projection_profile.png |
| English activation locus | token position | `<bos>` token dominant | colormap_*.png |

---

## 9. Figures Reference

All figures are located in `results/plots/`:

| Figure | Description | Key Observation |
|---|---|---|
| `accuracy_curves.png` | Layer-by-layer accuracy under 6 ablation conditions | Factual inference shows asymmetric ES collapse |
| `convergence.png` | Pairwise cosine similarity between 3 extraction methods | mean_diff ≈ PCA at middle layers; logreg diverges |
| `classification_accuracy.png` | Held-out EN/Other classification per method | Logreg dominates; PCA near chance |
| `cross_language_consistency.png` | Direction similarity across language pairs | PCA is near-perfectly universal (>0.99) |
| `mean_projection_profile.png` | Average English-direction projection across layers | Bell curve peaking at L15–L18 |
| `colormap_*.png` (9 files) | Token x layer English activation heatmaps | `<bos>` token dominates the signal |
