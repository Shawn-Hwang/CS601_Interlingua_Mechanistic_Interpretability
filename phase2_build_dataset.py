"""
Phase 2: Reasoning Dataset Construction.

Generates parallel English/Spanish reasoning tasks with three types:
1. Relational reasoning (logical deductions with proper nouns)
2. One-step factual inference (deductions from stated premises)
3. Pattern completion (short sequence patterns)

Pre-filters to keep only examples where the model answers correctly
in both languages, ensuring a clean baseline.

Usage:
    python phase2_build_dataset.py
"""

import json
import random
from pathlib import Path

from tqdm import tqdm

import config
import utils


# ── Templates ────────────────────────────────────────────────────────────────
# Each template produces (prompt_en, prompt_es, answer_en, answer_es)

# Names used across templates (culturally neutral proper nouns)
NAMES = [
    "Ana", "Carlos", "Maria", "Juan", "Pedro", "Sofia", "Luis",
    "Elena", "Diego", "Laura", "Miguel", "Isabel", "Pablo", "Carmen",
    "Roberto", "Lucia", "Andres", "Rosa", "Fernando", "Marta",
]

COLORS_EN = ["black", "white", "brown", "gray", "orange"]
COLORS_ES = ["negro", "blanco", "marron", "gris", "naranja"]

ANIMALS_EN = ["cat", "dog", "bird", "rabbit", "fish"]
ANIMALS_ES = ["gato", "perro", "pajaro", "conejo", "pez"]


def generate_relational_reasoning() -> list[dict]:
    """
    Generate relational reasoning examples.
    Logical deductions with proper nouns and comparison relations.
    """
    examples = []
    comparisons = [
        ("taller", "tall", "mas alto/alta", "alto/alta"),
        ("older", "old", "mayor", "mayor"),
        ("faster", "fast", "mas rapido/rapida", "rapido/rapida"),
    ]

    # Type 1: Three-way transitive comparison (who is the tallest/shortest?)
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        names = random.sample(NAMES, 3)
        a, b, c = names
        comp_en, adj_en, comp_es, adj_es = random.choice(comparisons)

        # a > b > c, ask for smallest
        prompt_en = (
            f"If {a} is {comp_en} than {b}, and {b} is {comp_en} than {c}, "
            f"who is the least {adj_en}? Answer with just the name."
        )
        prompt_es = (
            f"Si {a} es {comp_es.split('/')[0]} que {b}, y {b} es "
            f"{comp_es.split('/')[0]} que {c}, "
            f"quien es el/la menos {adj_es.split('/')[0]}? "
            f"Responde solo con el nombre."
        )
        answer = c

        examples.append({
            "prompt_en": prompt_en,
            "prompt_es": prompt_es,
            "answer_en": answer,
            "answer_es": answer,  # Names are the same in both languages
            "task_type": "relational",
        })

    # Type 2: Two-way comparison (who is taller?)
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        names = random.sample(NAMES, 2)
        a, b = names
        comp_en, adj_en, comp_es, adj_es = random.choice(comparisons)

        prompt_en = (
            f"{a} is {comp_en} than {b}. Who is {comp_en}? "
            f"Answer with just the name."
        )
        prompt_es = (
            f"{a} es {comp_es.split('/')[0]} que {b}. "
            f"Quien es {comp_es.split('/')[0]}? "
            f"Responde solo con el nombre."
        )
        answer = a

        examples.append({
            "prompt_en": prompt_en,
            "prompt_es": prompt_es,
            "answer_en": answer,
            "answer_es": answer,
            "task_type": "relational",
        })

    random.shuffle(examples)
    return examples


def generate_factual_inference() -> list[dict]:
    """
    Generate one-step factual inference examples.
    Simple deductions from stated premises.
    """
    examples = []

    # Type 1: All X of Y are Z. Y has an X named W. What color is W?
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        name = random.choice(NAMES)
        idx = random.randint(0, len(COLORS_EN) - 1)
        color_en, color_es = COLORS_EN[idx], COLORS_ES[idx]
        a_idx = random.randint(0, len(ANIMALS_EN) - 1)
        animal_en, animal_es = ANIMALS_EN[a_idx], ANIMALS_ES[a_idx]
        pet_name = random.choice([n for n in NAMES if n != name])

        prompt_en = (
            f"All of {name}'s {animal_en}s are {color_en}. "
            f"{name} has a {animal_en} named {pet_name}. "
            f"What color is {pet_name}? Answer with just the color."
        )
        prompt_es = (
            f"Todos los {animal_es}s de {name} son {color_es}. "
            f"{name} tiene un {animal_es} llamado {pet_name}. "
            f"De que color es {pet_name}? Responde solo con el color."
        )
        examples.append({
            "prompt_en": prompt_en,
            "prompt_es": prompt_es,
            "answer_en": color_en,
            "answer_es": color_es,
            "task_type": "factual_inference",
        })

    # Type 2: If X then Y. X is true. Is Y true?
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        name = random.choice(NAMES)
        scenarios = [
            (
                f"If it rains, {name} takes an umbrella. It is raining.",
                f"Does {name} take an umbrella? Answer yes or no.",
                f"Si llueve, {name} lleva un paraguas. Esta lloviendo.",
                f"Lleva {name} un paraguas? Responde si o no.",
                "yes", "si",
            ),
            (
                f"Everyone in {name}'s family speaks Spanish. {name} is in the family.",
                f"Does {name} speak Spanish? Answer yes or no.",
                f"Todos en la familia de {name} hablan espanol. {name} esta en la familia.",
                f"Habla {name} espanol? Responde si o no.",
                "yes", "si",
            ),
            (
                f"All students in the class passed the exam. {name} is a student in the class.",
                f"Did {name} pass the exam? Answer yes or no.",
                f"Todos los estudiantes de la clase aprobaron el examen. {name} es estudiante de la clase.",
                f"Aprobo {name} el examen? Responde si o no.",
                "yes", "si",
            ),
        ]
        premise_en, question_en, premise_es, question_es, ans_en, ans_es = random.choice(scenarios)

        examples.append({
            "prompt_en": f"{premise_en} {question_en}",
            "prompt_es": f"{premise_es} {question_es}",
            "answer_en": ans_en,
            "answer_es": ans_es,
            "task_type": "factual_inference",
        })

    random.shuffle(examples)
    return examples


def generate_pattern_completion() -> list[dict]:
    """
    Generate pattern completion examples.
    Short sequence patterns requiring one logical step.
    """
    examples = []

    # Arithmetic sequences
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        start = random.randint(1, 20)
        step = random.choice([2, 3, 4, 5, 10])
        seq = [start + step * i for i in range(5)]
        answer = str(seq[-1] + step)
        seq_str = ", ".join(str(x) for x in seq)

        prompt_en = (
            f"What comes next in the sequence: {seq_str}, ? "
            f"Answer with just the number."
        )
        prompt_es = (
            f"Que sigue en la secuencia: {seq_str}, ? "
            f"Responde solo con el numero."
        )
        examples.append({
            "prompt_en": prompt_en,
            "prompt_es": prompt_es,
            "answer_en": answer,
            "answer_es": answer,  # Numbers are language-independent
            "task_type": "pattern_completion",
        })

    # Geometric sequences
    for _ in range(config.N_EXAMPLES_PER_TASK * config.CANDIDATE_MULTIPLIER):
        start = random.choice([1, 2, 3])
        ratio = random.choice([2, 3])
        seq = [start * (ratio ** i) for i in range(5)]
        if seq[-1] > 1000:
            continue
        answer = str(seq[-1] * ratio)
        seq_str = ", ".join(str(x) for x in seq)

        prompt_en = (
            f"What comes next in the sequence: {seq_str}, ? "
            f"Answer with just the number."
        )
        prompt_es = (
            f"Que sigue en la secuencia: {seq_str}, ? "
            f"Responde solo con el numero."
        )
        examples.append({
            "prompt_en": prompt_en,
            "prompt_es": prompt_es,
            "answer_en": answer,
            "answer_es": answer,
            "task_type": "pattern_completion",
        })

    random.shuffle(examples)
    return examples


# ── Generators registry ──────────────────────────────────────────────────────

TASK_GENERATORS = {
    "relational": generate_relational_reasoning,
    "factual_inference": generate_factual_inference,
    "pattern_completion": generate_pattern_completion,
}


# ── Pre-filtering ────────────────────────────────────────────────────────────

def filter_examples(
    model, examples: list[dict], target_count: int
) -> list[dict]:
    """
    Keep only examples where model answers correctly in both EN and ES.
    Stop once target_count passing examples are collected.
    """
    passed = []
    total_tested = 0

    for ex in tqdm(examples, desc="Filtering"):
        if len(passed) >= target_count:
            break
        total_tested += 1

        # Test English
        correct_en, _ = utils.evaluate_completion(
            model, ex["prompt_en"], ex["answer_en"]
        )
        if not correct_en:
            continue

        # Test Spanish
        correct_es, _ = utils.evaluate_completion(
            model, ex["prompt_es"], ex["answer_es"]
        )
        if not correct_es:
            continue

        ex["id"] = len(passed)
        passed.append(ex)

    pass_rate = len(passed) / total_tested if total_tested > 0 else 0
    print(
        f"  Filtered: {len(passed)}/{total_tested} passed "
        f"(pass rate: {pass_rate:.1%})"
    )

    if pass_rate < config.BASELINE_ACCURACY_THRESHOLD:
        print(
            f"  WARNING: Pass rate {pass_rate:.1%} is below threshold "
            f"{config.BASELINE_ACCURACY_THRESHOLD:.0%}. "
            f"Consider redesigning templates for this task type."
        )

    return passed


def save_dataset(examples: list[dict], task_type: str) -> None:
    """Save filtered dataset to JSON."""
    path = config.DATA_DIR / f"{task_type}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(examples)} examples to {path}")


def main():
    config.ensure_dirs()
    random.seed(42)
    model = utils.load_model()

    for task_type, generator in TASK_GENERATORS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_type}")
        print(f"{'='*60}")

        # Generate candidates
        candidates = generator()
        print(f"  Generated {len(candidates)} candidates")

        # Filter through model
        filtered = filter_examples(
            model, candidates, config.N_EXAMPLES_PER_TASK
        )

        # Save
        save_dataset(filtered, task_type)

    print(f"\nPhase 2 complete. Datasets saved to: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
