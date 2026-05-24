"""Data loading and preprocessing for PAN-CLEF 2026 Reasoning Trajectory Detection."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Subtask 1: Source Detection
# ---------------------------------------------------------------------------

def load_subtask1(split: str = "train") -> list[dict]:
    """Load all subtask1 data for a given split.

    Each record has: problem_id, solution_id, problem, solution, generator, detailed_generator.
    We add a binary label: 0 = human, 1 = llm.
    """
    if split == "test":
        folder = DATA_DIR / "subtask1" / "test"
        all_records = []
        for f in sorted(folder.glob("*.jsonl")):
            all_records.extend(load_jsonl(f))
        return all_records

    if split == "train":
        folder = DATA_DIR / "subtask1" / "train"
        prefix = "train_"
    else:
        folder = DATA_DIR / "subtask1" / "validation"
        prefix = "valid_"

    all_records = []
    for f in sorted(folder.glob(f"{prefix}*.jsonl")):
        records = load_jsonl(f)
        all_records.extend(records)

    # Add binary label
    for r in all_records:
        r["label"] = 0 if r["generator"] == "human" else 1

    return all_records


def load_subtask1_by_generator(split: str = "train") -> dict[str, list[dict]]:
    """Load subtask1 data grouped by generator."""
    records = load_subtask1(split)
    by_gen = {}
    for r in records:
        gen = r["detailed_generator"]
        by_gen.setdefault(gen, []).append(r)
    return by_gen


# ---------------------------------------------------------------------------
# Subtask 2: Safety Detection
# ---------------------------------------------------------------------------

SAFETY_LABEL_MAP = {
    "safe": 0,
    "potentially": 1,
    "potentially_unsafe": 1,
    "potentially unsafe": 1,
    "unsafe": 2,
}


def load_subtask2(split: str = "train") -> list[dict]:
    """Load all subtask2 data for a given split.

    Each record has: id, query, generator, reasoning_trace, label, detailed_label.
    We map label to int: safe=0, potentially_unsafe=1, unsafe=2.
    """
    if split == "test":
        folder = DATA_DIR / "subtask2" / "test"
        all_records = []
        for f in sorted(folder.glob("*.jsonl")):
            all_records.extend(load_jsonl(f))
        return all_records

    if split == "train":
        folder = DATA_DIR / "subtask2" / "train"
        prefix = "train_"
    else:
        folder = DATA_DIR / "subtask2" / "validation"
        prefix = "valid_"

    all_records = []
    for f in sorted(folder.glob(f"{prefix}*.jsonl")):
        records = load_jsonl(f)
        all_records.extend(records)

    # Map string label to int
    for r in all_records:
        raw_label = r["label"].strip().lower()
        r["label_int"] = SAFETY_LABEL_MAP.get(raw_label, 1)

    return all_records


def parse_reasoning_steps(trace: str) -> list[str]:
    """Parse a reasoning trace into individual steps.

    Steps are delimited by 'Step N:' markers.
    """
    import re
    parts = re.split(r"Step \d+:\s*", trace)
    steps = [p.strip() for p in parts if p.strip()]
    return steps


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_subtask1_stats(split: str = "train"):
    records = load_subtask1(split)
    log.info("\n=== Subtask 1 (%s) ===", split)
    log.info("Total samples: %d", len(records))
    from collections import Counter
    gen_counts = Counter(r["detailed_generator"] for r in records)
    label_counts = Counter(r["generator"] for r in records)
    log.info("Labels: %s", dict(label_counts))
    log.info("Generators: %s", dict(gen_counts))
    sol_lens = [len(r["solution"].split()) for r in records]
    log.info("Solution length (words): min=%d, median=%d, max=%d", min(sol_lens), int(np.median(sol_lens)), max(sol_lens))


def print_subtask2_stats(split: str = "train"):
    records = load_subtask2(split)
    log.info("\n=== Subtask 2 (%s) ===", split)
    log.info("Total samples: %d", len(records))
    from collections import Counter
    label_counts = Counter(r["label"] for r in records)
    log.info("Labels: %s", dict(label_counts))
    gen_counts = Counter(r["generator"] for r in records)
    log.info("Generators: %s", dict(gen_counts))
    step_counts = [len(r["detailed_label"]) for r in records]
    log.info("Steps per trace: min=%d, median=%d, max=%d", min(step_counts), int(np.median(step_counts)), max(step_counts))
    all_step_labels = [l for r in records for l in r["detailed_label"]]
    log.info("Step labels: safe=%d, unsafe=%d", all_step_labels.count(0), all_step_labels.count(1))


if __name__ == "__main__":
    print_subtask1_stats("train")
    print_subtask1_stats("validation")
    print_subtask2_stats("train")
    print_subtask2_stats("validation")
