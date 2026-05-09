"""Data loading and preprocessing for PAN@CLEF 2026 Reasoning Trajectory Detection.

Supports configurable data directories and both subtask formats.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Default data directory; override via load functions' data_dir parameter
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to a .jsonl file.

    Returns:
        List of parsed JSON objects.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.debug("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Subtask 1: Source Detection
# ---------------------------------------------------------------------------

def load_subtask1(split: str = "train",
                  data_dir: Optional[Path] = None) -> list[dict]:
    """Load subtask 1 data for a given split.

    Each record has: problem_id, solution_id, problem, solution,
    generator, detailed_generator. A binary label is added:
    0 = human, 1 = llm.

    Args:
        split: One of 'train', 'validation', 'test'.
        data_dir: Override default data directory.

    Returns:
        List of record dicts with 'label' field added.
    """
    base = (data_dir or DEFAULT_DATA_DIR) / "subtask1"

    if split == "test":
        folder = base / "test"
        all_records = []
        for f in sorted(folder.glob("*.jsonl")):
            all_records.extend(load_jsonl(f))
        return all_records

    if split == "train":
        folder = base / "train"
        prefix = "train_"
    else:
        folder = base / "validation"
        prefix = "valid_"

    all_records = []
    for f in sorted(folder.glob(f"{prefix}*.jsonl")):
        all_records.extend(load_jsonl(f))

    for r in all_records:
        r["label"] = 0 if r["generator"] == "human" else 1

    log.info("Loaded %d subtask1 %s records", len(all_records), split)
    return all_records


def load_subtask1_by_generator(split: str = "train",
                               data_dir: Optional[Path] = None) -> dict[str, list[dict]]:
    """Load subtask 1 data grouped by generator.

    Args:
        split: One of 'train', 'validation', 'test'.
        data_dir: Override default data directory.

    Returns:
        Dict mapping generator name to list of records.
    """
    records = load_subtask1(split, data_dir)
    by_gen: dict[str, list[dict]] = {}
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


def load_subtask2(split: str = "train",
                  data_dir: Optional[Path] = None) -> list[dict]:
    """Load subtask 2 data for a given split.

    Each record has: id, query, generator, reasoning_trace, label,
    detailed_label. Label is mapped to int: safe=0,
    potentially_unsafe=1, unsafe=2.

    Args:
        split: One of 'train', 'validation', 'test'.
        data_dir: Override default data directory.

    Returns:
        List of record dicts with 'label_int' field added.
    """
    base = (data_dir or DEFAULT_DATA_DIR) / "subtask2"

    if split == "test":
        folder = base / "test"
        all_records = []
        for f in sorted(folder.glob("*.jsonl")):
            all_records.extend(load_jsonl(f))
        return all_records

    if split == "train":
        folder = base / "train"
        prefix = "train_"
    else:
        folder = base / "validation"
        prefix = "valid_"

    all_records = []
    for f in sorted(folder.glob(f"{prefix}*.jsonl")):
        all_records.extend(load_jsonl(f))

    for r in all_records:
        raw_label = r["label"].strip().lower()
        r["label_int"] = SAFETY_LABEL_MAP.get(raw_label, 1)

    log.info("Loaded %d subtask2 %s records", len(all_records), split)
    return all_records


def parse_reasoning_steps(trace: str) -> list[str]:
    """Parse a reasoning trace into individual steps.

    Steps are delimited by 'Step N:' markers.

    Args:
        trace: Raw reasoning trace text.

    Returns:
        List of step text strings.
    """
    parts = re.split(r"Step \d+:\s*", trace)
    steps = [p.strip() for p in parts if p.strip()]
    return steps
