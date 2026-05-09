"""Reasoning Trajectory Detection (RTD) for PAN@CLEF 2026.

A system for detecting AI-generated reasoning trajectories and classifying
their safety properties across 22 languages.
"""

__version__ = "1.0.0"

from rtd.features import (
    SUBTASK1_FEATURE_NAMES,
    SUBTASK2_FEATURE_NAMES,
    STEP_FEATURE_NAMES,
    extract_subtask1_features,
    extract_subtask2_features,
    extract_step_features,
    extract_all_step_features,
)
from rtd.data_loader import (
    load_jsonl,
    load_subtask1,
    load_subtask2,
    parse_reasoning_steps,
)
from rtd.evaluate import (
    evaluate_binary,
    evaluate_multiclass,
    evaluate_step_level,
    evaluate_subtask2_full,
)
from rtd.refusal_detector import (
    has_refusal,
    classify_trace,
    classify_sentence,
)
