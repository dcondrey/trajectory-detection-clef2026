"""Subtask 2: Safety Detection training.

Combines heuristic threshold tuning with optional LLM ensemble
orchestration. The heuristic classifier uses three rules:
  1. Multilingual refusal detection (22 languages) -> safe
  2. Content-word Jaccard similarity -> unsafe if high overlap
  3. Trace length threshold -> unsafe if long without refusal

Usage:
    uv run python train_s2.py --data-dir data/
    uv run python train_s2.py --tune-thresholds
    uv run python train_s2.py --run-llm-ensemble --models gemini,groq_llama70b
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from rtd.data_loader import load_subtask2
from rtd.refusal_detector import (
    classify_trace,
    content_words,
    has_refusal,
    jaccard_similarity,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def evaluate_heuristic(records: list[dict], labels_true: list[str],
                       jaccard_thresh: float = 0.080,
                       length_thresh: int = 3500) -> dict:
    """Evaluate the heuristic classifier on a labeled dataset.

    Args:
        records: List of record dicts with 'query' and 'reasoning_trace'.
        labels_true: Ground truth labels ('safe' or 'unsafe').
        jaccard_thresh: Jaccard similarity threshold.
        length_thresh: Trace length threshold.

    Returns:
        Dict with F1 scores and confusion matrix counts.
    """
    from collections import Counter

    preds = []
    for rec in records:
        trace = rec["reasoning_trace"].strip()

        if len(trace) < 5:
            preds.append("safe")
            continue

        if has_refusal(trace):
            preds.append("safe")
            continue

        query_words = content_words(rec["query"])
        trace_words = content_words(trace)
        jacc = jaccard_similarity(query_words, trace_words)

        if jacc > jaccard_thresh:
            preds.append("unsafe")
        elif len(trace) > length_thresh:
            preds.append("unsafe")
        else:
            preds.append("safe")

    tp_safe = sum(1 for t, p in zip(labels_true, preds) if t == "safe" and p == "safe")
    fp_safe = sum(1 for t, p in zip(labels_true, preds) if t != "safe" and p == "safe")
    fn_safe = sum(1 for t, p in zip(labels_true, preds) if t == "safe" and p != "safe")

    tp_unsafe = sum(1 for t, p in zip(labels_true, preds) if t == "unsafe" and p == "unsafe")
    fp_unsafe = sum(1 for t, p in zip(labels_true, preds) if t != "unsafe" and p == "unsafe")
    fn_unsafe = sum(1 for t, p in zip(labels_true, preds) if t == "unsafe" and p != "unsafe")

    prec_safe = tp_safe / (tp_safe + fp_safe) if (tp_safe + fp_safe) > 0 else 0
    rec_safe = tp_safe / (tp_safe + fn_safe) if (tp_safe + fn_safe) > 0 else 0
    f1_safe = 2 * prec_safe * rec_safe / (prec_safe + rec_safe) if (prec_safe + rec_safe) > 0 else 0

    prec_unsafe = tp_unsafe / (tp_unsafe + fp_unsafe) if (tp_unsafe + fp_unsafe) > 0 else 0
    rec_unsafe = tp_unsafe / (tp_unsafe + fn_unsafe) if (tp_unsafe + fn_unsafe) > 0 else 0
    f1_unsafe = 2 * prec_unsafe * rec_unsafe / (prec_unsafe + rec_unsafe) if (prec_unsafe + rec_unsafe) > 0 else 0

    macro_f1 = (f1_safe + f1_unsafe) / 2

    return {
        "macro_f1": macro_f1,
        "f1_safe": f1_safe,
        "f1_unsafe": f1_unsafe,
        "pred_counts": dict(Counter(preds)),
        "true_counts": dict(Counter(labels_true)),
    }


def tune_thresholds(records: list[dict], labels_true: list[str]) -> dict:
    """Grid search over Jaccard and length thresholds.

    Args:
        records: Validation records.
        labels_true: Ground truth labels.

    Returns:
        Dict with best thresholds and score.
    """
    best_f1 = 0
    best_params = {"jaccard": 0.080, "length": 3500}

    for jt in np.arange(0.02, 0.20, 0.005):
        for lt in range(1000, 8000, 500):
            result = evaluate_heuristic(records, labels_true, jt, lt)
            if result["macro_f1"] > best_f1:
                best_f1 = result["macro_f1"]
                best_params = {"jaccard": float(jt), "length": lt}

    log.info("  Best thresholds: Jaccard=%.3f, Length=%d",
             best_params["jaccard"], best_params["length"])
    log.info("  Best macro F1: %.4f", best_f1)

    return {"thresholds": best_params, "macro_f1": best_f1}


def load_validation_binary(data_dir: Path | None = None) -> tuple[list[dict], list[str]]:
    """Load validation data with binary labels (safe vs unsafe).

    Maps potentially_unsafe -> unsafe for binary evaluation.
    """
    records = load_subtask2("validation", data_dir)
    labels = []
    label_map = {
        "safe": "safe",
        "potentially_unsafe": "unsafe",
        "unsafe": "unsafe",
    }
    for r in records:
        raw = r["label"].strip().lower().replace(" ", "_")
        labels.append(label_map.get(raw, "unsafe"))
    return records, labels


def main():
    parser = argparse.ArgumentParser(description="Train Subtask 2 safety classifier")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Output directory")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Run threshold grid search")
    parser.add_argument("--run-llm-ensemble", action="store_true",
                        help="Run LLM ensemble classification")
    parser.add_argument("--models", type=str,
                        default="gemini,groq_llama70b,together_llama70b,mistral_large",
                        help="Comma-separated model names for LLM ensemble")
    parser.add_argument("--split", default="validation",
                        help="Split to evaluate on")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  SUBTASK 2: Safety Detection Training")
    log.info("=" * 60)

    # Load validation data
    val_records, val_labels = load_validation_binary(args.data_dir)
    log.info("Loaded %d validation records", len(val_records))

    # Evaluate default heuristic
    log.info("\n--- Default Heuristic Evaluation ---")
    result = evaluate_heuristic(val_records, val_labels)
    log.info("  Macro F1:    %.4f", result["macro_f1"])
    log.info("  F1 (safe):   %.4f", result["f1_safe"])
    log.info("  F1 (unsafe): %.4f", result["f1_unsafe"])
    log.info("  Pred dist:   %s", result["pred_counts"])
    log.info("  True dist:   %s", result["true_counts"])

    # Threshold tuning
    if args.tune_thresholds:
        log.info("\n--- Threshold Grid Search ---")
        tune_result = tune_thresholds(val_records, val_labels)

        config = {
            "jaccard_threshold": tune_result["thresholds"]["jaccard"],
            "length_threshold": tune_result["thresholds"]["length"],
            "val_macro_f1": tune_result["macro_f1"],
        }
        config_path = args.output_dir / "subtask2_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info("  Config saved to %s", config_path)

    # LLM ensemble
    if args.run_llm_ensemble:
        log.info("\n--- LLM Ensemble Classification ---")
        from llm_ensemble.safety_classification import run_ensemble, write_submission

        model_list = [m.strip() for m in args.models.split(",")]
        results = asyncio.run(run_ensemble(
            args.split, model_list,
            cache_name="ensemble_train",
            data_dir=args.data_dir,
        ))

        output_dir = args.output_dir / "submissions" / "subtask2"
        write_submission(results, output_dir)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
