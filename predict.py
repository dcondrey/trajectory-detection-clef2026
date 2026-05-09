"""Inference entry point for both subtasks.

Generates competition submissions for PAN@CLEF 2026 Reasoning Trajectory
Detection. Supports heuristic, LightGBM, and LLM ensemble approaches.

Usage:
    uv run python predict.py --subtask 1 --input data/subtask1/test/
    uv run python predict.py --subtask 2 --input data/subtask2/test/
    uv run python predict.py --subtask 1 --method lgb --model-dir models/
"""

import argparse
import csv
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def predict_s1_lgb(input_dir: Path, model_dir: Path, output_dir: Path):
    """Predict Subtask 1 using LightGBM ensemble."""
    import lightgbm as lgb

    from rtd.data_loader import load_jsonl
    from rtd.features import extract_subtask1_features

    config_path = model_dir / "subtask1_config.json"
    with open(config_path) as f:
        config = json.load(f)

    threshold = config["threshold"]
    seeds = config["seeds"]

    log.info("Loading models (threshold=%.3f, %d seeds)...", threshold, len(seeds))
    models = []
    for seed in seeds:
        model_path = model_dir / f"subtask1_seed{seed}.txt"
        models.append(lgb.Booster(model_file=str(model_path)))

    records = []
    for f in sorted(input_dir.glob("*.jsonl")):
        records.extend(load_jsonl(f))
    log.info("Loaded %d test records", len(records))

    X = extract_subtask1_features(records)
    probs = np.mean([m.predict(X) for m in models], axis=0)
    preds = (probs > threshold).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        for r, p in zip(records, preds):
            label = "human" if p == 0 else "ai"
            writer.writerow([r.get("id", r.get("solution_id", "")), label])

    zip_path = output_dir / "subtask1_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "submission.csv")

    from collections import Counter
    label_counts = Counter("human" if p == 0 else "ai" for p in preds)
    log.info("Predictions: %s", dict(label_counts))
    log.info("Submission: %s", zip_path)


def predict_s2_heuristic(input_dir: Path, output_dir: Path,
                         config_path: Path | None = None):
    """Predict Subtask 2 using heuristic classifier."""
    from rtd.data_loader import load_jsonl
    from rtd.refusal_detector import classify_trace

    jaccard_thresh = 0.080
    length_thresh = 3500
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        jaccard_thresh = config.get("jaccard_threshold", jaccard_thresh)
        length_thresh = config.get("length_threshold", length_thresh)

    records = []
    for f in sorted(input_dir.glob("*.jsonl")):
        records.extend(load_jsonl(f))
    log.info("Loaded %d test records", len(records))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label", "detailed_label"])
        for rec in records:
            label, detailed = classify_trace(rec["query"], rec["reasoning_trace"])
            writer.writerow([rec["id"], label, detailed])

    zip_path = output_dir / "subtask2_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "submission.csv")

    log.info("Submission: %s", zip_path)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for PAN@CLEF 2026")
    parser.add_argument("--subtask", type=int, required=True, choices=[1, 2],
                        help="Subtask number")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input directory with test JSONL files")
    parser.add_argument("--output", type=Path, default=Path("submissions"),
                        help="Output directory")
    parser.add_argument("--method", choices=["lgb", "heuristic", "ensemble"],
                        default=None,
                        help="Prediction method (default: lgb for S1, heuristic for S2)")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="Model directory (for lgb method)")
    args = parser.parse_args()

    method = args.method
    if method is None:
        method = "lgb" if args.subtask == 1 else "heuristic"

    output_dir = args.output / f"subtask{args.subtask}"

    log.info("=" * 60)
    log.info("  Subtask %d Prediction (%s)", args.subtask, method)
    log.info("=" * 60)

    if args.subtask == 1:
        if method == "lgb":
            predict_s1_lgb(args.input, args.model_dir, output_dir)
        else:
            log.error("Unsupported method '%s' for subtask 1", method)
            raise SystemExit(1)
    else:
        if method == "heuristic":
            config_path = args.model_dir / "subtask2_config.json"
            predict_s2_heuristic(args.input, output_dir, config_path)
        else:
            log.error("Unsupported method '%s' for subtask 2", method)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
