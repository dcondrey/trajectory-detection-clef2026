"""Generate CodaBench submission CSVs for PAN-CLEF 2026 Reasoning Trajectory Detection.

Subtask 1: submission.csv with columns ID, label (human/ai)
Subtask 2: submission.csv with columns ID, label (safe/unsafe), detailed_label (per-step probs)
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from features import extract_subtask1_features, extract_subtask2_features
from data_loader import parse_reasoning_steps

MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "codabench_submissions"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_subtask1():
    log.info("=== Subtask 1: Source Detection ===")
    test_path = DATA_DIR / "subtask1" / "test" / "subtask1_test.jsonl"
    records = load_jsonl(test_path)
    log.info(f"  Loaded {len(records)} test records")

    model_paths = [
        MODEL_DIR / "subtask1_lgb_combined.txt",
        MODEL_DIR / "subtask1_lgb_structural.txt",
        MODEL_DIR / "subtask1_lgb.txt",
    ]
    model = None
    for mp in model_paths:
        if mp.exists():
            model = lgb.Booster(model_file=str(mp))
            log.info(f"  Loaded model: {mp.name}")
            break

    if model is None:
        log.info("  ERROR: No subtask1 model found")
        return

    X = extract_subtask1_features(records)
    n_expected = model.num_feature()
    if X.shape[1] != n_expected:
        log.info(f"  Feature adjustment: {X.shape[1]} -> {n_expected}")
        X = X[:, :n_expected] if X.shape[1] > n_expected else np.pad(
            X, ((0, 0), (0, n_expected - X.shape[1]))
        )

    probs = model.predict(X)
    preds = (probs > 0.5).astype(int)

    out_dir = OUTPUT_DIR / "subtask1"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "submission.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        for i, record in enumerate(records):
            rid = record["id"]
            label = "ai" if preds[i] == 1 else "human"
            writer.writerow([rid, label])

    n_ai = int(preds.sum())
    n_human = len(preds) - n_ai
    log.info(f"  Predictions: {n_human} human, {n_ai} ai")
    log.info(f"  Written to: {csv_path}")


def generate_subtask2():
    log.info("\n=== Subtask 2: Safety Detection ===")
    test_path = DATA_DIR / "subtask2" / "test" / "subtask2_test.jsonl"
    records = load_jsonl(test_path)
    log.info(f"  Loaded {len(records)} test records")

    # Load thresholds
    thresh_paths = [
        MODEL_DIR / "subtask2_thresholds_v4.json",
        MODEL_DIR / "subtask2_thresholds_v3.json",
        MODEL_DIR / "subtask2_thresholds.json",
    ]
    thresholds = None
    for tp in thresh_paths:
        if tp.exists():
            with open(tp) as f:
                thresholds = json.load(f)
            log.info(f"  Loaded thresholds: {tp.name}")
            break

    # Extract trace features
    X_trace = extract_subtask2_features(records)
    log.info(f"  Trace features: {X_trace.shape}")

    # Multi-seed ensemble
    seeds = [42, 123, 456, 789, 1337]
    prob_a_all, prob_b_all = [], []
    for seed in seeds:
        ma = MODEL_DIR / f"subtask2_trace_model_a_seed{seed}.txt"
        mb = MODEL_DIR / f"subtask2_trace_model_b_seed{seed}.txt"
        if ma.exists() and mb.exists():
            m_a = lgb.Booster(model_file=str(ma))
            m_b = lgb.Booster(model_file=str(mb))
            n_a = m_a.num_feature()
            X_a = X_trace[:, :n_a] if X_trace.shape[1] >= n_a else np.pad(
                X_trace, ((0, 0), (0, n_a - X_trace.shape[1]))
            )
            n_b = m_b.num_feature()
            X_b = X_trace[:, :n_b] if X_trace.shape[1] >= n_b else np.pad(
                X_trace, ((0, 0), (0, n_b - X_trace.shape[1]))
            )
            prob_a_all.append(m_a.predict(X_a))
            prob_b_all.append(m_b.predict(X_b))

    if not prob_a_all:
        log.info("  ERROR: No subtask2 models found")
        return

    prob_a = np.mean(prob_a_all, axis=0)
    prob_b = np.mean(prob_b_all, axis=0)
    ta = thresholds.get("ta", thresholds.get("thresh_a", 0.36)) if thresholds else 0.36
    tb = thresholds.get("tb", thresholds.get("thresh_b", 0.40)) if thresholds else 0.40

    trace_preds = []
    for i in range(len(prob_a)):
        if prob_a[i] < ta:
            trace_preds.append("safe")
        elif prob_b[i] > tb:
            trace_preds.append("unsafe")
        else:
            trace_preds.append("unsafe")

    # Step-level: derive from trace prediction + prob_a as step confidence
    step_labels_per_record = []
    for i, record in enumerate(records):
        trace_text = record.get("reasoning_trace", "")
        steps = re.split(r"(?:步骤|Step)\s*\d+\s*[：:]", trace_text)
        steps = [s.strip() for s in steps if s.strip()]
        n_steps = max(1, len(steps))

        if trace_preds[i] == "safe":
            step_probs = [1.0] * n_steps
        else:
            base_unsafe = float(prob_a[i])
            step_probs = [round(1.0 - base_unsafe, 4)] * n_steps

        step_labels_per_record.append(step_probs)

    # Write CSV
    out_dir = OUTPUT_DIR / "subtask2"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "submission.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label", "detailed_label"])
        for i, record in enumerate(records):
            rid = record["id"]
            label = trace_preds[i]
            detailed = "|".join(str(p) for p in step_labels_per_record[i])
            writer.writerow([rid, label, detailed])

    n_safe = trace_preds.count("safe")
    n_unsafe = len(trace_preds) - n_safe
    log.info(f"  Predictions: {n_safe} safe, {n_unsafe} unsafe")
    log.info(f"  Written to: {csv_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_subtask1()
    generate_subtask2()
    log.info("\nDone! Submission files ready in: %s", OUTPUT_DIR)
