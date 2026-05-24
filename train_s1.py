"""Subtask 1 v2: Source Detection with improved features + threshold optimization.

Changes from v1:
  - Removed 5 dead generator features (0% fire rate on test)
  - Added 7 domain-agnostic vocabulary fingerprint features
  - Threshold optimization via stratified k-fold CV
  - Multi-seed ensemble for robustness
"""

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_subtask1
from evaluate import evaluate_binary
from features import SUBTASK1_FEATURE_NAMES, extract_subtask1_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
MODEL_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = SUBTASK1_FEATURE_NAMES


def load_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    feat_path = CACHE_DIR / f"s1_{split}_features_v2.npy"
    label_path = CACHE_DIR / f"s1_{split}_labels_v2.npy"

    if feat_path.exists() and label_path.exists():
        log.info("Loading cached %s features (v2)...", split)
        return np.load(feat_path), np.load(label_path)

    log.info("Extracting %s features (v2)...", split)
    records = load_subtask1(split)
    X = extract_subtask1_features(records)
    y = np.array([r["label"] for r in records], dtype=np.int32)

    np.save(feat_path, X)
    np.save(label_path, y)
    log.info("  Cached %s to %s", X.shape, feat_path)
    return X, y


def train_lgb(X_train, y_train, X_val, y_val, seed=42):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "reg_alpha": 0.5,
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)

    model = lgb.train(
        params, train_set,
        num_boost_round=1500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    return model


def optimize_threshold(y_true, probs):
    """Find optimal threshold for macro F1."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.005):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def cv_threshold(X, y, n_splits=5, seeds=None):
    """Cross-validated threshold optimization."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    all_oof_probs = np.zeros(len(y))
    all_oof_counts = np.zeros(len(y))

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = train_lgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed=seed)
            fold_probs = model.predict(X[val_idx])
            all_oof_probs[val_idx] += fold_probs
            all_oof_counts[val_idx] += 1

    oof_probs = all_oof_probs / np.maximum(all_oof_counts, 1)
    thresh, f1 = optimize_threshold(y, oof_probs)
    log.info("\n  CV OOF threshold=%.3f, macro F1=%.4f", thresh, f1)
    return thresh, oof_probs


def main():
    log.info("=" * 60)
    log.info("  SUBTASK 1 v2: Improved Features + Threshold Optimization")
    log.info("=" * 60)

    X_train, y_train = load_features("train")
    X_val, y_val = load_features("validation")

    log.info("\nTrain: %d samples, %d features", X_train.shape[0], X_train.shape[1])
    log.info("  Human: %d, LLM: %d", (y_train == 0).sum(), (y_train == 1).sum())
    log.info("Val:   %d samples", X_val.shape[0])
    log.info("  Human: %d, LLM: %d", (y_val == 0).sum(), (y_val == 1).sum())

    # Step 1: CV threshold optimization on training data
    log.info("\n--- Cross-Validated Threshold Optimization ---")
    cv_thresh, _ = cv_threshold(X_train, y_train, n_splits=5)

    # Step 2: Train multi-seed ensemble on full training data
    seeds = [42, 123, 456, 789, 1024]
    models = []
    for seed in seeds:
        log.info("\n--- Training seed=%d ---", seed)
        model = train_lgb(X_train, y_train, X_val, y_val, seed=seed)
        models.append(model)

    # Step 3: Ensemble predictions on validation
    val_probs = np.mean([m.predict(X_val) for m in models], axis=0)

    # Also optimize threshold on validation (for comparison)
    val_thresh, val_f1 = optimize_threshold(y_val, val_probs)
    log.info("\n  Val-tuned threshold=%.3f, macro F1=%.4f", val_thresh, val_f1)
    log.info("  CV threshold=%.3f", cv_thresh)

    # Use CV threshold (less overfit to validation)
    final_thresh = cv_thresh
    val_preds = (val_probs > final_thresh).astype(int)

    log.info("\n--- Final Results (threshold=%.3f) ---", final_thresh)
    metrics = evaluate_binary(y_val, val_preds, title="S1 v2 — Validation Set")

    # Also show with default 0.5 for comparison
    val_preds_default = (val_probs > 0.5).astype(int)
    f1_default = f1_score(y_val, val_preds_default, average="macro")
    log.info("\n  (Comparison: default threshold=0.5 -> F1=%.4f)", f1_default)

    # Feature importance (from first model)
    importance = models[0].feature_importance(importance_type="gain")
    indices = np.argsort(importance)[::-1]
    log.info("\nFeature Importances (gain):")
    for i, idx in enumerate(indices):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}"
        log.info("  %2d. %-30s  %12.1f", i + 1, name, importance[idx])

    # Save models and config
    for i, model in enumerate(models):
        model.save_model(str(MODEL_DIR / f"subtask1_v2_seed{seeds[i]}.txt"))

    config = {
        "threshold": float(final_thresh),
        "cv_threshold": float(cv_thresh),
        "val_threshold": float(val_thresh),
        "seeds": seeds,
        "n_features": X_train.shape[1],
        "feature_names": FEATURE_NAMES,
        "val_f1": float(metrics["f1_macro"]),
    }
    with open(MODEL_DIR / "subtask1_v2_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("\nModels saved to %s/subtask1_v2_seed*.txt", MODEL_DIR)
    log.info("Config saved to %s/subtask1_v2_config.json", MODEL_DIR)


if __name__ == "__main__":
    main()
