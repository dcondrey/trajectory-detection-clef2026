"""Subtask 1: Source Detection training with vocabulary fingerprint features.

Trains a multi-seed LightGBM ensemble with cross-validated threshold
optimization. Key insight: domain-anchored features (LaTeX, boxed answers)
die under distribution shift. Domain-invariant vocabulary fingerprints
(hapax ratio, Yule's K, Heaps' exponent) survive.

Usage:
    uv run python train_s1.py --data-dir data/ --output-dir models/
    uv run python train_s1.py --seeds 42,123,456 --n-folds 5
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from rtd.data_loader import load_subtask1
from rtd.evaluate import evaluate_binary
from rtd.features import SUBTASK1_FEATURE_NAMES, extract_subtask1_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_features(split: str, cache_dir: Path,
                  data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load or extract and cache features for a split."""
    feat_path = cache_dir / f"s1_{split}_features.npy"
    label_path = cache_dir / f"s1_{split}_labels.npy"

    if feat_path.exists() and label_path.exists():
        log.info("Loading cached %s features...", split)
        return np.load(feat_path), np.load(label_path)

    log.info("Extracting %s features...", split)
    records = load_subtask1(split, data_dir)
    X = extract_subtask1_features(records)
    y = np.array([r["label"] for r in records], dtype=np.int32)

    np.save(feat_path, X)
    np.save(label_path, y)
    log.info("  Cached %s to %s", X.shape, feat_path)
    return X, y


def train_lgb(X_train, y_train, X_val, y_val, seed=42):
    """Train a single LightGBM model."""
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

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=SUBTASK1_FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=SUBTASK1_FEATURE_NAMES, reference=train_set)

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
    log.info("  CV OOF threshold=%.3f, macro F1=%.4f", thresh, f1)
    return thresh, oof_probs


def main():
    parser = argparse.ArgumentParser(description="Train Subtask 1 source detection model")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: data/)")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Output directory for models")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"),
                        help="Cache directory for features")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                        help="Comma-separated random seeds")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    log.info("=" * 60)
    log.info("  SUBTASK 1: Source Detection Training")
    log.info("=" * 60)

    X_train, y_train = load_features("train", args.cache_dir, args.data_dir)
    X_val, y_val = load_features("validation", args.cache_dir, args.data_dir)

    log.info("Train: %d samples, %d features", X_train.shape[0], X_train.shape[1])
    log.info("  Human: %d, LLM: %d", (y_train == 0).sum(), (y_train == 1).sum())
    log.info("Val:   %d samples", X_val.shape[0])
    log.info("  Human: %d, LLM: %d", (y_val == 0).sum(), (y_val == 1).sum())

    # Cross-validated threshold optimization
    log.info("\n--- Cross-Validated Threshold Optimization ---")
    cv_thresh, _ = cv_threshold(X_train, y_train, n_splits=args.n_folds, seeds=seeds)

    # Train multi-seed ensemble on full training data
    models = []
    for seed in seeds:
        log.info("\n--- Training seed=%d ---", seed)
        model = train_lgb(X_train, y_train, X_val, y_val, seed=seed)
        models.append(model)

    # Ensemble predictions on validation
    val_probs = np.mean([m.predict(X_val) for m in models], axis=0)
    val_thresh, val_f1 = optimize_threshold(y_val, val_probs)
    log.info("  Val-tuned threshold=%.3f, macro F1=%.4f", val_thresh, val_f1)
    log.info("  CV threshold=%.3f", cv_thresh)

    final_thresh = cv_thresh
    val_preds = (val_probs > final_thresh).astype(int)

    log.info("\n--- Final Results (threshold=%.3f) ---", final_thresh)
    metrics = evaluate_binary(y_val, val_preds, title="S1 Validation Set")

    # Feature importance
    importance = models[0].feature_importance(importance_type="gain")
    indices = np.argsort(importance)[::-1]
    log.info("\nFeature Importances (gain):")
    for i, idx in enumerate(indices):
        name = SUBTASK1_FEATURE_NAMES[idx] if idx < len(SUBTASK1_FEATURE_NAMES) else f"f{idx}"
        log.info("  %2d. %-30s  %12.1f", i + 1, name, importance[idx])

    # Save models and config
    for i, model in enumerate(models):
        model.save_model(str(args.output_dir / f"subtask1_seed{seeds[i]}.txt"))

    config = {
        "threshold": float(final_thresh),
        "cv_threshold": float(cv_thresh),
        "val_threshold": float(val_thresh),
        "seeds": seeds,
        "n_features": X_train.shape[1],
        "feature_names": SUBTASK1_FEATURE_NAMES,
        "val_f1": float(metrics["f1_macro"]),
    }
    with open(args.output_dir / "subtask1_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("\nModels saved to %s/subtask1_seed*.txt", args.output_dir)
    log.info("Config saved to %s/subtask1_config.json", args.output_dir)


if __name__ == "__main__":
    main()
