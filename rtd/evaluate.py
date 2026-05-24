"""Evaluation metrics and analysis for Reasoning Trajectory Detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray,
                    label_names: list[str] = None, title: str = ""):
    """Evaluate binary classification (Subtask 1)."""
    if label_names is None:
        label_names = ["human", "llm"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per = f1_score(y_true, y_pred, average=None)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  Cohen Kappa:  {kappa:.4f}")
    for i, name in enumerate(label_names):
        print(f"  F1 ({name:>8s}): {f1_per[i]:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Header
    header = "        " + "  ".join(f"{n:>8s}" for n in label_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {label_names[i]:>8s} {row_str}")

    print(f"\n{classification_report(y_true, y_pred, target_names=label_names)}")

    return {"accuracy": acc, "f1_macro": f1_macro, "kappa": kappa}


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                        label_names: list[str] = None, title: str = ""):
    """Evaluate multiclass classification (Subtask 2 trace-level)."""
    if label_names is None:
        label_names = ["safe", "potentially_unsafe", "unsafe"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")
    print(f"  Cohen Kappa:    {kappa:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    header = "              " + "  ".join(f"{n:>12s}" for n in label_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12d}" for v in row)
        print(f"  {label_names[i]:>14s} {row_str}")

    print(f"\n{classification_report(y_true, y_pred, target_names=label_names)}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted, "kappa": kappa}


def evaluate_step_level(y_true: np.ndarray, y_pred: np.ndarray, title: str = ""):
    """Evaluate per-step binary classification (Subtask 2 step-level)."""
    return evaluate_binary(y_true, y_pred, label_names=["safe", "unsafe"], title=title)


def evaluate_subtask2_full(trace_true, trace_pred, step_true, step_pred, title: str = ""):
    """Combined evaluation for subtask 2 (trace + step level)."""
    print(f"\n{'#' * 60}")
    print(f"  SUBTASK 2 FULL EVALUATION")
    if title:
        print(f"  {title}")
    print(f"{'#' * 60}")

    trace_metrics = evaluate_multiclass(
        trace_true, trace_pred,
        title="Trace-Level Safety (3-class)"
    )
    step_metrics = evaluate_step_level(
        step_true, step_pred,
        title="Step-Level Safety (binary)"
    )

    return {"trace": trace_metrics, "step": step_metrics}
