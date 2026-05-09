"""Evaluation metrics for Reasoning Trajectory Detection.

Provides standardized evaluation for both subtasks:
  - Subtask 1: Binary classification (human vs LLM)
  - Subtask 2: Multiclass trace-level + binary step-level safety
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray,
                    label_names: list[str] = None,
                    title: str = "") -> dict:
    """Evaluate binary classification (Subtask 1).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Names for classes (default: ["human", "llm"]).
        title: Optional title for the report.

    Returns:
        Dict with accuracy, f1_macro, and kappa.
    """
    if label_names is None:
        label_names = ["human", "llm"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per = f1_score(y_true, y_pred, average=None)
    kappa = cohen_kappa_score(y_true, y_pred)

    log.info("")
    log.info("=" * 60)
    if title:
        log.info("  %s", title)
        log.info("=" * 60)
    log.info("  Accuracy:     %.4f", acc)
    log.info("  F1 (macro):   %.4f", f1_macro)
    log.info("  Cohen Kappa:  %.4f", kappa)
    for i, name in enumerate(label_names):
        log.info("  F1 (%8s): %.4f", name, f1_per[i])

    cm = confusion_matrix(y_true, y_pred)
    log.info("")
    log.info("  Confusion Matrix:")
    header = "        " + "  ".join(f"{n:>8s}" for n in label_names)
    log.info("  %s", header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        log.info("  %8s %s", label_names[i], row_str)

    log.info("")
    log.info(classification_report(y_true, y_pred, target_names=label_names))

    return {"accuracy": acc, "f1_macro": f1_macro, "kappa": kappa}


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                        label_names: list[str] = None,
                        title: str = "") -> dict:
    """Evaluate multiclass classification (Subtask 2 trace-level).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Names for classes (default: safety categories).
        title: Optional title for the report.

    Returns:
        Dict with accuracy, f1_macro, f1_weighted, and kappa.
    """
    if label_names is None:
        label_names = ["safe", "potentially_unsafe", "unsafe"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    log.info("")
    log.info("=" * 60)
    if title:
        log.info("  %s", title)
        log.info("=" * 60)
    log.info("  Accuracy:       %.4f", acc)
    log.info("  F1 (macro):     %.4f", f1_macro)
    log.info("  F1 (weighted):  %.4f", f1_weighted)
    log.info("  Cohen Kappa:    %.4f", kappa)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    log.info("")
    log.info("  Confusion Matrix:")
    header = "              " + "  ".join(f"{n:>12s}" for n in label_names)
    log.info("  %s", header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12d}" for v in row)
        log.info("  %14s %s", label_names[i], row_str)

    log.info("")
    log.info(classification_report(y_true, y_pred, target_names=label_names))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
    }


def evaluate_step_level(y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = "") -> dict:
    """Evaluate per-step binary classification (Subtask 2 step-level).

    Args:
        y_true: Ground truth step labels.
        y_pred: Predicted step labels.
        title: Optional title for the report.

    Returns:
        Dict with accuracy, f1_macro, and kappa.
    """
    return evaluate_binary(
        y_true, y_pred, label_names=["safe", "unsafe"], title=title
    )


def evaluate_subtask2_full(trace_true, trace_pred,
                           step_true, step_pred,
                           title: str = "") -> dict:
    """Combined evaluation for subtask 2 (trace + step level).

    Args:
        trace_true: Ground truth trace-level labels.
        trace_pred: Predicted trace-level labels.
        step_true: Ground truth step-level labels.
        step_pred: Predicted step-level labels.
        title: Optional title for the report.

    Returns:
        Dict with 'trace' and 'step' sub-dicts of metrics.
    """
    log.info("")
    log.info("#" * 60)
    log.info("  SUBTASK 2 FULL EVALUATION")
    if title:
        log.info("  %s", title)
    log.info("#" * 60)

    trace_metrics = evaluate_multiclass(
        trace_true, trace_pred,
        title="Trace-Level Safety (3-class)"
    )
    step_metrics = evaluate_step_level(
        step_true, step_pred,
        title="Step-Level Safety (binary)"
    )

    return {"trace": trace_metrics, "step": step_metrics}
