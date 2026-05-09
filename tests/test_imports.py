"""Smoke tests: verify all rtd and llm_ensemble modules import."""

import logging

log = logging.getLogger(__name__)


def test_import_rtd_package():
    """Top-level rtd package imports without error."""
    import rtd
    assert hasattr(rtd, "__version__")
    log.info("rtd package version: %s", rtd.__version__)


def test_import_rtd_features():
    """rtd.features exposes feature extraction functions and name lists."""
    from rtd.features import (
        SUBTASK1_FEATURE_NAMES,
        SUBTASK2_FEATURE_NAMES,
        STEP_FEATURE_NAMES,
        extract_subtask1_features,
        extract_subtask2_features,
        extract_step_features,
        extract_all_step_features,
    )
    assert isinstance(SUBTASK1_FEATURE_NAMES, list)
    assert isinstance(SUBTASK2_FEATURE_NAMES, list)
    assert isinstance(STEP_FEATURE_NAMES, list)
    assert callable(extract_subtask1_features)
    assert callable(extract_subtask2_features)
    assert callable(extract_step_features)
    assert callable(extract_all_step_features)


def test_import_rtd_data_loader():
    """rtd.data_loader exposes loading and parsing functions."""
    from rtd.data_loader import (
        load_jsonl,
        load_subtask1,
        load_subtask2,
        parse_reasoning_steps,
    )
    assert callable(load_jsonl)
    assert callable(load_subtask1)
    assert callable(load_subtask2)
    assert callable(parse_reasoning_steps)


def test_import_rtd_evaluate():
    """rtd.evaluate exposes evaluation metric functions."""
    from rtd.evaluate import (
        evaluate_binary,
        evaluate_multiclass,
        evaluate_step_level,
        evaluate_subtask2_full,
    )
    assert callable(evaluate_binary)
    assert callable(evaluate_multiclass)
    assert callable(evaluate_step_level)
    assert callable(evaluate_subtask2_full)


def test_import_rtd_refusal_detector():
    """rtd.refusal_detector exposes detection and classification functions."""
    from rtd.refusal_detector import (
        has_refusal,
        classify_trace,
        classify_sentence,
    )
    assert callable(has_refusal)
    assert callable(classify_trace)
    assert callable(classify_sentence)


def test_import_llm_ensemble_package():
    """llm_ensemble package imports without error."""
    import llm_ensemble
    assert hasattr(llm_ensemble, "run_source_ensemble")
    assert hasattr(llm_ensemble, "source_vote")
    assert hasattr(llm_ensemble, "run_safety_ensemble")
    assert hasattr(llm_ensemble, "safety_vote")


def test_import_llm_ensemble_source_detection():
    """llm_ensemble.source_detection module imports."""
    from llm_ensemble.source_detection import run_ensemble, ensemble_vote
    assert callable(run_ensemble)
    assert callable(ensemble_vote)


def test_import_llm_ensemble_safety_classification():
    """llm_ensemble.safety_classification module imports."""
    from llm_ensemble.safety_classification import run_ensemble, ensemble_vote
    assert callable(run_ensemble)
    assert callable(ensemble_vote)
