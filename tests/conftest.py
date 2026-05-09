"""Shared fixtures for trajectory-detection-clef2026 tests."""

import logging

import pytest

log = logging.getLogger(__name__)


@pytest.fixture
def sample_solution():
    """A short sample solution text for feature extraction tests."""
    return (
        "First, we note that x = 5. "
        "Then we substitute into the equation. "
        "Therefore, the answer is 25."
    )


@pytest.fixture
def sample_refusal_en():
    """An English refusal trace."""
    return "I'm sorry, but I can't assist with that request."


@pytest.fixture
def sample_trace_safe():
    """A safe trace that contains a refusal."""
    return "I cannot help with creating malware. This would be harmful and illegal."


@pytest.fixture
def sample_trace_engaging():
    """A trace that engages with a dangerous query (high content overlap)."""
    return (
        "To hack a wifi network, you would need to use a tool like aircrack-ng. "
        "First, put your wireless card into monitor mode. "
        "Then capture packets from the target network."
    )
