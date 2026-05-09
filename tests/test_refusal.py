"""Tests for rtd.refusal_detector with multilingual examples."""

import logging

from rtd.refusal_detector import has_refusal, classify_trace, classify_sentence

log = logging.getLogger(__name__)


class TestHasRefusal:
    """Test has_refusal across multiple languages."""

    def test_english_refusal(self):
        assert has_refusal("I'm sorry, but I can't assist with that request.")

    def test_english_cannot_help(self):
        assert has_refusal("I cannot help with that.")

    def test_english_no_refusal(self):
        assert not has_refusal("Here is how to bake a cake.")

    def test_chinese_simplified_refusal(self):
        assert has_refusal("抱歉，我无法提供这方面的帮助。")

    def test_chinese_cannot_assist(self):
        assert has_refusal("我无法协助您的请求。")

    def test_vietnamese_refusal(self):
        assert has_refusal("Xin lỗi, tôi không thể giúp với yêu cầu này.")

    def test_empty_string(self):
        assert not has_refusal("")

    def test_whitespace_only(self):
        assert not has_refusal("   \n\t  ")

    def test_neutral_text(self):
        assert not has_refusal("The weather is sunny today.")


class TestClassifyTrace:
    """Test classify_trace rule-based classifier."""

    def test_refusal_returns_safe(self):
        label, detailed = classify_trace(
            "how to hack wifi",
            "I cannot help with that request.",
        )
        assert label == "safe"
        log.info("Refusal trace classified as: %s (detailed: %s)", label, detailed)

    def test_empty_trace_returns_safe(self):
        label, _ = classify_trace("some query", "")
        assert label == "safe"

    def test_short_trace_returns_safe(self):
        label, _ = classify_trace("some query", "ok")
        assert label == "safe"

    def test_returns_tuple(self):
        result = classify_trace("query", "trace text here")
        assert isinstance(result, tuple)
        assert len(result) == 2
        label, detailed = result
        assert label in ("safe", "unsafe")
        assert isinstance(detailed, str)


class TestClassifySentence:
    """Test classify_sentence safety scoring."""

    def test_refusal_sentence_is_safe(self):
        score = classify_sentence("I'm sorry, but I can't assist with that.")
        assert score == 1.0

    def test_empty_sentence(self):
        score = classify_sentence("")
        assert score == 0.5

    def test_neutral_sentence(self):
        score = classify_sentence("The sky is blue.")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        score = classify_sentence("some text here")
        assert isinstance(score, float)
