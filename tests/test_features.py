"""Tests for rtd.features with simple input strings."""

import logging

from rtd.features import (
    _vocab_richness,
    _avg_word_length,
    _digit_ratio,
    _punct_ratio,
    _compression_ratio,
    _entropy,
    _hapax_ratio,
    _yules_k,
    _heaps_exponent,
    _sentence_length_cv,
    _sentence_compression_cv,
    _function_word_ratio,
)

log = logging.getLogger(__name__)


class TestVocabRichness:

    def test_empty_string(self):
        assert _vocab_richness("") == 0.0

    def test_all_unique(self):
        assert _vocab_richness("the quick brown fox") == 1.0

    def test_repeated_words(self):
        result = _vocab_richness("the the the cat")
        assert 0.0 < result < 1.0
        log.info("vocab richness for repeated words: %.3f", result)


class TestAvgWordLength:

    def test_empty_string(self):
        assert _avg_word_length("") == 0.0

    def test_simple_sentence(self):
        result = _avg_word_length("hi there")
        assert result > 0.0


class TestDigitRatio:

    def test_empty_string(self):
        assert _digit_ratio("") == 0.0

    def test_no_digits(self):
        assert _digit_ratio("hello world") == 0.0

    def test_all_digits(self):
        assert _digit_ratio("12345") == 1.0

    def test_mixed(self):
        result = _digit_ratio("abc123")
        assert abs(result - 0.5) < 1e-9


class TestPunctRatio:

    def test_empty_string(self):
        assert _punct_ratio("") == 0.0

    def test_no_punct(self):
        assert _punct_ratio("hello") == 0.0


class TestCompressionRatio:

    def test_empty_string(self):
        assert _compression_ratio("") == 1.0

    def test_repetitive_text(self):
        repetitive = "word " * 200
        unique = " ".join(f"word{i}" for i in range(200))
        assert _compression_ratio(repetitive) < _compression_ratio(unique)


class TestEntropy:

    def test_empty_string(self):
        assert _entropy("") == 0.0

    def test_single_char(self):
        assert _entropy("aaaa") == 0.0

    def test_two_chars(self):
        result = _entropy("ab")
        assert abs(result - 1.0) < 1e-9


class TestHapaxRatio:

    def test_short_text(self):
        assert _hapax_ratio("a") == 0.0

    def test_all_unique(self):
        result = _hapax_ratio("the quick brown fox jumps over the lazy dog")
        assert 0.0 < result <= 1.0


class TestYulesK:

    def test_short_text(self):
        assert _yules_k("a") == 0.0

    def test_returns_positive(self):
        result = _yules_k("the the the cat cat dog bird fish fish fish")
        assert result > 0.0


class TestHeapsExponent:

    def test_short_text(self):
        assert _heaps_exponent("too short") == 0.0

    def test_returns_positive(self):
        text = " ".join(f"word{i}" for i in range(50))
        result = _heaps_exponent(text)
        assert result > 0.0


class TestSentenceLengthCV:

    def test_single_sentence(self):
        assert _sentence_length_cv("Just one sentence.") == 0.0

    def test_variable_lengths(self):
        text = "Short. This is a much longer sentence with many words in it."
        result = _sentence_length_cv(text)
        assert result > 0.0


class TestSentenceCompressionCV:

    def test_too_few_sentences(self):
        assert _sentence_compression_cv("One sentence.") == 0.0

    def test_multiple_sentences(self):
        text = (
            "This is a fairly normal sentence about nothing in particular. "
            "Numbers and data: 12345 67890 abcde fghij repeated repeated repeated. "
            "A unique creative flourish of vocabulary and expression here today."
        )
        result = _sentence_compression_cv(text)
        assert result >= 0.0


class TestFunctionWordRatio:

    def test_empty_string(self):
        assert _function_word_ratio("") == 0.0

    def test_all_function_words(self):
        result = _function_word_ratio("the a an is are")
        assert abs(result - 1.0) < 1e-9

    def test_mixed(self):
        result = _function_word_ratio("the cat sat on the mat")
        assert 0.0 < result < 1.0
