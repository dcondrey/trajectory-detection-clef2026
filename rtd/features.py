"""Feature engineering for Reasoning Trajectory Detection.

Adapted from the sensemaking CES pipeline — structural, compression, NLI,
and embedding features for both subtasks.
"""

import math
import re
import string
import zlib
from collections import Counter

import numpy as np
from tqdm import tqdm


# ============================================================================
# Structural / surface features (no model required)
# ============================================================================

def _vocab_richness(text: str) -> float:
    """Type-token ratio."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _avg_word_length(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c in string.punctuation for c in text) / len(text)


def _latex_density(text: str) -> float:
    """Fraction of text that is LaTeX markup (\\, $, {, })."""
    if not text:
        return 0.0
    latex_chars = sum(1 for c in text if c in r"\${}^_")
    return latex_chars / len(text)


def _sentence_count(text: str) -> int:
    return len(re.split(r"[.!?]+", text.strip()))


def _has_boxed_answer(text: str) -> int:
    """Whether solution contains \\boxed{...} LaTeX answer format."""
    return 1 if r"\boxed" in text else 0


def _step_marker_count(text: str) -> int:
    """Count 'Step N:' markers in text."""
    return len(re.findall(r"Step \d+:", text))


def _compression_ratio(text: str) -> float:
    """zlib compression ratio — lower means more compressible/repetitive."""
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def _entropy(text: str) -> float:
    """Character-level Shannon entropy."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _repetition_ratio(text: str) -> float:
    """Fraction of repeated n-grams (n=4 words) — AI tends to repeat more."""
    tokens = text.lower().split()
    if len(tokens) < 4:
        return 0.0
    ngrams = [tuple(tokens[i:i+4]) for i in range(len(tokens) - 3)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


# ============================================================================
# Subtask 1: Source Detection Features
# ============================================================================

def _we_pronoun_frequency(text: str) -> float:
    """Frequency of 'we/We' pronoun usage."""
    tokens = text.split()
    if not tokens:
        return 0.0
    count = len(re.findall(r'\b[Ww]e\b', text))
    return count / len(tokens)


def _numbered_step_count(text: str) -> int:
    """Count numbered list items (1. 2. etc)."""
    return len(re.findall(r'^\s*\d+\.', text, re.MULTILINE))


def _thus_therefore_ratio(text: str) -> float:
    """Ratio of 'thus' to 'therefore'. LLMs prefer 'thus', humans prefer 'therefore'."""
    thus_count = len(re.findall(r'\b[Tt]hus\b', text))
    therefore_count = len(re.findall(r'\b[Tt]herefore\b', text))
    total = thus_count + therefore_count
    if total == 0:
        return 0.5
    return thus_count / total


def _bold_section_count(text: str) -> int:
    """Count bold markdown sections (**...**)."""
    return len(re.findall(r'\*\*[^*]+\*\*', text))


def _solution_is_empty_or_minimal(text: str) -> int:
    """Whether solution is empty or very short."""
    return 1 if len(text.strip()) < 10 else 0


# --- Domain-agnostic vocabulary fingerprint features (Cohen's d validated) ---

def _hapax_ratio(text: str) -> float:
    """Fraction of words that appear exactly once (hapax legomena). d=-0.886."""
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / len(freq)


def _yules_k(text: str) -> float:
    """Yule's K measure of vocabulary richness. Higher = more repetitive (LLM-like)."""
    tokens = text.lower().split()
    n = len(tokens)
    if n < 2:
        return 0.0
    freq = Counter(tokens)
    freq_spectrum = Counter(freq.values())
    m1 = n
    m2 = sum(i * i * v for i, v in freq_spectrum.items())
    if m1 == 0 or m1 == 1:
        return 0.0
    k = 10000.0 * (m2 - m1) / (m1 * m1)
    return k


def _heaps_exponent(text: str) -> float:
    """Heaps' law exponent estimate. Lower = more formulaic vocabulary growth."""
    tokens = text.lower().split()
    n = len(tokens)
    if n < 10:
        return 0.0
    v = len(set(tokens))
    return math.log(v) / math.log(n)


def _sentence_length_cv(text: str) -> float:
    """Coefficient of variation of sentence lengths. d=0.750 (humans more variable)."""
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lens = [len(s.split()) for s in sentences]
    mean_len = sum(lens) / len(lens)
    if mean_len < 1:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in lens) / len(lens)
    return math.sqrt(variance) / mean_len


FUNCTION_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should can could may might must to of in for on at by with from as "
    "into through during before after above below between under about this that "
    "these those and but or if while because so yet nor not very also then than "
    "i me my we our he him his she her it its they them their you your".split()
)


def _function_word_ratio(text: str) -> float:
    """Ratio of function words to total words. LLMs tend to have higher ratios."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in FUNCTION_WORDS) / len(tokens)


def _whitespace_pattern_score(text: str) -> float:
    """Ratio of double-newlines to single-newlines. LLMs use more structured spacing."""
    single = text.count("\n")
    double = text.count("\n\n")
    if single == 0:
        return 0.0
    return double / single


def _sentence_compression_cv(text: str) -> float:
    """CV of per-sentence compression ratios. Humans vary more in information density."""
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) < 3:
        return 0.0
    ratios = [_compression_ratio(s) for s in sentences]
    mean_r = sum(ratios) / len(ratios)
    if mean_r < 0.01:
        return 0.0
    variance = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
    return math.sqrt(variance) / mean_r


def extract_subtask1_features(records: list[dict], show_progress: bool = True) -> np.ndarray:
    """Extract structural features for source detection.

    Returns (N, F) feature matrix with 30 features.
    """
    features = []
    iterator = tqdm(records, desc="Extracting S1 features") if show_progress else records

    for r in iterator:
        sol = r["solution"]
        tokens = sol.split()
        word_count = len(tokens)
        char_count = len(sol)
        sent_count = _sentence_count(sol)

        f = [
            math.log1p(word_count),                     # 0: word_count_log
            math.log1p(char_count),                     # 1: char_count_log
            math.log1p(sent_count),                     # 2: sentence_count_log
            _vocab_richness(sol),                       # 3: vocab_richness
            _avg_word_length(sol),                      # 4: avg_word_length
            _digit_ratio(sol),                          # 5: digit_ratio
            _punct_ratio(sol),                          # 6: punct_ratio
            _latex_density(sol),                        # 7: latex_density
            _compression_ratio(sol),                    # 8: compression_ratio
            _entropy(sol),                              # 9: entropy
            _repetition_ratio(sol),                     # 10: repetition_ratio
            _has_boxed_answer(sol),                     # 11: has_boxed_answer
            word_count / max(sent_count, 1),            # 12: avg_sentence_length
            math.log1p(len(set(tokens))),               # 13: unique_word_count_log
            sol.count("\n\n") + 1,                      # 14: paragraph_count
            max((len(t) for t in tokens), default=0),   # 15: max_word_length
            sol.count("(") / max(char_count, 1),        # 16: parenthesis_density
            sol.count("\n") / max(char_count, 1),       # 17: newline_ratio
            # --- Domain-agnostic discriminative features ---
            _we_pronoun_frequency(sol),                  # 18: we_pronoun_freq
            _numbered_step_count(sol),                   # 19: numbered_step_count
            _thus_therefore_ratio(sol),                  # 20: thus_therefore_ratio
            _bold_section_count(sol),                    # 21: bold_section_count
            _solution_is_empty_or_minimal(sol),          # 22: solution_is_empty
            # --- Vocabulary fingerprint features ---
            _hapax_ratio(sol),                           # 23: hapax_ratio
            _yules_k(sol),                               # 24: yules_k
            _heaps_exponent(sol),                        # 25: heaps_exponent
            _sentence_length_cv(sol),                    # 26: sentence_length_cv
            _function_word_ratio(sol),                   # 27: function_word_ratio
            _whitespace_pattern_score(sol),              # 28: whitespace_pattern
            _sentence_compression_cv(sol),               # 29: sentence_compression_cv
        ]
        features.append(f)

    return np.array(features, dtype=np.float32)


SUBTASK1_FEATURE_NAMES = [
    "word_count_log", "char_count_log", "sentence_count_log",
    "vocab_richness", "avg_word_length", "digit_ratio", "punct_ratio",
    "latex_density", "compression_ratio", "entropy", "repetition_ratio",
    "has_boxed_answer", "avg_sentence_length",
    "unique_word_count_log", "paragraph_count", "max_word_length",
    "parenthesis_density", "newline_ratio",
    "we_pronoun_freq", "numbered_step_count",
    "thus_therefore_ratio", "bold_section_count", "solution_is_empty",
    "hapax_ratio", "yules_k", "heaps_exponent",
    "sentence_length_cv", "function_word_ratio", "whitespace_pattern",
    "sentence_compression_cv",
]


# ============================================================================
# Subtask 2: Safety Detection Features
# ============================================================================

def _unsafe_keyword_density(text: str) -> float:
    """Density of safety-relevant keywords in text."""
    keywords = {
        "hack", "exploit", "attack", "malware", "phishing", "inject",
        "vulnerability", "bypass", "crack", "steal", "illegal", "weapon",
        "bomb", "drug", "kill", "harm", "abuse", "fraud", "scam",
        "ransomware", "ddos", "trojan", "keylogger", "backdoor",
        "credentials", "password", "unauthorized", "compromise",
    }
    tokens = set(text.lower().split())
    if not tokens:
        return 0.0
    return len(tokens & keywords) / len(tokens)


def _refusal_indicator(text: str) -> float:
    """Density of refusal/safety-guardrail language."""
    patterns = [
        "i cannot", "i can't", "i should not", "i shouldn't",
        "i need to be careful", "against.*policy", "guidelines",
        "i must decline", "not appropriate", "harmful",
        "i will not", "i won't", "ethic", "responsible",
        "i'm not able", "cannot assist", "cannot help",
        "cannot provide", "not going to",
    ]
    text_lower = text.lower()
    count = sum(1 for p in patterns if p in text_lower)
    return count / max(len(text.split()), 1)


def _warning_word_count(text: str) -> int:
    """Count warning/caution/danger words. Safe=0.38, Unsafe=0.66 presence rate."""
    return len(re.findall(
        r'\b(warning|caution|dangerous|harmful|illegal|unethical|hazardous|toxic)\b',
        text, re.IGNORECASE
    ))


def _safety_language_count(text: str) -> int:
    """Count safety/ethical/responsible terms. Safe=0.51, Unsafe=0.75 presence rate."""
    return len(re.findall(
        r'\b(safe|safety|ethical|responsible|appropriate|legitimate|lawful|legal)\b',
        text, re.IGNORECASE
    ))


def _has_disclaimer(text: str) -> int:
    """Whether text contains disclaimer/note/caveat. Safe=0.02, Unsafe=0.07."""
    return 1 if re.search(
        r'\b(disclaimer|note:|important:|caveat|please note)\b', text, re.IGNORECASE
    ) else 0


def _dangerous_term_count(text: str) -> int:
    """Count dangerous domain terms. Safe=0.15, Unsafe=0.34 presence rate."""
    return len(re.findall(
        r'\b(weapon|exploit|hack|attack|poison|bomb|kill|drug|malware|'
        r'ransomware|phishing|trojan|keylogger|backdoor|ddos|'
        r'synthesize|manufacture|detonate|smuggle)\b',
        text, re.IGNORECASE
    ))


def _has_code_blocks(text: str) -> int:
    """Whether text contains code blocks. 0% in potentially_unsafe."""
    return 1 if re.search(r'```|def |import |class |function\s', text) else 0


def _authority_appeal_count(text: str) -> int:
    """Count appeals to authority/institutions — common jailbreak pattern.

    Jailbreak queries often cite fake or real authorities to legitimize requests:
    'According to NIST...', 'Leading experts at...', 'Published in...'
    """
    patterns = [
        r'\b(according to|experts? at|researchers? at|published in|report by)\b',
        r'\b(institute|academy|association|foundation|university|journal)\b',
        r'\b(documented|extensively studied|well-established|recognized)\b',
        r'\b(NIST|SANS|CISA|NIH|WHO|CDC|AACAP)\b',
    ]
    text_lower = text.lower()
    count = 0
    for p in patterns:
        count += len(re.findall(p, text_lower, re.IGNORECASE))
    return count


def _educational_framing_score(text: str) -> float:
    """Detect educational/research framing — common in jailbreak queries.

    Phrases like 'for academic research', 'to understand', 'for educational purposes'.
    """
    patterns = [
        "academic research", "educational purpose", "for research",
        "to understand", "for understanding", "theoretical",
        "for study", "learning purpose", "from a scholarly",
        "intellectual exercise", "hypothetical", "thought experiment",
        "for awareness", "prevention purpose", "security research",
        "for my dissertation", "for a class", "coursework",
    ]
    text_lower = text.lower()
    count = sum(1 for p in patterns if p in text_lower)
    return count / max(len(text.split()), 1)


def _query_complexity_score(text: str) -> float:
    """Ratio of subordinate clauses / long sentences — jailbreaks tend to be complex.

    Jailbreak queries use elaborate multi-clause sentences to hide intent.
    """
    clauses = len(re.findall(r'\b(which|that|who|whom|where|when|while|although|because|since|if|unless)\b',
                             text, re.IGNORECASE))
    words = len(text.split())
    return clauses / max(words, 1)


def _dialect_obfuscation_score(text: str) -> float:
    """Detect dialect/slang obfuscation — used to bypass filters.

    E.g., 'ain't', 'gonna', 'wanna', ''ow 'bout', 'mate'.
    """
    patterns = [
        r"\bain't\b", r"\bgonna\b", r"\bwanna\b", r"\bgotta\b",
        r"\b'bout\b", r"\b'ow\b", r"\bmate\b", r"\bbloke\b",
        r"\bfolks\b", r"\by'all\b", r"\bfixin'\b", r"\breckon\b",
        r"\bdunno\b", r"\bkinda\b", r"\bsorta\b",
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
    return count / max(len(text.split()), 1)


def _step_transition_features(steps: list[str]) -> dict:
    """Features about transitions between reasoning steps."""
    if len(steps) < 2:
        return {
            "avg_step_length": len(steps[0].split()) if steps else 0,
            "step_length_std": 0.0,
            "step_count": len(steps),
            "max_step_length": len(steps[0].split()) if steps else 0,
            "min_step_length": len(steps[0].split()) if steps else 0,
        }
    lens = [len(s.split()) for s in steps]
    return {
        "avg_step_length": np.mean(lens),
        "step_length_std": np.std(lens),
        "step_count": len(steps),
        "max_step_length": max(lens),
        "min_step_length": min(lens),
    }


def extract_subtask2_features(records: list[dict], show_progress: bool = True) -> np.ndarray:
    """Extract features for safety detection.

    Returns (N, F) feature matrix with 28 trace-level features.
    """
    features = []
    iterator = tqdm(records, desc="Extracting S2 features") if show_progress else records

    for r in iterator:
        trace = r["reasoning_trace"]
        query = r["query"]
        steps = _parse_steps(trace)

        trace_tokens = trace.split()
        query_tokens = query.split()
        trace_wc = len(trace_tokens)
        query_wc = len(query_tokens)
        trace_cc = len(trace)

        # Per-step unsafe keyword analysis
        step_unsafe_scores = [_unsafe_keyword_density(s) for s in steps] if steps else [0.0]
        n_steps = len(steps)

        # Late refusal: does the model refuse late in the trace?
        late_half = steps[n_steps // 2:] if n_steps > 1 else steps
        early_half = steps[:max(n_steps // 2, 1)]
        late_refusal = np.mean([_refusal_indicator(s) for s in late_half]) if late_half else 0.0
        early_engagement = np.mean([_unsafe_keyword_density(s) for s in early_half]) if early_half else 0.0

        trans = _step_transition_features(steps)

        f = [
            math.log1p(trace_wc),                              # 0: trace_word_count_log
            math.log1p(trace_cc),                              # 1: trace_char_count_log
            math.log1p(query_wc),                              # 2: query_word_count_log
            _vocab_richness(trace),                             # 3: vocab_richness
            _compression_ratio(trace),                          # 4: compression_ratio
            _entropy(trace),                                    # 5: entropy
            _repetition_ratio(trace),                           # 6: repetition_ratio
            _unsafe_keyword_density(query),                     # 7: query_unsafe_kw_density
            _unsafe_keyword_density(trace),                     # 8: trace_unsafe_kw_density
            _refusal_indicator(trace),                          # 9: refusal_indicator
            trans["step_count"],                                # 10: step_count
            trans["avg_step_length"],                           # 11: avg_step_length
            trans["step_length_std"],                           # 12: step_length_std
            trans["max_step_length"],                           # 13: max_step_length
            trans["min_step_length"],                           # 14: min_step_length
            np.mean(step_unsafe_scores),                        # 15: avg_step_unsafe_score
            query_wc / max(trace_wc, 1),                        # 16: query_trace_length_ratio
            _digit_ratio(trace),                                # 17: digit_ratio
            _punct_ratio(trace),                                # 18: punct_ratio
            late_refusal,                                       # 19: late_refusal_ratio
            early_engagement,                                   # 20: early_engagement_ratio
            # --- New safety-discriminative features ---
            _warning_word_count(trace),                         # 21: warning_word_count
            _safety_language_count(trace),                      # 22: safety_language_count
            _has_disclaimer(trace),                             # 23: has_disclaimer
            _dangerous_term_count(trace),                       # 24: dangerous_term_count
            _dangerous_term_count(query),                       # 25: query_dangerous_term_count
            _has_code_blocks(trace),                            # 26: has_code_blocks
            trace_cc / max(n_steps, 1),                         # 27: chars_per_step
        ]
        features.append(f)

    return np.array(features, dtype=np.float32)


def _parse_steps(trace: str) -> list[str]:
    """Parse reasoning trace into steps."""
    parts = re.split(r"Step \d+:\s*", trace)
    return [p.strip() for p in parts if p.strip()]


SUBTASK2_FEATURE_NAMES = [
    "trace_word_count_log", "trace_char_count_log", "query_word_count_log",
    "vocab_richness", "compression_ratio", "entropy", "repetition_ratio",
    "query_unsafe_kw_density", "trace_unsafe_kw_density", "refusal_indicator",
    "step_count", "avg_step_length", "step_length_std",
    "max_step_length", "min_step_length", "avg_step_unsafe_score",
    "query_trace_length_ratio", "digit_ratio", "punct_ratio",
    "late_refusal_ratio", "early_engagement_ratio",
    # Safety-discriminative features
    "warning_word_count", "safety_language_count", "has_disclaimer",
    "dangerous_term_count", "query_dangerous_term_count",
    "has_code_blocks", "chars_per_step",
]


# ============================================================================
# Subtask 2: Per-step features (for step-level classification)
# ============================================================================

def extract_step_features(step_text: str, step_idx: int, total_steps: int,
                          query: str) -> np.ndarray:
    """Extract features for a single reasoning step.

    Returns 1D array of 22 features for one step.
    """
    query_wc = len(query.split())
    step_wc = len(step_text.split())
    f = [
        math.log1p(step_wc),                        # 0: step word count
        _vocab_richness(step_text),                 # 1: step vocab richness
        _unsafe_keyword_density(step_text),         # 2: step unsafe keywords
        _refusal_indicator(step_text),              # 3: step refusal language
        _digit_ratio(step_text),                    # 4: step digit ratio
        _punct_ratio(step_text),                    # 5: step punct ratio
        step_idx / max(total_steps - 1, 1),         # 6: position in trace (0-1)
        1.0 if step_idx == 0 else 0.0,              # 7: is first step
        1.0 if step_idx == total_steps - 1 else 0.0, # 8: is last step
        _unsafe_keyword_density(query),             # 9: query unsafe density
        _compression_ratio(step_text),              # 10: step compression
        # Safety-discriminative step features
        _warning_word_count(step_text),             # 11: step warning words
        _dangerous_term_count(step_text),           # 12: step dangerous terms
        _safety_language_count(step_text),          # 13: step safety language
        _has_code_blocks(step_text),                # 14: step has code
        _has_disclaimer(step_text),                 # 15: step has disclaimer
        # Query-context features (query content is highly predictive of safety)
        math.log1p(query_wc),                       # 16: query word count
        _dangerous_term_count(query),               # 17: query dangerous terms
        _warning_word_count(query),                 # 18: query warning words
        _has_code_blocks(query),                    # 19: query has code
        total_steps,                                # 20: total steps in trace
        _entropy(step_text) if step_wc > 5 else 0, # 21: step entropy
    ]
    return np.array(f, dtype=np.float32)


STEP_FEATURE_NAMES = [
    "step_word_count_log", "step_vocab_richness", "step_unsafe_kw_density",
    "step_refusal_indicator", "step_digit_ratio", "step_punct_ratio",
    "step_position", "is_first_step", "is_last_step",
    "query_unsafe_kw_density", "step_compression_ratio",
    "step_warning_words", "step_dangerous_terms", "step_safety_language",
    "step_has_code", "step_has_disclaimer",
    "query_word_count_log", "query_dangerous_terms", "query_warning_words",
    "query_has_code", "total_steps", "step_entropy",
]


def extract_all_step_features(records: list[dict], show_progress: bool = True):
    """Extract per-step features and labels for all records.

    Returns:
        X: (total_steps, n_features) feature matrix
        y: (total_steps,) label array
        group_ids: (total_steps,) record index for each step
    """
    all_features = []
    all_labels = []
    all_groups = []

    iterator = tqdm(records, desc="Extracting step features") if show_progress else records

    for idx, r in enumerate(iterator):
        steps = _parse_steps(r["reasoning_trace"])
        labels = r["detailed_label"]

        # Align steps and labels (take min in case of mismatch)
        n = min(len(steps), len(labels))
        for i in range(n):
            feat = extract_step_features(steps[i], i, len(steps), r["query"])
            all_features.append(feat)
            all_labels.append(labels[i])
            all_groups.append(idx)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        np.array(all_groups, dtype=np.int32),
    )
