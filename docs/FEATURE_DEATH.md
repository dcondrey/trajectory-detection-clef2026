# When Features Die: Domain Shift in AI-Generated Text Detection

## The Problem

We trained on math. We tested on business, finance, coding, and math. 84% of the test set came from domains absent in training. Features that achieved 97% validation F1 collapsed to 68% on the test set.

This document analyzes why, introduces a taxonomy of feature robustness under domain shift, and identifies vocabulary fingerprints as the surviving signal class.

## The Fire Rate Table

"Fire rate" measures how often a feature produces a non-default value. A feature with 0% fire rate on the test set is dead: it contributes nothing to classification.

| Feature | Train Fire Rate | Test Fire Rate | Status |
|---|---|---|---|
| `has_final_answer` | 93.6% (DeepSeek) | 0.0% | **DEAD** |
| `reasoning_tag_present` | 99.3% (K2Think) | 0.0% | **DEAD** |
| `has_boxed_answer` | 99.0% (all AI) | 29.3% | **DYING** |
| `latex_density` | 95%+ (math) | 16% | **DYING** |
| `step_marker_count` | 88% | 31% | **DYING** |
| `inline_display_math_ratio` | 91% | 14% | **DEAD** |
| `vocab_richness` | 100% | 100% | Alive |
| `compression_ratio` | 100% | 100% | Alive |
| `hapax_ratio` | 100% | 100% | Alive |
| `yules_k` | 100% | 100% | Alive |
| `heaps_exponent` | 100% | 100% | Alive |
| `sentence_length_cv` | 98% | 97% | Alive |

## Feature Taxonomy Under Domain Shift

### Domain-Anchored Features (Fragile)

Features that depend on the specific domain or content format of the training data. These fail catastrophically when the test distribution shifts.

**Examples:**
- `has_boxed_answer`: Only meaningful for math solutions. Business text never uses `\boxed{}`.
- `latex_density`: Math-specific. Finance and coding use different notation.
- `has_final_answer`: Generator-specific artifact of DeepSeek's output format.
- `reasoning_tag_present`: Generator-specific artifact of K2Think's XML tags.
- `inline_display_math_ratio`: Only computable on LaTeX-containing text.

**Failure mode:** These features achieve high importance during training because they correlate strongly with generator identity *within the training domain*. The model learns "text with `\boxed{}` is AI-generated" which is true for math but meaningless for business.

**Lesson:** Any feature that references domain-specific formatting, notation, or structure is at risk. If you can imagine a domain where the feature would always return its default value, it is domain-anchored.

### Domain-Portable Features (Moderate)

Features that transfer across *some* domains but not all. They measure properties that exist in many text types but whose distributions vary.

**Examples:**
- `compression_ratio`: Works across domains but calibration differs. Math text compresses differently than business prose.
- `digit_ratio`: Useful for math and finance, meaningless for pure prose.
- `step_marker_count`: Common in instructional text (math, coding) but rare in analytical text (business, finance).
- `bold_section_count`: Depends on markdown usage conventions, which vary by domain.

**Failure mode:** These features contribute positive signal but their optimal thresholds shift across domains. A compression ratio of 0.45 might separate human from AI in math, but the boundary might be 0.52 in business.

### Domain-Invariant Features (Robust)

Features that measure *how* text is generated rather than *what* it is about. These survive domain shift because they capture fundamental properties of the generation process.

**Examples and Cohen's d (human vs GPT-5-Nano):**

| Feature | Cohen's d | Interpretation |
|---|---|---|
| `hapax_ratio` | -0.886 | Humans produce more words used exactly once. LLMs draw from a more concentrated vocabulary. |
| `sentence_length_cv` | 0.750 | Humans write sentences of varying length. LLMs produce more uniform sentence lengths. |
| `yules_k` | 0.620 | LLMs have higher vocabulary repetitiveness (Yule's K). |
| `heaps_exponent` | -0.540 | Human vocabulary grows faster with text length (higher Heaps exponent). |
| `function_word_ratio` | 0.380 | LLMs use slightly more function words due to preference for complete grammatical constructions. |
| `sentence_compression_cv` | 0.310 | Humans vary more in per-sentence information density. |

**Why they work:** These features capture statistical properties of word choice, sentence construction, and information distribution that are consequences of the generation mechanism itself. A human writing about business uses a different vocabulary than a human writing about math, but in both cases they produce more hapax legomena, more variable sentence lengths, and faster vocabulary growth than an LLM covering the same topic.

## The Vocabulary Fingerprint Insight

The strongest domain-invariant features form a coherent cluster: they all measure different aspects of **vocabulary usage patterns**. We call this cluster the "vocabulary fingerprint."

### Why Vocabulary Fingerprints Are Domain-Invariant

LLMs generate text by sampling from a learned probability distribution over tokens. This process produces characteristic statistical signatures:

1. **Vocabulary concentration**: LLMs tend to reuse a smaller set of high-probability words, leading to lower hapax ratios and higher Yule's K.

2. **Uniform density**: The sampling process produces more consistent information density across sentences, reducing compression ratio CV.

3. **Formulaic growth**: Vocabulary grows more slowly with text length because the model draws from a fixed distribution, yielding lower Heaps exponents.

4. **Structural regularity**: Sentence lengths are more uniform because the model has consistent generation dynamics, regardless of topic.

These properties hold regardless of whether the LLM is writing about mathematics, business strategy, or Python code. The fingerprint is in the *process*, not the *product*.

## Practical Recommendations

1. **Audit every feature** for domain dependence before deploying. Ask: "In what domain would this feature always return its default value?"

2. **Compute fire rates** on held-out domains. If a feature fires at < 10% on any anticipated test domain, it is fragile.

3. **Prefer vocabulary fingerprint features** for cross-domain generalization. Hapax ratio, Yule's K, and Heaps' exponent are your anchors.

4. **Domain-anchored features are not useless.** They are highly discriminative *within their domain*. Use them, but weight them lower in ensemble settings where domain shift is expected.

5. **Validate with Cohen's d**, not just feature importance. A feature can have high LightGBM importance because it perfectly separates training domains, not because it captures a generalizable signal.

## Impact on Our System

After replacing 5 dead generator-specific features with 7 vocabulary fingerprint features:

| Metric | Before (v1) | After (v2) | Change |
|---|---|---|---|
| Validation F1 | 0.9746 | 0.7139 | -0.26 (less overfit) |
| Test F1 | 0.6809 | 0.78+ | +0.10 (better generalization) |

The validation F1 *decreased* because the new features are less discriminative on the training domain (math). But the test F1 *increased* because the surviving features generalize across domains. This is the textbook bias-variance tradeoff in action.
