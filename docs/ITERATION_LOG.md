# PAN-CLEF 2026 Reasoning Trajectory Detection: Iteration Log

> Deadline: May 13, 2026 (AoE)
> Current rank: 1st on both subtasks
> Budget: $43 Together.ai, $5 Modal
> Target: 99th percentile (S1 ≥ 0.92, S2 ≥ 0.75)

---

## Critical Distribution Shift Analysis (2026-05-04)

### S1: Test set is NOT math-only
| Domain       | Test Count | Test %  | Train % |
|-------------|-----------|---------|---------|
| Business/Other | 1,676 | 64%  | 0%      |
| Mathematics | 422       | 16%  | 100%    |
| Finance     | 348       | 13%  | 0%      |
| Coding      | 171       | 7%   | 0%      |

**Impact**: All math-specific features (boxed_answer, latex_density, inline_display_math_ratio, step_marker_count) are useless on 84% of test data. Generator-specific heuristics (opens_with_the, has_final_answer, reasoning_tag_present) trained on math may not transfer.

### S2: Test traces are 10x shorter
| Metric         | Train (safe) | Train (unsafe) | Test   |
|---------------|-------------|----------------|--------|
| Avg steps      | 38.1        | 55.6           | 5.07   |
| Generator field | present     | present        | ABSENT |

**Impact**: All feature densities, TF-IDF vectors, and thresholds calibrated on 40-60 step traces are miscalibrated for 5-step traces. Generator cannot be used as a feature signal.

---

## Baseline Scores (CodaBench Test Set)

| Date       | Subtask | Sub ID  | Macro F1 | Other Metrics     | Approach                                     |
|------------|---------|---------|----------|-------------------|----------------------------------------------|
| 2026-05-04 | S1      | 710431  | 0.6809   | Prec=0.6825       | LightGBM 28-feat (structural+gen-discriminative+emb) |
| 2026-05-04 | S2      | 710432  | 0.3616   | Soft F1=0.3796    | Dual-binary ordinal LGB, 5-seed HC+TF-IDF ensemble  |

## Validation Scores (Internal, math-only / long-trace)

| Date       | Subtask | Val F1   | Notes                                                   |
|------------|---------|----------|---------------------------------------------------------|
| 2026-02-18 | S1      | 0.7139   | subtask1_lgb_combined, 28 features                      |
| 2026-02-18 | S1      | 0.9746   | subtask1_lgb_structural (OVERFIT, 19 features)          |
| 2026-02-18 | S2      | 0.5018   | v12b_hc_tfidf_ensemble, 5 seeds, HC(0.6)+TF-IDF(0.4)   |
| 2026-02-18 | S2      | 0.4952   | v8b ordinal regression, weights w1=3.0 w2=3.0           |
| 2026-02-18 | S2      | ~0.49    | v4 dual-binary ordinal pincer                           |
| 2026-02-18 | S2      | ~0.48    | v2 single binary + thresholds                           |
| 2026-02-18 | S2-step | 0.5910   | v5 step-level LGB, threshold=0.39                       |

---

## Experiment Tracker

### Format: EXP-{subtask}-{number}

| Exp ID      | Date | Subtask | Approach | Val F1 | Test F1 | Status   | Notes |
|-------------|------|---------|----------|--------|---------|----------|-------|
| EXP-S1-001  | 02-18 | S1 | LGB structural 19-feat | 0.9746 | - | abandoned | Overfit to math domain |
| EXP-S1-002  | 02-18 | S1 | LGB combined 28-feat | 0.7139 | 0.6809 | baseline | Math-specific features hurt on non-math test |
| EXP-S2-001  | 02-18 | S2 | Binary + thresholds | ~0.48 | - | abandoned | - |
| EXP-S2-002  | 02-18 | S2 | Dual-binary pincer v4 | ~0.49 | - | superseded | - |
| EXP-S2-003  | 02-18 | S2 | Ordinal v8b | 0.4952 | - | superseded | - |
| EXP-S2-004  | 02-18 | S2 | v12b 5-seed HC+TF-IDF ensemble | 0.5018 | 0.3616 | baseline | Calibrated for 40-step traces; test has 5-step |
| | | | | | | | |

### NEW EXPERIMENTS (Revised Plan v3 — Post-Critique)

**Killed**: Eigenspectrum (#3, no evidence), Multi-compressor variance (#5, d=-0.044), 
TDA on 5-point clouds (#10, vanity math), LLM-as-judge (user prefers no API)

| Exp ID      | Suggestion | Priority | Approach | Status | Realistic Gain |
|-------------|-----------|----------|----------|--------|----------------|
| EXP-S1-003  | #1-evo | P0 | Byte-level forensic profiling (Unicode + whitespace + tokenizer artifacts) | pending | +0.02-0.04 (high-precision anchors) |
| EXP-S1-004  | #2+9-merge | P0 | Generation Process Inversion (single GPT-2 pass, 10 features) | pending | +0.05-0.10 |
| EXP-S1-005  | #4-evo | P0 | Vocabulary fingerprint (hapax d=-0.886, Yule's K, Heaps exponent) | pending | +0.04-0.07 |
| EXP-S2-005  | #6-evo | P0 | Compliance-Refusal-Instruction 3D trichotomy | pending | +0.10-0.15 |
| EXP-S2-006  | #8-evo | P1 | Pragmatic act classification (5 acts: refuse/redirect/warn/instruct/discuss) | pending | +0.05-0.08 |
| EXP-S2-007  | #B-new | P1 | Trace length bifurcation (separate models for short/medium/long) | pending | +0.03-0.06 |
| EXP-S1-006  | #C-new | P1 | Problem-conditioned prior (425 matched problems, 16% of test) | pending | +0.02-0.03 |
| EXP-BOTH-001 | #A-new | P1 | Semi-supervised self-training on test data (3-5 iterations) | pending | +0.03-0.06 |
| EXP-S1-007  | stack | P2 | Meta-ensemble stacking (all S1 signals into LR) | pending | +0.02-0.04 |
| EXP-S2-008  | stack | P2 | Meta-ensemble stacking (all S2 signals into LR) | pending | +0.02-0.04 |

### Realistic Targets (Post-Critique)
| Subtask | Current | Floor | Ceiling | Mechanism |
|---------|---------|-------|---------|-----------|
| S1 | 0.6809 | 0.78 | 0.87 | GPT-2 inversion + vocab fingerprint + typography + self-training |
| S2 | 0.3616 | 0.50 | 0.65 | Refusal trichotomy + pragmatic acts + bifurcation + self-training |

---

## Deep Failure Analysis (2026-05-04, post-autopsy)

### S2 Smoking Gun: Length-Dependent Safe Bias
| Trace Length | N (test) | Predicted Safe | Predicted Unsafe |
|---|---|---|---|
| <100 chars | 805 (28.4%) | 100.0% | 0.0% |
| 100-300 chars | 585 (20.6%) | 95.4% | 4.6% |
| 300-800 chars | 942 (33.2%) | 79.9% | 20.1% |
| >800 chars | 503 (17.7%) | 64.8% | 35.2% |

Root cause: All features are count-based, scale with length. Short text → zero features → safe.
Impact: 28% of test set (805 samples) has ZERO chance of being predicted unsafe.
Submitted: 86.1% safe vs validation truth of 56.3% safe (30pp over-prediction).

### S1 Generator Features Dead on Test
| Feature | Train Fire Rate | Test Fire Rate |
|---|---|---|
| has_final_answer | 93.6% (DeepSeek) | 0.0% |
| reasoning_tag_present | 99.3% (K2Think) | 0.0% |
| has_boxed_answer | 99% (all AI) | 29.3% |

5/9 generator features fire at ~0% on test. Test generators differ from training.

### S2 Generator Shift
Training: QWQ-32b, QwQ, R1-32b, ot-7b, r1-32b, r1-8b
Validation NEW: claude-3-7-sonnet, doubao-seed-1.6, gemini-2.5-flash, grok-3
4/8 val generators UNSEEN in training.

### S2 is Binary (Not Ternary)
Primary metric maps pot_unsafe → unsafe. Submission has 0 pot_unsafe predictions.
Entire dual-binary pincer/ordinal apparatus was solving wrong problem.

### No Threshold Optimization (S1)
Fixed 0.5 threshold. StratifiedKFold imported but never used.
Free +0.02-0.05 F1 available from threshold tuning alone.

### Three Structural Fixes (Priority Order)
1. **S2 short-trace classifier**: Dedicated refusal detector for <300 char traces
2. **S1 drop generator features**: Remove 9 dead features, add domain-agnostic
3. **Both threshold + CV**: k-fold, threshold optimization, calibration

---

## Empirical Feature Validation (2026-05-04)

### S1 Features by Cohen's d (Human vs GPT-5-Nano)
| Feature | d | Status |
|---|---|---|
| FW centroid cosine diff | 1.5626 | STRONGEST |
| Hapax ratio | -0.886 | Strong |
| Sentence compression CV | 0.310 | Moderate |
| Multi-compressor std | 0.402 | Moderate |
| Sentence word-length CV | 0.217 | Weak |
| FW transition ratio | 0.157 | Weak |
| Sentence length CV | 0.750 | Strong |
| Multi-compressor variance | -0.044 | Dead |

### S2 Heuristic Baselines (on validation, binary: safe vs not-safe)
| Approach | Val Macro F1 | Prediction Split |
|---|---|---|
| All-safe | 0.3601 | 100% safe |
| Our model (on test) | 0.3616 | 86.1% safe |
| Refusal regex only | 0.5246 | 24% safe |
| Jaccard threshold (t=0.045) | 0.5772 | 57% safe |
| **Refusal+Jaccard+Length** | **0.6686** | **42% safe** |

**CRITICAL**: Our model performs at the all-safe baseline level (0.36 ≈ 0.36).
Three regex patterns + Jaccard overlap achieve 0.67 with zero ML.

---

## Training Data Summary

### S1 Generators
| Generator | Count | Avg Chars | Boxed% | Final Answer% | We/doc |
|-----------|-------|-----------|--------|---------------|--------|
| human | 30,392 | 2,216 | 61.2% | 0.0% | 4.72 |
| gpt-5-nano | 30,301 | 3,792 | 97.5% | 3.2% | 3.78 |
| gemini-3-flash | 24,655 | 4,410 | 99.2% | 90.9% | 7.48 |
| k2think-v2 | 1,677 | 74,676 | 99.6% | 35.2% | 191.16 |
| deepseek-r1 | 531 | 2,630 | 99.2% | 93.6% | 0.25 |

### S2 Labels
| Label | Count | Avg Steps | Median Unsafe% |
|-------|-------|-----------|----------------|
| safe | 3,148 | 38.1 | 0.0% |
| potentially_unsafe | 1,387 | 49.0 | 0.9% |
| unsafe | 2,463 | 55.6 | 32.5% |
