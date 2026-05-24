"""Capability-asymmetry ablation for Subtask 1 (Source Detection).

Tests whether the Opus-Sonnet ensemble's success comes from:
  (a) capability-tier asymmetry (different tiers), or
  (b) same-family agreement (shared training corpus/alignment)

Ensemble pairs tested:
  1. Sonnet + GPT-4o-mini — different-family, same-tier       (~$3.50)
  2. Sonnet + Llama 3.3   — frontier + open-weight            (Sonnet shared)
  3. Qwen3 + Llama 3.3    — different-family, same-tier       (free, cached)

Reference: Opus+Sonnet submission (0.85 F1) loaded from CSV for comparison.
All pairs use the same agreement + LightGBM tiebreak logic as the winning system.

Usage:
  # Step 1: Collect classifications (run once, results are cached)
  uv run python src/ablation_capability_asymmetry.py --collect

  # Step 2: Evaluate all pairs (uses cached results, no API calls)
  uv run python src/ablation_capability_asymmetry.py --evaluate

  # Both in one go:
  uv run python src/ablation_capability_asymmetry.py --collect --evaluate

Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY environment variables.
Cached models (no API needed): qwen3_235b, llama_3.3_70b (from earlier experiments).
Total cost: ~$3-5 (Sonnet calls shared across two pairs, GPT-4o-mini is cheap).
"""

import asyncio
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_subtask1
from evaluate import evaluate_binary

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT / "cache" / "ablation_s1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT / "models"

SYSTEM_PROMPT = """You are detecting whether a solution to a problem was written by a human or by an AI language model.

Analyze the PROBLEM and SOLUTION below. Look for:
- AI tells: overly structured formatting, bullet points, step-by-step breakdowns, hedging language ("it's important to note"), formulaic transitions, excessive detail, consistent tone throughout
- Human tells: informal language, shortcuts, skipped steps, inconsistent formatting, personal voice, domain expertise shown naturally, errors/corrections, varying sentence complexity

Respond with exactly one word: human or ai"""


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------

def parse_label(text: str | None) -> str:
    """Parse LLM response to human/ai label."""
    if text is None:
        return "unknown"
    text = text.strip().lower()
    if "human" in text:
        return "human"
    if "ai" in text or "llm" in text or "machine" in text or "artificial" in text:
        return "ai"
    return "unknown"


async def call_openrouter(problem: str, solution: str, model: str) -> str | None:
    """Call OpenRouter API for model classification."""
    import httpx

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/dcondrey/trajectory-detection-clef2026",
            },
            json={
                "model": model,
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"PROBLEM: {problem}\n\nSOLUTION:\n{solution}"},
                ],
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().lower()
        log.warning("  OpenRouter %s error %d: %s", model, resp.status_code, resp.text[:200])
        return None


async def call_openai_compatible(problem: str, solution: str, model: str,
                                url: str, api_key_env: str) -> str | None:
    """Call any OpenAI-compatible API (OpenAI, Groq, Together)."""
    import httpx

    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"PROBLEM: {problem}\n\nSOLUTION:\n{solution}"},
                ],
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().lower()
        log.warning("  %s error %d: %s", model, resp.status_code, resp.text[:200])
        return None


# ---------------------------------------------------------------------------
# Batch classification with caching
# ---------------------------------------------------------------------------

async def classify_all(records: list[dict], model_name: str,
                       caller, batch_size: int = 10) -> dict[str, str]:
    """Classify all records with one model, caching results to disk."""
    cache_path = CACHE_DIR / f"{model_name}.jsonl"

    # Load existing cache
    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    cached[r["id"]] = r["label"]
        log.info("  %s: loaded %d cached results", model_name, len(cached))

    to_classify = [r for r in records
                   if r.get("id", r.get("solution_id", "")) not in cached]

    if not to_classify:
        log.info("  %s: all %d samples already cached", model_name, len(cached))
        return cached

    log.info("  %s: classifying %d remaining samples...", model_name, len(to_classify))

    sem = asyncio.Semaphore(2)

    async def _call_one(record):
        rid = record.get("id", record.get("solution_id", ""))
        problem = record.get("problem", "")[:500]
        solution = record["solution"][:3000]
        async with sem:
            for attempt in range(5):
                try:
                    resp = await caller(problem, solution)
                    if resp is not None:
                        return rid, parse_label(resp)
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                except Exception as e:
                    if attempt == 4:
                        log.warning("  %s failed on %s: %s", model_name, rid, e)
                        return rid, "unknown"
                    await asyncio.sleep(2 ** attempt)
            return rid, "unknown"

    with open(cache_path, "a") as cache_f:
        for batch_start in range(0, len(to_classify), batch_size):
            batch = to_classify[batch_start:batch_start + batch_size]
            tasks = [_call_one(r) for r in batch]
            results = await asyncio.gather(*tasks)

            for rid, label in results:
                cached[rid] = label
                cache_f.write(json.dumps({"id": rid, "label": label}) + "\n")
                cache_f.flush()

            done = min(batch_start + batch_size, len(to_classify))
            if done % 100 == 0 or done == len(to_classify):
                log.info("    %s progress: %d / %d", model_name, done, len(to_classify))

    return cached


def load_legacy_cache(model_name: str, records: list[dict]) -> dict[str, str] | None:
    """Load results from the original experiment caches."""
    legacy_dir = PROJECT / "cache" / "llm_ensemble_s1"

    # Map model names to legacy cache files
    legacy_map = {
        "qwen3_235b": "qwen3_235b_test.jsonl",
        "llama_3.3_70b": "groq_llama_v2_test.jsonl",
    }

    if model_name not in legacy_map:
        return None

    legacy_path = legacy_dir / legacy_map[model_name]
    if not legacy_path.exists():
        return None

    cached = {}
    with open(legacy_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rid = r.get("id", "")
                # Legacy format varies: some have "label", some have "model_labels"
                if "label" in r:
                    cached[rid] = parse_label(r["label"])
                elif "model_labels" in r:
                    # Take the first non-unknown label
                    for lbl in r["model_labels"].values():
                        parsed = parse_label(lbl)
                        if parsed != "unknown":
                            cached[rid] = parsed
                            break

    if cached:
        log.info("  %s: loaded %d results from legacy cache", model_name, len(cached))
        # Also write to ablation cache for consistency
        cache_path = CACHE_DIR / f"{model_name}.jsonl"
        if not cache_path.exists():
            with open(cache_path, "w") as f:
                for rid, label in cached.items():
                    f.write(json.dumps({"id": rid, "label": label}) + "\n")

    return cached


# ---------------------------------------------------------------------------
# Agreement + LightGBM tiebreak (replicates the winning system logic)
# ---------------------------------------------------------------------------

def load_lightgbm_predictions(records: list[dict]) -> dict[str, str]:
    """Load or compute LightGBM predictions for tiebreaking."""
    import lightgbm as lgb
    sys.path.insert(0, str(PROJECT / "src"))
    from features import extract_subtask1_features

    # Find the best model
    model_paths = [
        MODEL_DIR / "subtask1_v2_seed42.txt",
        MODEL_DIR / "subtask1_lgb_combined.txt",
        MODEL_DIR / "subtask1_lgb.txt",
    ]
    model = None
    for mp in model_paths:
        if mp.exists():
            model = lgb.Booster(model_file=str(mp))
            log.info("  LightGBM tiebreaker: %s", mp.name)
            break

    if model is None:
        log.warning("  No LightGBM model found, using 'ai' as default tiebreak")
        return {r.get("id", r.get("solution_id", "")): "ai" for r in records}

    X = extract_subtask1_features(records)
    n_expected = model.num_feature()
    if X.shape[1] != n_expected:
        X = X[:, :n_expected] if X.shape[1] > n_expected else np.pad(
            X, ((0, 0), (0, n_expected - X.shape[1]))
        )

    probs = model.predict(X)
    preds = {}
    for i, r in enumerate(records):
        rid = r.get("id", r.get("solution_id", ""))
        preds[rid] = "ai" if probs[i] > 0.5 else "human"
    return preds


def agreement_ensemble(labels_a: dict[str, str], labels_b: dict[str, str],
                       lgb_preds: dict[str, str], record_ids: list[str]) -> list[str]:
    """Agreement-based ensemble with LightGBM tiebreak.

    When both models agree → use their label.
    When they disagree → use LightGBM prediction.
    """
    predictions = []
    n_agree = 0
    n_disagree = 0

    for rid in record_ids:
        a = labels_a.get(rid, "unknown")
        b = labels_b.get(rid, "unknown")

        if a == "unknown" and b == "unknown":
            predictions.append(lgb_preds.get(rid, "ai"))
            n_disagree += 1
        elif a == "unknown":
            predictions.append(b)
            n_disagree += 1
        elif b == "unknown":
            predictions.append(a)
            n_disagree += 1
        elif a == b:
            predictions.append(a)
            n_agree += 1
        else:
            predictions.append(lgb_preds.get(rid, "ai"))
            n_disagree += 1

    log.info("    Agreement: %d / %d (%.1f%%)",
             n_agree, len(record_ids), 100 * n_agree / len(record_ids))

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAIRS = {
    "sonnet_gpt4omini": {
        "description": "Different-family, same-tier",
        "model_a": "sonnet",
        "model_b": "gpt4o_mini",
    },
    "sonnet_llama": {
        "description": "Same-family-tier + open-weight",
        "model_a": "sonnet",
        "model_b": "llama_3.3_70b",
    },
    "qwen3_llama": {
        "description": "Different-family, same-tier (open-weight)",
        "model_a": "qwen3_235b",
        "model_b": "llama_3.3_70b",
    },
}

# Reference: the winning Opus+Sonnet submission (0.85 F1) loaded from CSV
REFERENCE_PAIR = "opus_sonnet (submitted, 0.85 F1)"

MODEL_CALLERS = {
    "sonnet": lambda p, s: call_openrouter(p, s, "anthropic/claude-sonnet-4"),
    "gpt4o_mini": lambda p, s: call_openai_compatible(
        p, s, "gpt-4o-mini",
        "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "llama_3.3_70b": lambda p, s: call_openrouter(
        p, s, "meta-llama/llama-3.3-70b-instruct"),
    "qwen3_235b": lambda p, s: call_openrouter(
        p, s, "qwen/qwen3-235b-a22b"),
}

LEGACY_MODELS: set[str] = set()


async def collect_classifications(records: list[dict]):
    """Collect classifications from all models that need API calls."""
    # First, import any legacy caches
    for model_name in LEGACY_MODELS:
        legacy = load_legacy_cache(model_name, records)
        if not legacy:
            log.warning("  %s: no legacy cache found, will need API calls", model_name)

    # Determine which models need fresh API calls
    models_needed = set()
    for pair in PAIRS.values():
        for key in ("model_a", "model_b"):
            m = pair[key]
            if m not in LEGACY_MODELS:
                cache_path = CACHE_DIR / f"{m}.jsonl"
                if cache_path.exists():
                    with open(cache_path) as f:
                        n_cached = sum(1 for line in f if line.strip())
                    if n_cached >= len(records):
                        continue
                models_needed.add(m)

    if not models_needed:
        log.info("All models already cached. Skipping collection.")
        return

    log.info("Models needing API calls: %s", models_needed)

    for model_name in sorted(models_needed):
        if model_name in MODEL_CALLERS:
            await classify_all(records, model_name, MODEL_CALLERS[model_name])
        else:
            log.warning("  No caller for %s", model_name)


def evaluate_pairs(records: list[dict]):
    """Evaluate all ensemble pairs and print comparison table."""
    record_ids = [r.get("id", r.get("solution_id", "")) for r in records]

    # Load ground truth (test set has no labels, so we compare to submission)
    # Check if we have ground truth labels
    has_labels = "label" in records[0] or "generator" in records[0]

    # Load Opus+Sonnet submission as reference
    opus_sonnet_preds = {}
    sub_path = PROJECT / "codabench_submissions" / "subtask1" / "submission_v2.csv"
    if not sub_path.exists():
        sub_path = PROJECT / "codabench_submissions" / "subtask1" / "submission.csv"
    if sub_path.exists():
        with open(sub_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                opus_sonnet_preds[row["ID"]] = row["label"]
        log.info("  Loaded Opus+Sonnet reference: %d samples from %s",
                 len(opus_sonnet_preds), sub_path.name)

    if not has_labels:
        log.info("\nTest set has no ground truth labels.")
        log.info("Comparing pairs by agreement with Opus+Sonnet reference (0.85 F1)\n")

    # Load LightGBM predictions
    lgb_preds = load_lightgbm_predictions(records)

    # Load all model labels
    all_labels = {}
    for model_name in set(p[k] for p in PAIRS.values() for k in ("model_a", "model_b")):
        cache_path = CACHE_DIR / f"{model_name}.jsonl"
        if cache_path.exists():
            labels = {}
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        labels[r["id"]] = r["label"]
            all_labels[model_name] = labels
            log.info("  Loaded %s: %d samples", model_name, len(labels))
        else:
            log.warning("  Missing cache for %s", model_name)
            all_labels[model_name] = {}

    # Evaluate each pair
    log.info("\n" + "=" * 80)
    log.info("  CAPABILITY-ASYMMETRY ABLATION RESULTS")
    log.info("=" * 80)

    results_table = []

    for pair_name, pair_info in PAIRS.items():
        model_a = pair_info["model_a"]
        model_b = pair_info["model_b"]
        desc = pair_info["description"]

        labels_a = all_labels.get(model_a, {})
        labels_b = all_labels.get(model_b, {})

        if not labels_a or not labels_b:
            log.warning("  Skipping %s: missing data", pair_name)
            continue

        log.info("\n--- %s: %s ---", pair_name, desc)
        log.info("    Models: %s + %s", model_a, model_b)

        preds = agreement_ensemble(labels_a, labels_b, lgb_preds, record_ids)

        pred_counts = Counter(preds)
        log.info("    Predictions: %s", dict(pred_counts))

        if has_labels:
            y_true = np.array([r.get("label", 0) for r in records])
            y_pred = np.array([1 if p == "ai" else 0 for p in preds])
            metrics = evaluate_binary(y_true, y_pred,
                                      title=f"{pair_name}: {desc}")
            results_table.append({
                "pair": pair_name,
                "description": desc,
                "model_a": model_a,
                "model_b": model_b,
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
            })

    # Summary table
    if results_table:
        log.info("\n" + "=" * 80)
        log.info("  SUMMARY")
        log.info("=" * 80)
        log.info("  %-25s %-35s %6s %6s", "Pair", "Description", "F1", "Acc")
        log.info("  " + "-" * 75)
        for r in sorted(results_table, key=lambda x: -x["f1_macro"]):
            log.info("  %-25s %-35s %6.3f %6.3f",
                     r["pair"], r["description"], r["f1_macro"], r["accuracy"])

        results_path = CACHE_DIR / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(results_table, f, indent=2)
        log.info("\n  Results saved to: %s", results_path)
    else:
        # No ground truth — compare each pair against the Opus+Sonnet submission
        log.info("\n" + "=" * 80)
        log.info("  PAIR COMPARISON vs OPUS+SONNET REFERENCE (0.85 F1)")
        log.info("=" * 80)
        log.info("  %-25s %10s %10s %10s", "Pair", "Agree/Ref", "Pct", "Pair-Agree")
        log.info("  " + "-" * 60)

        for pair_name, pair_info in PAIRS.items():
            model_a = pair_info["model_a"]
            model_b = pair_info["model_b"]
            labels_a = all_labels.get(model_a, {})
            labels_b = all_labels.get(model_b, {})
            if not labels_a or not labels_b:
                continue

            pair_preds = agreement_ensemble(labels_a, labels_b, lgb_preds, record_ids)

            # Agreement with Opus+Sonnet reference
            ref_agree = sum(1 for rid, pred in zip(record_ids, pair_preds)
                           if opus_sonnet_preds.get(rid) == pred)
            ref_pct = 100 * ref_agree / len(record_ids) if record_ids else 0

            # Internal pair agreement rate
            internal_agree = sum(1 for rid in record_ids
                                if labels_a.get(rid) == labels_b.get(rid)
                                and labels_a.get(rid) != "unknown")
            internal_total = sum(1 for rid in record_ids
                                if labels_a.get(rid, "unknown") != "unknown"
                                and labels_b.get(rid, "unknown") != "unknown")
            internal_pct = 100 * internal_agree / internal_total if internal_total else 0

            log.info("  %-25s %7d/%d %9.1f%% %9.1f%%",
                     pair_name, ref_agree, len(record_ids), ref_pct, internal_pct)

        if opus_sonnet_preds:
            log.info("\n  Note: Opus+Sonnet reference achieved 0.85 F1 on leaderboard.")
            log.info("  Higher agreement with reference suggests similar quality;")
            log.info("  submit each pair to leaderboard for true F1 comparison.")


async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Capability-asymmetry ablation experiment")
    parser.add_argument("--collect", action="store_true",
                        help="Collect classifications from API models")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate all pairs")
    parser.add_argument("--split", default="test",
                        help="Data split to use (default: test)")
    args = parser.parse_args()

    if not args.collect and not args.evaluate:
        parser.print_help()
        return

    records = load_subtask1(args.split)
    log.info("Loaded %d records (%s)", len(records), args.split)

    if args.collect:
        await collect_classifications(records)

    if args.evaluate:
        evaluate_pairs(records)


if __name__ == "__main__":
    asyncio.run(main())
