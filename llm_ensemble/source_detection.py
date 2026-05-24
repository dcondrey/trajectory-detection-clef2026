"""Multi-LLM ensemble classifier for Subtask 1 (Source Detection).

Calls multiple LLMs to classify solutions as human or AI-generated,
ensembles via majority vote. Can also combine with LightGBM features.

Working APIs: Gemini 2.5 Flash, Groq (Llama 3.3 70B), Together (Llama 3.3 70B), Mistral Large
"""

import asyncio
import csv
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_subtask1

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT / "cache" / "llm_ensemble_s1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are detecting whether a solution to a problem was written by a human or by an AI language model.

Analyze the PROBLEM and SOLUTION below. Look for:
- AI tells: overly structured formatting, bullet points, step-by-step breakdowns, hedging language ("it's important to note"), formulaic transitions, excessive detail, consistent tone throughout
- Human tells: informal language, shortcuts, skipped steps, inconsistent formatting, personal voice, domain expertise shown naturally, errors/corrections, varying sentence complexity

Respond with exactly one word: human or ai"""


def _openai_compatible_call(url, api_key, model):
    """Create an async caller for OpenAI-compatible APIs."""
    import httpx

    async def caller(problem: str, solution: str) -> str | None:
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

    return caller


async def call_gemini(problem: str, solution: str) -> str | None:
    """Call Gemini API."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": f"PROBLEM: {problem}\n\nSOLUTION:\n{solution}"}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            except (KeyError, IndexError):
                return None
        log.warning("  Gemini error %d: %s", resp.status_code, resp.text[:200])
        return None


def build_callers() -> dict:
    """Build available LLM callers from environment variables."""
    callers = {}

    if os.environ.get("GEMINI_API_KEY"):
        callers["gemini"] = call_gemini

    if os.environ.get("GROQ_API_KEY"):
        callers["groq_llama70b"] = _openai_compatible_call(
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ["GROQ_API_KEY"],
            "llama-3.3-70b-versatile",
        )

    if os.environ.get("TOGETHER_API_KEY"):
        callers["together_llama70b"] = _openai_compatible_call(
            "https://api.together.xyz/v1/chat/completions",
            os.environ["TOGETHER_API_KEY"],
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )

    if os.environ.get("MISTRAL_API_KEY"):
        callers["mistral_large"] = _openai_compatible_call(
            "https://api.mistral.ai/v1/chat/completions",
            os.environ["MISTRAL_API_KEY"],
            "mistral-large-latest",
        )

    return callers


def parse_label(response: str | None) -> str:
    """Parse LLM response to human/ai label."""
    if response is None:
        return "unknown"
    text = response.strip().lower()
    if "human" in text:
        return "human"
    if "ai" in text or "llm" in text or "machine" in text or "artificial" in text:
        return "ai"
    return "unknown"


async def classify_single(record: dict, models: list[str], callers: dict,
                          semaphores: dict) -> dict:
    """Classify a single record with multiple LLMs."""
    problem = record.get("problem", "")[:500]
    solution = record["solution"][:3000]

    results = {}
    tasks = []
    for model_name in models:
        caller = callers[model_name]

        async def _call(name=model_name, fn=caller):
            async with semaphores[name]:
                for attempt in range(3):
                    try:
                        resp = await fn(problem, solution)
                        return name, resp
                    except Exception as e:
                        if attempt == 2:
                            log.warning("  %s failed: %s", name, e)
                            return name, None
                        await asyncio.sleep(2 ** attempt)

        tasks.append(_call())

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for item in responses:
        if isinstance(item, Exception):
            continue
        name, resp = item
        results[name] = parse_label(resp)

    return {
        "id": record.get("id", record.get("solution_id", "")),
        "model_labels": results,
    }


def ensemble_vote(model_labels: dict) -> str:
    """Majority vote. Ties go to ai."""
    labels = [l for l in model_labels.values() if l != "unknown"]
    if not labels:
        return "ai"
    human_count = sum(1 for l in labels if l == "human")
    ai_count = sum(1 for l in labels if l == "ai")
    if human_count > ai_count:
        return "human"
    return "ai"


async def run_ensemble(split: str, models: list[str], batch_size: int = 15,
                       cache_name: str = "default") -> list[dict]:
    """Run multi-LLM ensemble on all records."""
    records = load_subtask1(split)
    log.info("Loaded %d records (%s)", len(records), split)

    callers = build_callers()
    available = [m for m in models if m in callers]
    missing = [m for m in models if m not in callers]
    if missing:
        log.warning("Missing API keys for: %s", missing)
    models = available
    log.info("Using models: %s", models)

    cache_path = CACHE_DIR / f"{cache_name}_{split}.jsonl"

    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    cached[r["id"]] = r
        log.info("Loaded %d cached results", len(cached))

    to_classify = [r for r in records if r.get("id", r.get("solution_id", "")) not in cached]
    log.info("Need to classify %d records", len(to_classify))

    if to_classify:
        semaphores = {m: asyncio.Semaphore(5) for m in models}
        if "gemini" in semaphores:
            semaphores["gemini"] = asyncio.Semaphore(3)
        if "mistral_large" in semaphores:
            semaphores["mistral_large"] = asyncio.Semaphore(2)

        with open(cache_path, "a") as cache_f:
            for batch_start in range(0, len(to_classify), batch_size):
                batch = to_classify[batch_start:batch_start + batch_size]
                tasks = [classify_single(r, models, callers, semaphores) for r in batch]
                results = await asyncio.gather(*tasks)

                for result in results:
                    if result["id"] in cached:
                        cached[result["id"]]["model_labels"].update(result["model_labels"])
                        result = cached[result["id"]]
                    cached[result["id"]] = result
                    cache_f.write(json.dumps(result) + "\n")
                    cache_f.flush()

                done = min(batch_start + batch_size, len(to_classify))
                log.info("  Progress: %d / %d", done, len(to_classify))

    all_results = []
    for rec in records:
        rec_id = rec.get("id", rec.get("solution_id", ""))
        result = cached[rec_id]
        label = ensemble_vote(result["model_labels"])
        all_results.append({
            "id": rec_id,
            "label": label,
            "model_labels": result["model_labels"],
        })

    from collections import Counter
    label_counts = Counter(r["label"] for r in all_results)
    log.info("\nEnsemble predictions:")
    for lbl, cnt in sorted(label_counts.items()):
        log.info("  %s: %d (%.1f%%)", lbl, cnt, 100 * cnt / len(all_results))

    for model_name in models:
        model_labels = [r["model_labels"].get(model_name, "unknown") for r in all_results]
        model_counts = Counter(model_labels)
        log.info("  %s: %s", model_name, dict(model_counts))

    agree = sum(1 for r in all_results
                if len(set(l for l in r["model_labels"].values() if l != "unknown")) <= 1)
    log.info("  Agreement: %d / %d (%.1f%%)", agree, len(all_results),
             100 * agree / len(all_results))

    return all_results


def write_submission(results: list[dict], output_dir: Path):
    """Write submission CSV and zip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        for r in results:
            writer.writerow([r["id"], r["label"]])

    zip_path = output_dir / "subtask1_llm_ensemble.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "submission.csv")

    log.info("Submission: %s (%d predictions)", zip_path, len(results))
    return zip_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", default="gemini,groq_llama70b,together_llama70b,mistral_large")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--cache-name", default="ensemble_v1")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    log.info("=" * 60)
    log.info("  S1 Multi-LLM Ensemble Classifier")
    log.info("=" * 60)

    results = asyncio.run(run_ensemble(
        args.split, models,
        batch_size=args.batch_size,
        cache_name=args.cache_name,
    ))

    output_dir = PROJECT / "codabench_submissions" / args.split / "subtask1"
    write_submission(results, output_dir)


if __name__ == "__main__":
    main()
