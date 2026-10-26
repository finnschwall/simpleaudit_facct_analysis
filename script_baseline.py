"""
SimpleAudit Baseline Evaluation Script
=======================================

Run each model against all scenario packs, then split the combined results
into per-pack JSON files with summary scores.

Usage
-----
Run experiments:

    python script.py

Split combined results into per-pack files:

    python script.py --split-only

Override settings via CLI flags or environment variables:

    INFERENCE_BASE_URL=http://gpu:8000/v1 \\
    JUDGE_BASE_URL=http://gpu:8001/v1 \\
    python script.py --output-dir ./my-results

See ``python script.py --help`` for all options.
"""

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from simpleaudit.experiment import AuditExperiment


# ── Configuration ─────────────────────────────────────────────────────────────

# Inference server (vLLM)
INFERENCE_BASE_URL = os.environ.get("INFERENCE_BASE_URL", "http://localhost:9001/v1")
INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY", "")

# Judge model (separate vLLM instance)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:9002/v1")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", INFERENCE_API_KEY)

# Models to evaluate
MODELS = [
    "google/gemma-3-4b-it",
    "Qwen/Qwen3.5-4B",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3.5-9B",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

# Scenario packs
PACKS = [
    "safety_v1",
    "health_v1",
    "system_prompt_v1",
    "norwegian_language_v1",
    "norwegian_safety_v1",
    "norwegian_healthcare_v1",
    "norwegian_public_sector_v1",
]

# Experiment settings
N_REPETITIONS = 10
MAX_TURNS = 5
MAX_WORKERS = 10
MAX_RETRIES = 5
RETRY_DELAY = 0

# Output directories
BASE_DIR = Path(os.environ.get("BASE_DIR", "./results"))
SAVE_DIR = BASE_DIR / "sa-all_packs"
SPLIT_DIR = BASE_DIR / "sa-baseline"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_packs(pack_names):
    """Load all scenarios from the given scenario packs."""
    scenarios = []
    for pack_name in pack_names:
        ds = load_dataset("SimulaMet/simpleaudit-scenario-packs", pack_name, split="test")
        for item in ds:
            scenarios.append({
                "name": f"{pack_name}__{item['name']}",
                "description": item["description"],
            })
    return scenarios


def model_config(model_name):
    """Build model config dict for a given model name."""
    return {
        "model": model_name,
        "provider": "vllm",
        "base_url": INFERENCE_BASE_URL,
        "api_key": INFERENCE_API_KEY,
    }


def run_experiment(model_info, scenarios, save_dir, *,
                    n_repetitions=None, max_turns=None, max_workers=None):
    """Run an audit experiment with retry logic."""
    if n_repetitions is None:
        n_repetitions = N_REPETITIONS
    if max_turns is None:
        max_turns = MAX_TURNS
    if max_workers is None:
        max_workers = MAX_WORKERS
    attempt = 0
    while True:
        try:
            experiment = AuditExperiment(
                models=[model_info],
                judge_model=JUDGE_MODEL,
                judge_provider="vllm",
                judge_base_url=JUDGE_BASE_URL,
                judge_api_key=JUDGE_API_KEY,
                n_repetitions=n_repetitions,
                save_dir=str(save_dir),
            )
            experiment.run(scenarios, max_turns=max_turns, max_workers=max_workers)
            return True
        except Exception as e:
            attempt += 1
            print(f"  Error (attempt {attempt}): {e}")
            traceback.print_exc()
            if MAX_RETRIES and attempt >= MAX_RETRIES:
                print("  Max retries reached. Skipping.")
                return False
            time.sleep(RETRY_DELAY)


# ── Result Splitting ──────────────────────────────────────────────────────────

SEVERITY_SCORES = {
    "ERROR": 0, "critical": 0,
    "high": 1, "medium": 2, "low": 3, "pass": 4,
}


def compute_score(results):
    """Compute normalized score (0-100) from severity distribution."""
    if not results:
        return 0.0
    total = sum(SEVERITY_SCORES.get(r["severity"]) for r in results)
    return round((total / (len(results) * 4)) * 100, 1)


def split_run_into_packs(run_path, model_label):
    """Split a combined run JSON into per-pack result files."""
    with open(run_path) as f:
        data = json.load(f)

    # Group results by scenario pack
    by_pack = defaultdict(list)
    for result in data["results"]:
        scenario_name = result["scenario_name"]
        for pack_name in PACKS:
            if scenario_name.startswith(f"{pack_name}__"):
                result = dict(result)
                result["scenario_name"] = scenario_name[len(pack_name) + 2:]
                by_pack[pack_name].append(result)
                break
        else:
            print(f"  WARNING: no pack matched for '{scenario_name}'")

    # Write per-pack result files
    for pack_name, results in by_pack.items():
        out_dir = SPLIT_DIR / pack_name / model_label
        out_dir.mkdir(parents=True, exist_ok=True)

        severity_dist = defaultdict(int)
        for r in results:
            severity_dist[r["severity"]] += 1

        pack_data = {
            **data,
            "results": results,
            "summary": {
                **data.get("summary", {}),
                "total_scenarios": len(results),
                "score": compute_score(results),
                "passed": sum(1 for r in results if r["severity"] == "pass"),
                "failed": sum(1 for r in results if r["severity"] != "pass"),
                "severity_distribution": dict(severity_dist),
            },
        }

        out_path = out_dir / run_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pack_data, f, indent=2, ensure_ascii=False)
        print(f"  -> {out_path} ({len(results)} scenarios)")


def split_all_results():
    """Split all combined result files into per-pack outputs."""
    if not SAVE_DIR.exists():
        print(f"Save directory not found: {SAVE_DIR}")
        return

    for model_dir in sorted(d for d in SAVE_DIR.iterdir() if d.is_dir()):
        run_files = sorted(model_dir.glob("run_*.json"))
        if not run_files:
            continue
        print(f"\nSplitting {model_dir.name} ({len(run_files)} runs)")
        for run_file in run_files:
            split_run_into_packs(run_file, model_dir.name)

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleAudit baseline evaluation script.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Split combined results into per-pack files instead of running experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Base output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=N_REPETITIONS,
        help="Number of repetitions per scenario (default: %(default)s).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help="Maximum turns per audit (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Parallel workers (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override output dirs based on CLI
    save_dir = args.output_dir / "sa-all_packs"
    split_dir = args.output_dir / "sa-baseline"

    if args.split_only:
        # Temporarily override module-level SPLIT_DIR
        global SPLIT_DIR, SAVE_DIR
        SPLIT_DIR = split_dir
        SAVE_DIR = save_dir
        split_all_results()
        return

    print("Loading scenarios...")
    scenarios = load_packs(PACKS)
    print(f"  Total: {len(scenarios)} scenarios across {len(PACKS)} packs\n")

    for model_name in MODELS:
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        model_info = model_config(model_name)
        success = run_experiment(
            model_info, scenarios, save_dir,
            n_repetitions=args.repetitions,
            max_turns=args.max_turns,
            max_workers=args.workers,
        )

        if success:
            print(f"  Completed {model_name}\n")
        else:
            print(f"  Failed {model_name}\n")


if __name__ == "__main__":
    main()
