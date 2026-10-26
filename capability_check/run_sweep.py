#!/usr/bin/env python3
"""Run the capability sweep: 6 models (3 safe/abliterated pairs) x 7 EuroEval datasets.

One `euroeval` call per (model, dataset) cell. zebra-puzzles-hard-en runs through
euroeval_jsonfix.py so the server enforces JSON output (see README).

EuroEval appends one record per cell to <out>/euroeval_benchmark_results.jsonl; per-cell
logs go to <out>/logs/. Rerunning skips cells that already have a record with the
requested number of iterations, so an interrupted sweep can be resumed.

Usage:
    python run_sweep.py --dry-run                 # print the commands
    python run_sweep.py                           # full sweep, n=10, into out/
    python run_sweep.py --datasets arc --models xs_safe,xs_abl --num-iterations 1
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {  # model id on the server == GGUF file name without .gguf
    "xs_safe": "Qwen3.5-4B-Q6_K",
    "xs_abl": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q6_K",
    "m_safe": "Qwen3.5-35B-A3B-Q6_K",
    "m_abl": "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K",
    "s_safe": "Qwen3.5-9B-Q6_K",
    "s_abl": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K",
}
DATASETS = ["nrk-quiz-qa", "mmlu-no", "multi-ifeval-nb", "norquad",
            "mmlu", "arc", "zebra-puzzles-hard-en"]
JSONFIX_DATASETS = {"zebra-puzzles-hard-en"}


def done_cells(results_file, num_iterations):
    done = set()
    if os.path.exists(results_file):
        for line in open(results_file):
            if not line.strip():
                continue
            rec = json.loads(line)
            details = rec["eval_library"]["additional_details"]
            raw = details["raw_results"]
            raw = json.loads(raw) if isinstance(raw, str) else raw
            if len(raw) == num_iterations:
                done.add((rec["model_info"]["id"], details["dataset"]))
    return done


def command(model_name, dataset, args):
    if dataset in JSONFIX_DATASETS and not args.no_jsonfix:
        prog = [sys.executable, os.path.join(HERE, "euroeval_jsonfix.py")]
    else:
        prog = ["euroeval"]
    return prog + [
        "--model", model_name,
        "--api-base", args.api_base,
        "--api-key", "dummy",
        "--dataset", dataset,
        "--num-iterations", str(args.num_iterations),
        "--verbose",
        "--force",  # without it EuroEval returns a cached record instead of rerunning
        "--cache-dir", args.cache_dir,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8003/v1")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--cache-dir", default=os.path.join(HERE, ".euroeval_cache"))
    ap.add_argument("--num-iterations", type=int, default=10)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--timeout", type=int, default=8 * 3600, help="per cell, seconds")
    ap.add_argument("--no-jsonfix", action="store_true",
                    help="run zebra with EuroEval's default json_object request")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results_file = os.path.join(args.out, "euroeval_benchmark_results.jsonl")
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    done = done_cells(results_file, args.num_iterations)
    env = dict(os.environ, OPENAI_API_KEY="dummy")

    # Pair-adjacent order: each safe model directly followed by its abliterated version.
    for dataset in args.datasets.split(","):
        for model_id in args.models.split(","):
            model_name = MODELS[model_id]
            cmd = command(model_name, dataset, args)
            if args.dry_run:
                print(" ".join(cmd))
                continue
            if (model_name, dataset) in done:
                print(f"skip {model_id} x {dataset} (done)")
                continue
            print(f"run  {model_id} x {dataset} (n={args.num_iterations})", flush=True)
            t0 = time.time()
            log_path = os.path.join(log_dir, f"{model_id}__{dataset}__n{args.num_iterations}.log")
            with open(log_path, "w") as log:
                try:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        cwd=args.out, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            print(f"     exit {rc}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
