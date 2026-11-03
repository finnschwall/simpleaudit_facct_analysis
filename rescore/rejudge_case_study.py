"""Re-grade the stored §7 case-study transcripts with another SimpleAudit judge config.

Uses SimpleAudit's own `rejudge` (transcripts unchanged, only the judge differs).
Output mirrors the input layout:  <out>/<judge_cfg>__<judge_model>/<pack>/<model>/run_k.json

Usage
  python rejudge_case_study.py --src <procurement_case_study/judgeBy_..._split_by_scenario> \
      --out rejudged --judge binary_abstention --provider anthropic --model claude-haiku-5-5 \
      [--runs 0-9] [--models borealis-27b,gemma-3-27b] [--concurrency 16] [--limit-files N]

  For your own vLLM server:  --provider vllm --base-url http://host:port/v1 --model Qwen/...
API key: --key-file, or the provider's usual env var.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from simpleaudit.reframing import make_judge_client, rejudge_async
from simpleaudit.results import AuditResults


def parse_runs(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(x) for x in spec.split(",")}


def collect(src, runs, model_filter):
    files = []
    for f in sorted(Path(src).glob("*/*/run_*.json")):
        model = f.parent.name
        if "270m" in model or not re.search(r"(borealis|gemma)", model):
            continue
        if model_filter and not any(m in model for m in model_filter):
            continue
        if int(re.search(r"run_(\d+)", f.name).group(1)) in runs:
            files.append(f)
    return files


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="binary_abstention")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--runs", default="0-9")
    ap.add_argument("--models", default="")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--files-in-flight", type=int, default=4)
    ap.add_argument("--limit-files", type=int, default=0)
    a = ap.parse_args()

    key = Path(a.key_file).read_text().strip() if a.key_file else None
    client = make_judge_client(a.provider, api_key=key, base_url=a.base_url)
    files = collect(a.src, parse_runs(a.runs), [m for m in a.models.split(",") if m])
    if a.limit_files:
        files = files[: a.limit_files]
    tag = f"{a.judge}__{a.model.replace('/', '_')}"
    out_root = Path(a.out) / tag
    todo = [f for f in files if not (out_root / f.parent.parent.name / f.parent.name / f.name).exists()]
    print(f"{len(files)} files selected, {len(files) - len(todo)} already done, {len(todo)} to go -> {out_root}", flush=True)

    file_sem = asyncio.Semaphore(a.files_in_flight)
    stats = {"done": 0, "tin": 0, "tout": 0, "err": 0}
    t0 = time.time()

    async def one(f):
        async with file_sem:
            res = await rejudge_async(str(f), client, a.model, judge=a.judge,
                                      max_concurrency=a.concurrency, max_retries=4, retry_backoff=2.0)
            dest = out_root / f.parent.parent.name / f.parent.name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = res.to_dict()
            payload["rejudge_meta"] = {"source": str(f), "judge": a.judge, "judge_model": a.model,
                                       "provider": a.provider}
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(dest)
            for r in res.results:
                stats["tin"] += r.judge_input_tokens or 0
                stats["tout"] += r.judge_output_tokens or 0
                stats["err"] += r.severity == "ERROR"
            stats["done"] += 1
            if stats["done"] % 10 == 0 or stats["done"] == len(todo):
                print(f"  {stats['done']}/{len(todo)} files  in={stats['tin']:,} out={stats['tout']:,} "
                      f"errors={stats['err']}  {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(f) for f in todo))
    print(f"DONE tokens in={stats['tin']:,} out={stats['tout']:,} errors={stats['err']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
