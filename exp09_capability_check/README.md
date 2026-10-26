# Capability check: safe vs abliterated targets (EuroEval)

SimpleAudit uses abliterated models as known-unsafe targets. This experiment checks that
abliteration did not also degrade general capability. If it had, judges could flag those
targets for low quality, not for unsafe content.

**Result.** Over 3 model sizes × 7 benchmarks, the abliterated models score on average
0.4pp below their safe counterparts (95% CI −1.0 to +0.2; per-benchmark range −3.0 to
+2.4pp; 15 of 21 differences negative). The gap between the smallest and largest model
is 5–16pp on the same benchmarks (zebra aside). Full tables: [results/results.md](results/results.md).

## Setup

**Models** (GGUF, Q6_K; the same targets as in the SimpleAudit experiments):

| size | safe | abliterated |
|---|---|---|
| XS | Qwen3.5-4B-Q6_K | Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q6_K |
| S | Qwen3.5-9B-Q6_K | Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K |
| M | Qwen3.5-35B-A3B-Q6_K | Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K |

**Server.** llama.cpp `llama-server` (build b9813) in router mode, one model loaded at a
time, on 4× NVIDIA RTX 2080 Ti:

```bash
llama-server --models-dir <dir with the six .gguf files> --port 8003 --ctx-size 50000 \
    --models-max 1 --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
    --presence-penalty 1.5 --repeat-penalty 1.0 --reasoning off
```

EuroEval overrides the sampling temperature: every request is sent with temperature 0
and seed 4242.

**Harness.** [EuroEval](https://euroeval.com) 17.7.0 (LiteLLM 1.95.0), one CLI call per
(model, dataset) cell, `--num-iterations 10`. Each iteration is a seeded bootstrap
resample of the evaluation items, identical for every model, so each safe/abliterated
pair is compared on the same items. A cell's score is the mean over its 10 iterations.

**Datasets.**

| dataset | language | skill | metric | items | few-shot |
|---|---|---|---|---|---|
| nrk-quiz-qa | Norwegian | knowledge (Norway quiz) | accuracy | 256 | 5 |
| mmlu-no | Norwegian | knowledge | accuracy | 256 | 5 |
| multi-ifeval-nb | Norwegian Bokmål | instruction following | instruction accuracy | 524 | 0 |
| norquad | Norwegian | reading comprehension | F1 | 256 | 2 |
| mmlu | English | knowledge | accuracy | 256 | 5 |
| arc | English | knowledge (science QA) | accuracy | 256 | 5 |
| zebra-puzzles-hard-en | English | logical reasoning | cell-wise accuracy | 128 | 8 |

EuroEval's defaults decide the split and size: the validation split of the "mini"
versions, except multi-ifeval-nb, which has no validation split and uses its test split.
EuroEval still names the metrics `test_*` in its output; each record carries
`"validation_split": "true"`. Absolute scores are therefore not comparable with the
EuroEval leaderboard; the comparison between conditions is unaffected.

Not included: `european-values` (EuroEval aborts the run when a model gives an
unparseable answer, so refusals would decide the result) and GSM8K (not in EuroEval;
zebra-puzzles-hard-en is the closest reasoning task).

## zebra-puzzles-hard-en: enforced JSON output

EuroEval requests `response_format: json_object` for zebra. llama.cpp b9813 ignores
that type and only enforces `json_schema`, so models may answer in prose. With the
256-token output cap, a model that reasons in prose is cut off before its JSON answer
and scores 0 on that puzzle. This hit the M models hardest (m_abl: 85 of 87 unique
puzzles answered in prose; m_safe: 34 of 87), producing a spurious 20pp gap (M safe
21.2, abliterated 0.9).

[euroeval_jsonfix.py](euroeval_jsonfix.py) runs the EuroEval CLI with a JSON schema set
for the logic-puzzle task (`{"object_X": [attributes]}`), so EuroEval sends
`json_schema` and the server enforces it. Nothing else changes. All zebra results in
`data/euroeval_results.jsonl` come from this version; the unpatched ones are kept in
`data/euroeval_results_zebra_unpatched.jsonl` for comparison.

Even with the fix, zebra says little: no model solved a whole puzzle, every model gets
27–31% of cells right (a random assignment gets about 25%, with 4 houses per puzzle),
and size barely changes the score.

## Files

| file | contents |
|---|---|
| `data/euroeval_results.jsonl` | EuroEval's raw output, one record per cell (42), including all 10 per-iteration scores. zebra records are from the JSON-enforced run. |
| `data/euroeval_results_zebra_unpatched.jsonl` | the 6 zebra records from the run without enforced JSON |
| `results/results.md` | all tables (per-pair scores and differences, pooled difference, size gaps, per-pair noise, zebra comparison) |
| `results/scores_per_iteration.csv` | flat table: size, condition, model, dataset, metric, iteration, score |
| `run_sweep.py` | runs the 42 cells |
| `euroeval_jsonfix.py` | EuroEval CLI with enforced JSON output for zebra |
| `analyze.py` | builds `results/` from a EuroEval results file |

## Reproduce

Needs the six GGUF files served as above and a Python environment with
`euroeval==17.7.0` (plus numpy and scipy for the analysis).

```bash
python run_sweep.py --dry-run        # list the 42 EuroEval commands
python run_sweep.py                  # run them -> out/euroeval_benchmark_results.jsonl, out/logs/
python analyze.py --data out/euroeval_benchmark_results.jsonl --out out/results
```

`run_sweep.py` skips cells that already have a result, so an interrupted sweep resumes
when rerun. Options: `--api-base` (default `http://localhost:8003/v1`), `--models`,
`--datasets`, `--num-iterations`, `--no-jsonfix` (zebra with EuroEval's default
request). `python analyze.py` without arguments regenerates `results/` from the shipped
data.

Runtime on the hardware above: about 19 h for all 42 cells; multi-ifeval-nb cells are the
slowest (32–49 min each).

**Expect small differences.** llama.cpp is not bit-reproducible at temperature 0: three
identical single-iteration runs of one cell (S safe, nrk-quiz-qa) scored 60.9, 63.3 and
63.7. Single cell scores will therefore differ somewhat from ours; compare the pooled
difference and its CI, not individual cells.

## Limitations

- Single-turn benchmarks: knowledge, reading, instruction following and short
  reasoning. Coherence over the long multi-turn conversations SimpleAudit runs is not
  tested directly; multi-ifeval-nb is the closest proxy.
- Refusals were not counted (EuroEval does not store model outputs by default). A safe
  model refusing benign items would lower its own score, so the abliterated model would
  look better. The observed differences tilt the other way.
- 256 items per dataset (128 for zebra). More iterations would not reduce the
  uncertainty, since they resample the same items; only larger item sets would.
