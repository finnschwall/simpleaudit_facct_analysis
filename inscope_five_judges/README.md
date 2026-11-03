# The in-scope pack under five judge configurations

The 47 in-scope scenarios (`hei_refusal_v1`: 29 answer, 18 decline) were run single-turn against the chatbot itself: its retrieval pipeline, with GPT-4 generating, as configured on 26 October 2026, after both audited releases. Each judge configuration scored its own run, and all five configurations used the same judge model (`Qwen3-30B-A3B-Instruct-2507`). Reply lengths correlate at 0.84–0.92 across the five runs.

This run is reported for what it shows about the judges, not as a finding about the chatbot.

## Result

Spearman correlation between reply length and verdict, by the scenario's expected outcome (95% bootstrap CI):

| Judge configuration | Score | Answer-marked (n=29) | Decline-marked (n=18) |
|---|---|---|---|
| Safety rubric | 76.6 | −0.50 [−0.71, −0.23] | −0.60 [−0.88, −0.15] |
| Domain judge (`helsedir_sexhealth_no`) | 66.0 | −0.20 [−0.56, 0.21] | −0.77 [−0.87, −0.50] |
| Domain judge, retrieval (`helsedir_sexhealth_no_rag`) | 55.9 | −0.24 [−0.58, 0.14] | −0.69 [−0.90, −0.26] |
| Abstention (AbstentionBench) | 53.7 | −0.56 [−0.80, −0.16] | −0.70 [−0.85, −0.40] |
| Abstention (binary; the expected outcome decides) | 84.6 | +0.44 [0.32, 0.66] | −0.49 [−0.79, −0.03] |

- **The safety rubric still prefers shorter replies among full answers.** On answer-marked replies of at least 300 words its correlation is −0.49 [−0.72, −0.17]; the domain judges' are −0.29 and −0.31, with intervals that include zero.
- **The difference between the safety rubric and the domain judge is not established.** Their intervals overlap, and this is one run of 47 scenarios.

## Files

- `data/simulachat_<judge>.json`: one run per judge configuration.
- `data/comparison.csv`: per-scenario verdicts for the safety and domain judges.
- `helsedir_check.py`: the analysis. Run it from this folder with `python helsedir_check.py data`.
