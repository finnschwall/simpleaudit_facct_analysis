# Length check on the Borealis–Gemma case study

Does reply length account for the Borealis–Gemma score gap reported under the safety rubric (judge and auditor Qwen3.5-122B-A10B-Q4_K_S, 36 scenarios, 10 reruns)? The reproduction matches the paper: Borealis scores 4.6/27.9/42.3/43.7, the deltas are +0.8/+14.2/+7.0/+6.0, and the 27B critical rates are 15.3% vs 22.8%.

## Results (paper configuration)

| | 1B | 4B | 12B | 27B |
|---|---|---|---|---|
| Target words per conversation, Borealis / Gemma | 1195 / 2228 | 817 / 2325 | 754 / 2503 | 742 / 2339 |
| Borealis − Gemma, as reported [95% CI] | +0.8 [−0.5, 2.5] | +14.2 [8.7, 21.7] | +7.0 [1.5, 11.7] | +6.0 [0.7, 11.0] |
| Same, length-matched* [95% CI] | −2.2 [−4.2, 0.2] | +3.7 [−2.4, 10.4] | +1.3 [−6.7, 9.8] | −3.0 [−12.9, 7.8] |

\*Five length bins per scenario and size, keeping only bins that hold both families (about 22% of runs). CIs come from a bootstrap over scenarios.

- Borealis is shorter in all 36 scenarios. Across scenarios, its lead grows with its length deficit (r = −0.41).
- Within a model and scenario, longer reruns score lower: −12.5 points per unit of log length (SE 2.3).
- Length does not explain the size trend: Gemma's length is flat while its score rises from 3.8 to 37.6.
- In the earlier run judged by Qwen3-30B-A3B-Instruct-2507, Gemma is ahead by 2–6 points at every size (not significant), and length is unrelated to score.

## Files

`build_dataset.py` builds the per-run table (`runs.csv`), `regressions.py` runs the analysis, and `make_figure.py` draws `length_check.png`.

## Reproduce (from this folder)

```
python build_dataset.py ../procurement_case_study runs.csv
python regressions.py runs.csv                                  # paper configuration
python regressions.py runs.csv qwen3-30b-a3b-instruct-2507      # earlier configuration
python make_figure.py runs.csv length_check.png
```
