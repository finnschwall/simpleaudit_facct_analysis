Analysis code for

_When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels_

## Notebooks

| Notebook | Contents |
|----------|----------|
| [simpleaudit_analysis.ipynb](simpleaudit_analysis.ipynb) | Main results: variance decomposition, judge comparison, scenario profiles, safe vs. unsafe target detection |
| [stability_analysis.ipynb](stability_analysis.ipynb) | Reproducibility analysis: run-to-run variance, inter-run agreement, stability across repetitions |
| [petri_analysis_paper.ipynb](petri_analysis_paper.ipynb) | Petri baseline: analogous analyses for the Petri comparison system |
| [token_usage_simpleaudit.ipynb](token_usage_simpleaudit.ipynb) | Token cost breakdown for SimpleAudit runs + aggregate with petri|
| [token_usage_petri.ipynb](token_usage_petri.ipynb) | Token cost breakdown for Petri runs |

## Data

- `comparison_results/norwegian_safety/` — SimpleAudit experiment results (safe and unsafe targets)
- `petri_comparison_results/norwegian_safety/` and `norwegian_safety_unsafe/` — Petri experiment results

Layout within these two results directory: `T={size}[_UNSAFE]__A={size}[_UNSAFE]/J={size}[_UNSAFE]/experiment_results.json`

- `cross_pack_scenario_effects/` — data for the Appendix M cross-pack records. Layout: `sa-baseline/{pack_name}/{model_label}/run_{0..9}.json`
- `procurement_case_study/` — data for the §7 case-study records. Layout: `judgeBy_{judge_model}_split_by_scenario/{pack_name}/{model_label}/run_{0..9}.json`
- `capability_check/` — capability check of the safe/abliterated target pairs on 7 EuroEval benchmarks (3 sizes, 10 iterations): raw results, run and analysis scripts; see its [README](capability_check/README.md)
- `mapping_sensitivity/` — severity-to-score mapping sensitivity (appendix "Sensitivity to the Severity-to-Score Mapping"): re-scores the stored verdicts in `comparison_results/norwegian_safety/` and `procurement_case_study/judgeBy_Qwen3.5-122B-A10B-Q4_K_S_split_by_scenario/` under linear, quadratic, critical-veto and lexicographic mappings; standard library only, expected output included; see its [README](mapping_sensitivity/README.md)


## Code

- [analyze_data.py](analyze_data.py) — analysis library for simpleaudit (loaders, filters, plots)
- [generate_simpleaudit_data.py](generate_simpleaudit_data.py) / [generate_petri_data.py](generate_petri_data.py) — scripts that produced the raw data
- [models.py](models.py), [scenarios.py](scenarios.py) — model registry and scenario definitions
- [script_baseline.py](script_baseline.py) — standalone script that generated `cross_pack_scenario_effects/` and `procurement_case_study/`; runs each model against all scenario packs via `AuditExperiment`, then splits the combined per-model results into per-pack JSON files with a recomputed score