"""Q1, Q2, Q3 and the case-study re-score (see ../README.md).

All functions take an instrument column name so the robustness variants reuse them.
Conversation-level instrument columns hold 'higher = safer' scores (NaN = excluded).
"""

import numpy as np
import pandas as pd

from data import CFG, JUDGES, ROOT, safety_value
from stats import PairAgreement, auroc, auroc_ci, diff_means_ci, partial_eta2

Q1 = CFG["q1"]
Q2 = CFG["q2"]
SPLIT = CFG["scenario_split"]


# --------------------------------------------------------------------------- run scores

def trivial_run_scores(conv, cols):
    """Run score = mean of conversation scores over the run's scenarios (NaN conversations
    excluded from the mean). One row per validation run (cluster)."""
    keys = ["cluster", "cell", "target", "auditor", "target_safe", "target_size", "run"]
    return conv.groupby(keys, sort=True)[cols].mean().reset_index()


# --------------------------------------------------------------------------- Q1

def q1_cells():
    return [(a, t) for a in Q1["auditors"] for t in Q1["target_sizes"]]


def q1_llm(llm_runs, published):
    """AUROC + CI for the LLM judges from stored run scores; reproduction check."""
    rows = []
    for j in Q1["llm_judges"]:
        for a, t in q1_cells():
            r = llm_runs[(llm_runs.judge == j) & (llm_runs.auditor == a)]
            safe = r[r.target == t].sort_values("run")["score"].to_numpy()
            uns = r[r.target == t + "_UNSAFE"].sort_values("run")["score"].to_numpy()
            p, lo, hi = auroc_ci(safe, uns)
            pub = published.get((j, a, t))
            rows.append({"instrument": f"J={j}", "auditor": a, "target_size": t,
                         "n_safe": len(safe), "n_unsafe": len(uns),
                         "auroc": p, "ci_lo": lo, "ci_hi": hi,
                         "published_auroc": pub,
                         "reproduces_published": (pub is not None and round(p, 3) == pub)})
    return pd.DataFrame(rows)


def q1_trivial(runs, inst_col, jl_lower, with_ci=True, label=None):
    """AUROC per cell for a trivial instrument, with match/fail per cell.
    jl_lower: {(auditor, size): lower CI bound of the J=L AUROC}."""
    rows = []
    for a, t in q1_cells():
        r = runs[runs.auditor == a]
        safe = r[r.target == t].sort_values("run")[inst_col].to_numpy()
        uns = r[r.target == t + "_UNSAFE"].sort_values("run")[inst_col].to_numpy()
        if np.isnan(safe).any() or np.isnan(uns).any():
            raise ValueError(f"NaN run score for {inst_col} in cell A={a} T={t}")
        if with_ci:
            p, lo, hi = auroc_ci(safe, uns)
        else:
            p, lo, hi = auroc(safe, uns), float("nan"), float("nan")
        rows.append({"instrument": label or inst_col, "auditor": a, "target_size": t,
                     "n_safe": len(safe), "n_unsafe": len(uns),
                     "auroc": p, "ci_lo": lo, "ci_hi": hi,
                     "jl_ci_lo": jl_lower[(a, t)],
                     "match": bool(p >= jl_lower[(a, t)]),
                     "fail": bool(lo <= 0.5) if with_ci else None})
    return pd.DataFrame(rows)


def q1_counts(df):
    return {"n_cells": len(df), "n_match": int(df["match"].sum()),
            "n_fail": int(df["fail"].sum()) if df["fail"].notna().all() else None}


# --------------------------------------------------------------------------- Q2

def setting_frame(name, val, case):
    """Conversations and reference column (higher = safer) for one Q2 setting."""
    s = Q2["settings"][name]
    if s["data"] == "validation":
        d = val[val.auditor.isin(s["auditors"]) & val.target.isin(s["targets"])].copy()
        if s["scenarios"] != "all":
            d = d[d.scenario.isin(SPLIT[s["scenarios"]])]
        d["ref"] = d[f"sev_{s['reference']}"].map(safety_value)
        if "judge_files" in s:   # S1: J=L files; conversations are identical across judge files
            if d[f"sev_{s['judge_files']}"].isna().any():
                raise ValueError(f"{name}: missing J={s['judge_files']} verdicts")
    else:
        d = case[case.pack == s["pack"]].copy()
        d["ref"] = d["sev_caseL"].map(safety_value)
    return d


def check_scenarios(name, d):
    s = Q2["settings"][name]
    if s["data"] == "validation" and s["scenarios"] != "all":
        assert set(d.scenario) == set(SPLIT[s["scenarios"]]), (name, set(d.scenario))


def agreement(d, inst_col, with_ci=True):
    return PairAgreement(d, "ref", inst_col).full(with_ci)


def ladder_frame(val):
    L = Q2["ladder"]
    d = val[val.target.isin(L["targets"]) & val.auditor.isin(L["auditors"])].copy()
    d["ref"] = d[f"sev_{L['reference']}"].map(safety_value)
    for j in JUDGES:
        if d[f"sev_{j}"].isna().any():
            raise ValueError(f"ladder: J={j} missing on some conversations")
        d[f"J={j}"] = d[f"sev_{j}"].map(safety_value)
    return d


# --------------------------------------------------------------------------- Q3

def q3_frame(val_runs):
    s = CFG["q3"]
    return val_runs[val_runs.auditor.isin(s["auditors"]) & val_runs.target.isin(s["targets"])]


# --------------------------------------------------------------------------- case study

def case_llm(mod):
    """Judge-L run scores per (model, column) via the mapping script (linear, ERROR = 0),
    plus the script's own deltas and the published-table check."""
    root = ROOT / CFG["paths"]["case_records"]
    runs, pairs, _ = mod.load_case(root, "zero")
    res = mod.case_analysis(runs, pairs)
    per_run = {}
    for size in CFG["case_study"]["sizes"]:
        for fam in ("borealis", "gemma"):
            model = pairs[size][fam]
            for col in CFG["case_study"]["columns"]:
                per_run[(model, col)] = [mod.score(u, "linear") for u in mod.units(runs[model], col)]
    n_ok = 0
    for size, rows in mod.PUBLISHED_TABLE10.items():
        for col, (pub_d, _pub_cr) in zip(mod.COLUMNS, rows):
            n_ok += mod.r1(res["delta"][(size, col, "linear")]) == pub_d
    return pairs, per_run, res, n_ok


def case_trivial_run_scores(case, inst_col, model, column):
    """Per-run trivial score for one model and column: mean of conversation scores over the
    column's scenarios ('overall' pools all four packs, i.e. 36 scenarios)."""
    d = case[case.model == model]
    if column != "overall":
        d = d[d.pack == column]
    g = d.groupby("run")
    if column == "overall":
        packs = g["pack"].nunique()
        if (packs != 4).any():
            raise ValueError(f"{model}: run without all four packs")
    return g[inst_col].mean().sort_index().to_numpy()


def case_deltas(case, inst_cols, mod):
    pairs, llm_runs, res, n_ok = case_llm(mod)
    rows = []
    for size in CFG["case_study"]["sizes"]:
        b, g = pairs[size]["borealis"], pairs[size]["gemma"]
        for col in CFG["case_study"]["columns"]:
            d, lo, hi = diff_means_ci(llm_runs[(b, col)], llm_runs[(g, col)])
            assert abs(d - res["delta"][(size, col, "linear")]) < 1e-9
            row = {"size": size, "column": col, "borealis": b, "gemma": g,
                   "n_runs_borealis": len(llm_runs[(b, col)]), "n_runs_gemma": len(llm_runs[(g, col)]),
                   "delta_llm_L": d, "llm_ci_lo": lo, "llm_ci_hi": hi,
                   "llm_ci_excludes_0": bool(lo > 0 or hi < 0)}
            for c in inst_cols:
                tb = case_trivial_run_scores(case, c, b, col)
                tg = case_trivial_run_scores(case, c, g, col)
                row[f"delta_{c}"] = float(np.mean(tb) - np.mean(tg))
                row[f"sign_match_{c}"] = bool(np.sign(row[f"delta_{c}"]) == np.sign(d))
            rows.append(row)
    return pd.DataFrame(rows), n_ok


def sign_summary(deltas, inst_cols):
    sub = deltas[deltas.llm_ci_excludes_0]
    return {c: {"n_differences": len(deltas), "n_sign_match": int(deltas[f"sign_match_{c}"].sum()),
                "n_llm_ci_excludes_0": len(sub),
                "n_sign_match_in_ci_subset": int(sub[f"sign_match_{c}"].sum())}
            for c in inst_cols}
