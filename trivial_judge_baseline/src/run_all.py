#!/usr/bin/env python3
"""Single entry point: regenerates every output of the trivial-judge baseline from the raw
records. Deterministic (all randomness from the config seed).

    python trivial_judge_baseline/src/run_all.py [--no-keyword]

Without the word list (or with --no-keyword) only the parts that do not need it run:
data checks, length judge, LLM-judge AUROC reproduction and Q3 for the LLM judges.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyses import (Q2, agreement, case_deltas, check_scenarios, ladder_frame, q1_counts,  # noqa: E402
                      q1_llm, q1_trivial, q3_frame, setting_frame, sign_summary,
                      trivial_run_scores)
from data import (CFG, JUDGES, TJB, load_case, load_mapping_module, load_validation,  # noqa: E402
                  load_wordlist, target_turns)
from judges import last, length_features, phrase_strings, share, turn_matches  # noqa: E402
from stats import partial_eta2  # noqa: E402

RES = TJB / "results"
KW_VARIANTS = ["keyword", "keyword_en_only", "keyword_prefix", "keyword_last_turn"]


def add_length(df, conversations):
    feats = [length_features(conversations[c]) for c in df.conv_id]
    df["n_target_turns"] = [f[0] for f in feats]
    df["mean_chars"] = [f[1] for f in feats]
    df["length"] = [f[2] for f in feats]


def keyword_features(df, conversations, entries):
    """Adds the primary keyword column and the variants; returns per-conversation per-turn
    substring match sets and the leave-one-phrase-out bookkeeping."""
    strings, owners = phrase_strings(entries, ("en", "nb"))
    sidx = {s: i for i, s in enumerate(strings)}
    en_set = frozenset(sidx[e["en"]] for e in entries)
    matches = {}
    cols = {c: [] for c in KW_VARIANTS}
    for cid in df.conv_id:
        turns = target_turns(conversations[cid])
        m = [turn_matches(t, strings, "substring") for t in turns]
        p = [turn_matches(t, strings, "prefix") for t in turns]
        matches[cid] = m
        cols["keyword"].append(share([bool(x) for x in m]))
        cols["keyword_en_only"].append(share([bool(x & en_set) for x in m]))
        cols["keyword_prefix"].append(share([bool(x) for x in p]))
        cols["keyword_last_turn"].append(last([bool(x) for x in m]))
    for c, v in cols.items():
        df[c] = v
    # leave-one-phrase-out: dropping entry k removes the strings owned only by entry k
    removed = {k: frozenset(i for i, o in enumerate(owners) if o == {k}) for k in range(len(entries))}
    return matches, removed, strings, en_set


def loo_column(df, matches, removed_k):
    return [share([bool(x - removed_k) for x in matches[c]]) for c in df.conv_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-keyword", action="store_true", help="run only the parts that do not need the word list")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # ------------------------------------------------------------------ data
    val, llm_runs, checks, conv_val = load_validation()
    case, conv_case = load_case()
    conversations = {**conv_val, **conv_case}
    for df in (val, case):
        add_length(df, conversations)
    n_cells = val.cell.nunique()
    say("== Data ==")
    say(f"validation cells used: {n_cells}; conversations (each once): {len(val)}; "
        f"runs: {val.cluster.nunique()}")
    used = [c for c in checks if c["n_runs"] is not None]
    skipped = [f"{c['cell']}/J={c['judge']}" for c in checks if c["n_runs"] is None]
    say(f"transcript-equality check (SHA-256 of scenario name, description and conversation, per run x "
        f"scenario position): {sum(bool(c['identical_to_first_judge']) for c in used)}/{len(used)} judge "
        f"files identical to the first judge file of their cell (an assertion stops the run otherwise); "
        f"judge dirs skipped for lacking experiment_results.json: {skipped}")
    say(f"LLM run records used (cell x judge x run): {len(llm_runs)}; ERROR verdicts in them: "
        f"{int(llm_runs.n_error.sum())}")
    say(f"case-study conversations: {len(case)} ({case.cluster.nunique()} model-runs, "
        f"{case.pack.nunique()} packs); ERROR verdicts: {int((case.sev_caseL == 'ERROR').sum())}")
    zero_val = int((val.n_target_turns == 0).sum())
    zero_case = int((case.n_target_turns == 0).sum())
    say(f"conversations with zero target turns (score NaN, excluded): validation {zero_val}, case {zero_case}")
    (RES / "data_checks.json").write_text(json.dumps(
        {"transcript_equality": checks, "zero_target_turns": {"validation": zero_val, "case": zero_case}},
        indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ keyword judge
    entries, wl_path = load_wordlist()
    use_kw = entries is not None and not args.no_keyword
    inst = ["length"] + (KW_VARIANTS if use_kw else [])
    if use_kw:
        m_val, removed, strings, en_set = keyword_features(val, conversations, entries)
        m_case, _, _, _ = keyword_features(case, conversations, entries)
        say(f"word list: {len(entries)} entries, {len(strings)} distinct strings (en+nb), "
            f"{len(en_set)} distinct en strings")
    else:
        for df in (val, case):
            for c in KW_VARIANTS:
                df[c] = np.nan
        say(f"word list not used ({'--no-keyword' if entries is not None else 'missing: ' + str(wl_path)}); "
            "keyword parts skipped")

    # conversation_scores.csv
    out_cols = (["conv_id", "dataset", "cell_or_model", "pack", "target", "auditor", "run", "scenario",
                 "n_target_turns", "mean_chars", "length"] + KW_VARIANTS
                + [f"sev_{j}" for j in JUDGES] + ["sev_caseL"])
    cs = pd.concat([val.assign(cell_or_model=val.cell, pack=None),
                    case.assign(cell_or_model=case.model, target=None, auditor=None)], ignore_index=True)
    cs.reindex(columns=out_cols).to_csv(RES / "conversation_scores.csv", index=False)

    vruns = trivial_run_scores(val, inst)

    # ------------------------------------------------------------------ Q1
    mod = load_mapping_module()
    say("\n== Q1: AUROC safe vs abliterated (runs) ==")
    q1l = q1_llm(llm_runs, mod.PUBLISHED_AUROC)
    n_rep = int(q1l.reproduces_published.sum())
    say(f"CHECK LLM-judge AUROC vs published grid: {n_rep}/{len(q1l)} cells reproduce at 3 decimals")
    jl_lower = {(r.auditor, r.target_size): r.ci_lo for r in q1l[q1l.instrument == "J=L"].itertuples()}
    q1_frames = [q1l]
    q1_count = {}
    for c in inst:
        q = q1_trivial(vruns, c, jl_lower)
        q1_frames.append(q)
        q1_count[c] = q1_counts(q)
    q1all = pd.concat(q1_frames, ignore_index=True)
    q1all.to_csv(RES / "q1_auroc.csv", index=False)
    for r in q1all.itertuples():
        extra = f" match={r.match} fail={r.fail}" if isinstance(r.match, (bool, np.bool_)) else \
            f" published={r.published_auroc}"
        say(f"  {r.instrument:<18} A={r.auditor:<2} T={r.target_size:<2} AUROC={r.auroc:.3f} "
            f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]{extra}")
    for c in inst:
        k = q1_count[c]
        say(f"  {c}: match {k['n_match']}/6, fail {k['n_fail']}/6")

    # ------------------------------------------------------------------ Q2
    say("\n== Q2: pair agreement with reference ==")
    q2rows = []
    frames = {}
    for name, s in Q2["settings"].items():
        d = setting_frame(name, val, case)
        check_scenarios(name, d)
        frames[name] = d
        for c in (["length", "keyword", "keyword_en_only"] if use_kw else ["length"]):
            r = agreement(d, c)
            q2rows.append({"setting": name, "instrument": c, "reference": s["reference"], **r})
    q2 = pd.DataFrame(q2rows)
    q2_cols = ["setting", "instrument", "reference", "n_pairs", "agreement", "ci_lo", "ci_hi",
               "agreement_no_ties", "tie_rate", "n_conversations", "n_dropped_nan", "n_clusters",
               "n_agree", "n_tie", "n_disagree"]
    q2[q2_cols].to_csv(RES / "q2_agreement.csv", index=False)
    for r in q2.itertuples():
        say(f"  {r.setting:<17} {r.instrument:<16} ref={r.reference:<12} pairs={r.n_pairs:>6} "
            f"agree={r.agreement:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] no-ties={r.agreement_no_ties:.3f} "
            f"ties={r.tie_rate:.3f}")

    say("\n== Q2 ladder (targets S, M; A in M, L, XL; reference J=XL) ==")
    lad = ladder_frame(val)
    # primary: pairs dropped only if the reference or this instrument has an ERROR verdict.
    # secondary: common set, conversations where none of the five judges has an ERROR verdict.
    jcols = [f"J={j}" for j in JUDGES]
    common = lad[lad[jcols].notna().all(axis=1)]
    lrows = []
    for pair_set, frame in (("per_instrument", lad), ("common_no_error", common)):
        say(f" pair set {pair_set}: {len(frame)} conversations")
        for c in Q2["ladder"]["instruments"]:
            if c.startswith("keyword") and not use_kw:
                continue
            r = agreement(frame, c)
            lrows.append({"instrument": c, "pair_set": pair_set, "reference": Q2["ladder"]["reference"], **r})
            say(f"  {c:<16} pairs={r['n_pairs']:>6} agree={r['agreement']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
                f"no-ties={r['agreement_no_ties']:.3f} ties={r['tie_rate']:.3f}")
    pd.DataFrame(lrows)[["instrument", "pair_set", "reference"] + q2_cols[3:]].to_csv(
        RES / "q2_ladder.csv", index=False)

    # ------------------------------------------------------------------ Q3
    say("\n== Q3: partial eta^2, score ~ C(target) + C(auditor), Type II ==")
    q3rows = []
    v3 = q3_frame(vruns)
    l3 = q3_frame(llm_runs)
    for c in CFG["q3"]["instruments"]:
        if c.startswith("J="):
            r = partial_eta2(l3[l3.judge == c[2:]], "score")
        elif c in inst:
            r = partial_eta2(v3, c)
        else:
            continue
        q3rows.append({"instrument": c, **r})
        say(f"  {c:<10} n={r['n_runs']} eta2_target={r['eta2_target']:.3f} eta2_auditor={r['eta2_auditor']:.3f}")
    pd.DataFrame(q3rows).to_csv(RES / "q3_eta2.csv", index=False)

    # ------------------------------------------------------------------ case study
    say("\n== Case study: Borealis - Gemma deltas ==")
    case_m = case[~case.model.isin(CFG["case_study"]["excluded_models"])]
    deltas, n_ok = case_deltas(case_m, inst, mod)
    say(f"CHECK Judge-L deltas vs published table: {n_ok}/20 reproduce at one decimal")
    deltas.to_csv(RES / "case_study_deltas.csv", index=False)
    signs = sign_summary(deltas, inst)
    (RES / "case_study_sign_agreement.json").write_text(json.dumps(signs, indent=1), encoding="utf-8")
    for c, s in signs.items():
        say(f"  {c}: sign match {s['n_sign_match']}/{s['n_differences']}; in CI-excludes-0 subset "
            f"{s['n_sign_match_in_ci_subset']}/{s['n_llm_ci_excludes_0']}")

    if use_kw:
        robustness(val, case, jl_lower, entries, m_val, m_case, removed, say)

    (RES / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- keyword robustness

def robustness(val, case, jl_lower, entries, m_val, m_case, removed, say):
    say("\n== Robustness of the keyword judge ==")
    rob_settings = ["S2", "S2-s", "S3-safety"]
    rows = []

    def evaluate(variant, vdf, cdf, col, with_ci):
        runs = trivial_run_scores(vdf, [col])
        q = q1_trivial(runs, col, jl_lower, with_ci=True, label=variant)
        row = {"variant": variant, **{f"q1_{k}": v for k, v in q1_counts(q).items()
                                      if k in ("n_match", "n_fail")}}
        for name in rob_settings:
            d = setting_frame(name, vdf, cdf)
            r = agreement(d, col, with_ci)
            row[f"{name}_agreement"] = r["agreement"]
            row[f"{name}_ci_lo"], row[f"{name}_ci_hi"] = r["ci_lo"], r["ci_hi"]
        e = partial_eta2(q3_frame(runs), col)
        row["eta2_target"], row["eta2_auditor"] = e["eta2_target"], e["eta2_auditor"]
        return row, q

    primary, _ = evaluate("primary", val, case, "keyword", True)
    rows.append(primary)
    for v in ["keyword_en_only", "keyword_prefix", "keyword_last_turn"]:
        row, q = evaluate(v, val, case, v, True)
        rows.append(row)
    loo = []
    for k, rem in removed.items():
        vdf = val.assign(_loo=loo_column(val, m_val, rem))
        cdf = case.assign(_loo=loo_column(case, m_case, rem))
        row, _ = evaluate(f"leave_out_{k}", vdf, cdf, "_loo", False)
        row["dropped_entry"] = k
        row["dropped_en"], row["dropped_nb"] = entries[k]["en"], entries[k]["nb"]
        row["n_strings_removed"] = len(rem)
        loo.append(row)
        rows.append(row)
    rob = pd.DataFrame(rows)
    rob.to_csv(RES / "robustness.csv", index=False)
    loo = pd.DataFrame(loo)
    rng = {c: [float(loo[c].min()), float(loo[c].max())]
           for c in ["q1_n_match", "q1_n_fail", "S2_agreement", "S2-s_agreement", "S3-safety_agreement",
                     "eta2_target", "eta2_auditor"]}
    for r in rows[:4]:
        say(f"  {r['variant']:<18} Q1 match {r['q1_n_match']}/6 fail {r['q1_n_fail']}/6; "
            f"S2 {r['S2_agreement']:.3f} [{r['S2_ci_lo']:.3f}, {r['S2_ci_hi']:.3f}]; "
            f"S2-s {r['S2-s_agreement']:.3f}; S3-safety {r['S3-safety_agreement']:.3f}; "
            f"eta2 target {r['eta2_target']:.3f} auditor {r['eta2_auditor']:.3f}")
    say(f"  leave-one-phrase-out ({len(loo)} entries), min-max: " +
        "; ".join(f"{c} {a:.3f}-{b:.3f}" for c, (a, b) in rng.items()))


if __name__ == "__main__":
    main()
