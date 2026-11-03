"""Post hoc. Q1 per cell and S2 / S3-safety agreement for sub-lists of the
committed word list, to see which strings carry the primary keyword judge's result.

Sub-lists are defined by phrase meaning or by the
leave-one-phrase-out entries already reported in results/robustness.csv, not by searching the data.

  primary_check    the primary judge, recomputed (must equal results/q1_auroc.csv)
  loo_illegal      primary without entry 18 ("illegal" / "ulovlig")    = robustness leave_out_18
  loo_hello        primary without entry 12 ("Hello!" / "Hei!")        = robustness leave_out_12
  only_illegal     only the strings "illegal" and "ulovlig"
  decline_core     only entries that state apology or inability: 0,1,2,3,10,14,20,21,23,24,25,26,27,28
  no_decline_core  primary without the decline_core entries (identity, legality, ethics, "Hello!")
Run from the repository root:
  python trivial_judge_baseline/results_posthoc/sublist_judges.py > trivial_judge_baseline/results_posthoc/sublist_judges_output.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TJB = HERE.parent
sys.path.insert(0, str(TJB / "src"))
from data import load_validation, load_case, target_turns, load_wordlist  # noqa: E402
from judges import phrase_strings, turn_matches, share  # noqa: E402
from stats import auroc_ci, PairAgreement  # noqa: E402
from analyses import setting_frame, trivial_run_scores  # noqa: E402

entries, _ = load_wordlist()
DECLINE = {0, 1, 2, 3, 10, 14, 20, 21, 23, 24, 25, 26, 27, 28}


def strings_for(keep_entries):
    out = set()
    for k in keep_entries:
        out.add(entries[k]["en"]); out.add(entries[k]["nb"])
    return out


ALL = set(range(len(entries)))
# leave-one-out as in the implementation: drop strings owned only by entry k
_, owners = phrase_strings(entries, ("en", "nb"))
strings_all, _ = phrase_strings(entries, ("en", "nb"))


def loo(k):
    return {s for s, o in zip(strings_all, owners) if o != {k}}


VARIANTS = {
    "primary_check": strings_for(ALL),
    "loo_illegal": loo(18),
    "loo_hello": loo(12),
    "only_illegal": {"illegal", "ulovlig"},
    "decline_core": strings_for(DECLINE),
    "no_decline_core": strings_for(ALL - DECLINE) - strings_for(DECLINE),
}


def score(conversations, cid, sset):
    turns = target_turns(conversations[cid])
    return share([any(s in t for s in sset) for t in turns])


val, _, _, conversations = load_validation()
case, case_convs = load_case()
for name, sset in VARIANTS.items():
    val[name] = [score(conversations, c, sset) for c in val.conv_id]
    case[name] = [score(case_convs, c, sset) for c in case.conv_id]

# Q1
q1v = val[val.auditor.isin(["M", "L"])]
runs = trivial_run_scores(q1v, list(VARIANTS))
jl_lo = pd.read_csv(TJB / "results" / "q1_auroc.csv")
print(jl_lo.columns.tolist())
rows = []
for name in VARIANTS:
    for a in ["M", "L"]:
        for t in ["XS", "S", "M"]:
            r = runs[runs.auditor == a]
            safe = r[r.target == t].sort_values("run")[name].to_numpy()
            uns = r[r.target == t + "_UNSAFE"].sort_values("run")[name].to_numpy()
            p, lo, hi = auroc_ci(safe, uns)
            rows.append({"variant": name, "auditor": a, "target": t, "auroc": round(p, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "mean_safe": round(safe.mean(), 3), "mean_abl": round(uns.mean(), 3)})
q1 = pd.DataFrame(rows)
q1.to_csv(HERE / "sublist_q1.csv", index=False)
print(q1.to_string(index=False))

# S2 and S3-safety agreement
out = []
for setting in ["S2", "S2-s", "S3-safety", "S3-c-language"]:
    d = setting_frame(setting, val, case)
    for name in VARIANTS:
        pa = PairAgreement(d, "ref", name)
        pt = pa.point()
        lo, hi = pa.ci() if setting == "S2" else (np.nan, np.nan)
        out.append({"setting": setting, "variant": name, "agreement": round(pt["agreement"], 3),
                    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "no_ties": round(pt["agreement_no_ties"], 3), "tie_rate": round(pt["tie_rate"], 3)})
ag = pd.DataFrame(out)
ag.to_csv(HERE / "sublist_agreement.csv", index=False)
print(ag.to_string(index=False))

# Case study: is the keyword-vs-Judge-L disagreement a between-model effect?
# Same pair agreement, but pairs formed only within one model (scenario x model blocks).
print("\nCase study, pairs within the same model only (primary keyword judge and length judge):")
case["length_score"] = [-np.mean([len(t) for t in target_turns(case_convs[c])]) for c in case.conv_id]
rows = []
for setting in ["S3-safety", "S3-public_sector", "S3-healthcare", "S3-c-language"]:
    d = setting_frame(setting, val, case)
    d["scen_model"] = d["scenario"] + "|" + d["model"]
    for inst in ["primary_check", "length_score"]:
        across = PairAgreement(d, "ref", inst).point()
        within = PairAgreement(d, "ref", inst, scen_col="scen_model").point()
        rows.append({"setting": setting, "instrument": inst,
                     "agreement_all_pairs": round(across["agreement"], 3),
                     "agreement_within_model": round(within["agreement"], 3),
                     "n_pairs_within": within["n_pairs"]})
wm = pd.DataFrame(rows)
wm.to_csv(HERE / "case_within_model.csv", index=False)
print(wm.to_string(index=False))
