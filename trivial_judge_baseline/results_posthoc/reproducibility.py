"""Post hoc: the paper's reproducibility step applied to the trivial judges.

The 10 runs of a cell are 10 different conversations scored by the same judge, so a
deterministic judge still has run-to-run variation from the conversations.

1. Detection reliability vs k: the paper's own function (analyze_data.compute_detection_stability),
   rank-based, pooled target sizes, per auditor.
2. Scenario-profile stability, scale-free: for k runs drawn without replacement, Spearman
   correlation between the k-run per-scenario mean profile and the 10-run profile, per
   (target, auditor) cell, averaged over cells. (The paper uses MAD in score units, which is
   not comparable across instruments.)
3. Split-half ranking reproducibility: split the 10 runs of every target into two random
   halves of 5, rank the targets by mean run score in each half, Kendall tau-b between the
   two rankings. Per auditor, over the 6 target levels and over the 3 safe sizes.

LLM run/conversation value = 25 * (4 - ordinal), ERROR = 0. Run from the repository root.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, ".")
import analyze_data as ad  # noqa: E402

ORD = {"pass": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
SEED, NB = 20261031, 1000
OUT = "trivial_judge_baseline/results_posthoc/"
INST = {"J=L": "sev_L", "J=M": "sev_M", "keyword": "keyword", "length": "length"}

conv = pd.read_csv("trivial_judge_baseline/results/conversation_scores.csv")
val = conv[(conv.dataset == "validation") & conv.auditor.isin(["M", "L"])].copy()
for name, col in INST.items():
    if col.startswith("sev_"):
        val[name] = val[col].map(lambda s: 25.0 * (4 - ORD[s]) if s in ORD else 0.0)
    else:
        val[name] = val[col]
val["target_safe"] = ~val.target.str.endswith("_UNSAFE")
runs = val.groupby(["target", "auditor", "run", "target_safe"])[list(INST)].mean().reset_index()

# --- 1. detection reliability vs k (paper's function) -----------------------------------
long = runs.melt(id_vars=["target", "auditor", "run", "target_safe"], value_vars=list(INST),
                 var_name="judge", value_name="score")
det = ad.compute_detection_stability(long, n_bootstrap=NB, rng_seed=42)
det = det[det.k <= 10]
print("== 1. Detection reliability vs k (paper's function; pooled sizes) ==")
for (a, j), g in det.groupby(["auditor", "judge"]):
    above = g[g.ci_lo > 0.5]
    k_min = int(above.k.min()) if not above.empty else ">10"
    r10 = g[g.k == 10].iloc[0]
    r1 = g[g.k == 1].iloc[0]
    print(f"  A={a} {j:8s} AUROC@k=1 {r1.auroc:.3f}  AUROC@k=10 {r10.auroc:.3f} "
          f"[{r10.ci_lo:.3f}, {r10.ci_hi:.3f}]  ci_lo>0.5 from k={k_min}")

# --- 2. scenario-profile stability (Spearman vs 10-run profile) --------------------------
rng = np.random.default_rng(SEED)
rows = []
for (t, a), g in val.groupby(["target", "auditor"]):
    piv = {n: g.pivot_table(index="run", columns="scenario", values=n) for n in INST}
    run_ids = sorted(g.run.unique())
    for k in (1, 3, 5):
        for _ in range(200):
            pick = rng.choice(run_ids, k, replace=False)
            for n, p in piv.items():
                full, sub = p.mean(), p.loc[pick].mean()
                rho = spearmanr(sub, full).statistic if sub.nunique() > 1 and full.nunique() > 1 else np.nan
                rows.append({"target": t, "auditor": a, "safe": not t.endswith("_UNSAFE"),
                             "k": k, "instrument": n, "rho": rho})
prof = pd.DataFrame(rows)
print("\n== 2. Scenario-profile stability: mean Spearman(k-run profile, 10-run profile) ==")
tab = prof.groupby(["safe", "k", "instrument"]).rho.mean().unstack("instrument")[list(INST)]
print(tab.round(2).to_string())
print("  share of draws with undefined rho (constant profile):")
print(prof.assign(undef=prof.rho.isna()).groupby(["safe", "instrument"]).undef.mean()
      .unstack("instrument")[list(INST)].round(2).to_string())

# --- 3. split-half ranking reproducibility ----------------------------------------------
rng = np.random.default_rng(SEED)
rows = []
for a, g in runs.groupby("auditor"):
    targets = sorted(g.target.unique())
    by_t = {t: g[g.target == t].sort_values("run") for t in targets}
    for _ in range(NB):
        halves = {}
        for t in targets:
            idx = rng.permutation(len(by_t[t]))
            halves[t] = (by_t[t].iloc[idx[:5]], by_t[t].iloc[idx[5:]])
        for n in INST:
            for subset, ts in (("all 6 levels", targets),
                               ("safe XS/S/M", [t for t in targets if not t.endswith("_UNSAFE")])):
                h1 = [halves[t][0][n].mean() for t in ts]
                h2 = [halves[t][1][n].mean() for t in ts]
                rows.append({"auditor": a, "subset": subset, "instrument": n,
                             "tau": kendalltau(h1, h2).statistic})
sh = pd.DataFrame(rows)
print("\n== 3. Split-half ranking reproducibility: mean Kendall tau-b between two 5-run halves ==")
print(sh.groupby(["subset", "auditor", "instrument"]).tau.mean().unstack("instrument")[list(INST)]
      .round(2).to_string())
print("  share of splits with identical ranking (tau = 1):")
print(sh.assign(same=np.isclose(sh.tau, 1.0)).groupby(["subset", "auditor", "instrument"]).same.mean()
      .unstack("instrument")[list(INST)].round(2).to_string())

det.to_csv(OUT + "reproducibility_detection_vs_k.csv", index=False)
tab.to_csv(OUT + "reproducibility_profile.csv")
sh.groupby(["subset", "auditor", "instrument"]).tau.describe().to_csv(OUT + "reproducibility_splithalf.csv")
