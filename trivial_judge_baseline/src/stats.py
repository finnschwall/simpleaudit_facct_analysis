"""Rank-based statistics: AUROC with within-group bootstrap, pair agreement with cluster
bootstrap, partial eta-squared, bootstrap CI of a difference in means.

Every bootstrap creates a fresh numpy Generator from the config seed, so each CI is
reproducible on its own and independent of the order in which analyses run.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from data import CFG

SEED = CFG["seed"]
NB = CFG["n_bootstrap"]
ALPHA = 1.0 - CFG["ci_level"]
Q = (100 * ALPHA / 2, 100 * (1 - ALPHA / 2))


def auroc(safe, unsafe):
    """P(safe > unsafe), ties 1/2."""
    s = np.asarray(safe, float)[:, None]
    u = np.asarray(unsafe, float)[None, :]
    return float(((s > u) + 0.5 * (s == u)).mean())


def auroc_ci(safe, unsafe, n_boot=NB, seed=SEED):
    """Point AUROC and percentile CI, resampling runs within each group."""
    safe, unsafe = np.asarray(safe, float), np.asarray(unsafe, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bs = safe[rng.integers(0, len(safe), len(safe))]
        bu = unsafe[rng.integers(0, len(unsafe), len(unsafe))]
        boot[b] = auroc(bs, bu)
    lo, hi = np.percentile(boot, Q)
    return auroc(safe, unsafe), float(lo), float(hi)


# --------------------------------------------------------------------------- pair agreement

class PairAgreement:
    """Agreement of an instrument with a reference on within-scenario pairs.

    Pairs: two conversations on the same scenario with different reference values (higher =
    safer for both). Agreement = share of pairs where the instrument also rates the
    reference-safer conversation as safer; instrument ties count tie_value (0.5).
    Rows with NaN reference or NaN instrument (ERROR verdicts, zero target turns) are
    dropped before pairs are formed.

    Cluster bootstrap: resample clusters (runs) with replacement; a conversation enters with
    the multiplicity of its run, and each pair is counted with the product of multiplicities.
    Two copies of the same conversation have equal reference values and never form a pair.
    """

    def __init__(self, df, ref_col, inst_col, scen_col="scenario", cluster_col="cluster"):
        d = df[[scen_col, cluster_col, ref_col, inst_col]].copy()
        d[ref_col] = pd.to_numeric(d[ref_col], errors="coerce")
        d[inst_col] = pd.to_numeric(d[inst_col], errors="coerce")
        self.n_dropped = int(d[[ref_col, inst_col]].isna().any(axis=1).sum())
        d = d.dropna(subset=[ref_col, inst_col])
        self.n_conversations = len(d)
        clusters = sorted(d[cluster_col].unique())
        self.clusters = clusters
        cidx = {c: k for k, c in enumerate(clusters)}
        tie_value = CFG["q2"]["tie_value"]
        self.blocks = []
        for _, g in d.groupby(scen_col, sort=True):
            r = g[ref_col].to_numpy(float)
            x = g[inst_col].to_numpy(float)
            safer = r[:, None] > r[None, :]                  # i reference-safer than j
            agree = safer & (x[:, None] > x[None, :])
            tie = safer & (x[:, None] == x[None, :])
            self.blocks.append({
                "codes": np.array([cidx[c] for c in g[cluster_col]]),
                "D": safer.astype(float),
                "A": agree.astype(float) + tie_value * tie.astype(float),
                "AG": agree.astype(float),
                "T": tie.astype(float),
            })

    def _sums(self, w_cluster=None):
        n_pairs = n_score = n_agree = n_tie = 0.0
        for b in self.blocks:
            if w_cluster is None:
                n_pairs += b["D"].sum(); n_score += b["A"].sum()
                n_agree += b["AG"].sum(); n_tie += b["T"].sum()
            else:
                w = w_cluster[b["codes"]]
                n_pairs += w @ b["D"] @ w; n_score += w @ b["A"] @ w
        return n_pairs, n_score, n_agree, n_tie

    def point(self):
        n_pairs, n_score, n_agree, n_tie = self._sums()
        n_dis = n_pairs - n_agree - n_tie
        return {
            "n_conversations": self.n_conversations,
            "n_dropped_nan": self.n_dropped,
            "n_clusters": len(self.clusters),
            "n_pairs": int(n_pairs),
            "agreement": n_score / n_pairs if n_pairs else float("nan"),
            "agreement_no_ties": n_agree / (n_agree + n_dis) if (n_agree + n_dis) else float("nan"),
            "tie_rate": n_tie / n_pairs if n_pairs else float("nan"),
            "n_agree": int(n_agree), "n_tie": int(n_tie), "n_disagree": int(n_dis),
        }

    def ci(self, n_boot=NB, seed=SEED):
        rng = np.random.default_rng(seed)
        k = len(self.clusters)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
            n_pairs, n_score, _, _ = self._sums(w)
            boot[b] = n_score / n_pairs if n_pairs else np.nan
        lo, hi = np.nanpercentile(boot, Q)
        return float(lo), float(hi)

    def full(self, with_ci=True):
        out = self.point()
        out["ci_lo"], out["ci_hi"] = self.ci() if with_ci else (float("nan"), float("nan"))
        return out


# --------------------------------------------------------------------------- Q3

def partial_eta2(runs, score_col="score"):
    """OLS score ~ C(target) + C(auditor), Type II SS; partial eta^2 = SS_f / (SS_f + SS_res)."""
    d = runs[["target", "auditor", score_col]].rename(columns={score_col: "y"})
    if d["y"].isna().any():
        raise ValueError("NaN run score in Q3 data")
    fit = smf.ols("y ~ C(target) + C(auditor)", data=d).fit()
    tab = anova_lm(fit, typ=2)
    ss_res = tab.loc["Residual", "sum_sq"]
    out = {"n_runs": len(d)}
    for f, name in (("C(target)", "target"), ("C(auditor)", "auditor")):
        out[f"eta2_{name}"] = float(tab.loc[f, "sum_sq"] / (tab.loc[f, "sum_sq"] + ss_res))
        out[f"F_{name}"] = float(tab.loc[f, "F"])
        out[f"p_{name}"] = float(tab.loc[f, "PR(>F)"])
    return out


# --------------------------------------------------------------------------- case study

def diff_means_ci(a, b, n_boot=NB, seed=SEED):
    """mean(a) - mean(b) with percentile CI, resampling runs within each model."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
    lo, hi = np.percentile(boot, Q)
    return float(a.mean() - b.mean()), float(lo), float(hi)
