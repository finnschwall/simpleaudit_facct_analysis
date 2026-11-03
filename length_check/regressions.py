"""Length-controlled analysis of the Borealis vs Gemma case study.

Usage: python regressions.py runs.csv [judge]
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ORDER = ["1B", "4B", "12B", "27B"]


def load(path, judge):
    df = pd.read_csv(path)
    d = df[(df.judge == judge) & df.family.notna() & df["size"].isin(ORDER)].copy()
    d["logw"] = np.log1p(d.target_words)
    d["B"] = (d.family == "Borealis").astype(int)
    d["cell"] = d.model + "|" + d.scenario
    # within-scenario standardised log length (pooled over models)
    d["logw_z"] = d.groupby("scenario").logw.transform(lambda x: (x - x.mean()) / x.std())
    return d


def fam_effect(fit, size):
    name = f"B:C(size)[{size}]"
    return fit.params[name], fit.bse[name], fit.pvalues[name]


def gap_models(d):
    out = []
    specs = {
        "raw (scenario FE)": "score ~ B:C(size) + C(size) + C(scenario)",
        "+ log length": "score ~ B:C(size) + C(size) + C(scenario) + logw",
        "+ log length x size": "score ~ B:C(size) + C(size) + C(scenario) + logw:C(size)",
        "+ log length x scenario": "score ~ B:C(size) + C(size) + C(scenario) + logw:C(scenario)",
    }
    for lab, f in specs.items():
        fit = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.scenario.astype("category").cat.codes})
        row = {"spec": lab}
        for z in ORDER:
            b, se, p = fam_effect(fit, z)
            row[z] = f"{b:+.1f} ({se:.1f}){'*' if p < .05 else ''}"
        if "logw" in fit.params:
            row["logw coef"] = f"{fit.params['logw']:+.1f} ({fit.bse['logw']:.1f})"
        out.append(row)
    return pd.DataFrame(out).set_index("spec")


def within_cell_length(d):
    """Length effect holding model and scenario fixed (variation across reruns only)."""
    fit = smf.ols("score ~ logw + C(cell)", d).fit(
        cov_type="cluster", cov_kwds={"groups": d.scenario.astype("category").cat.codes})
    sd_within = d.groupby("cell").logw.transform(lambda x: x - x.mean()).std()
    return fit.params["logw"], fit.bse["logw"], sd_within


def matched_gap(d, nbins=5):
    """Coarsened exact matching: within scenario x size, bin log length into quantiles
    (pooled over both families) and compare families inside bins that contain both."""
    res = {}
    for z in ORDER:
        dz = d[d["size"] == z].copy()
        dz["bin"] = dz.groupby("scenario").logw.transform(
            lambda x: pd.qcut(x.rank(method="first"), nbins, labels=False))
        g = dz.groupby(["scenario", "bin", "family"]).score.agg(["mean", "size"]).unstack("family")
        g = g.dropna()
        if g.empty:
            res[z] = (np.nan, 0, 0.0)
            continue
        w = np.minimum(g[("size", "Borealis")], g[("size", "Gemma")])
        gap = np.average(g[("mean", "Borealis")] - g[("mean", "Gemma")], weights=w)
        n_matched = int(w.sum() * 2)
        res[z] = (gap, n_matched, n_matched / len(dz))
    return res


def main(path, judge):
    d = load(path, judge)
    print(f"Judge: {judge}   N = {len(d)} scenario-runs\n")

    print("Mean target words per 5-turn conversation (and median)")
    print(d.groupby(["size", "family"]).target_words.agg(["mean", "median"]).unstack().reindex(ORDER).round(0), "\n")

    print("Correlation score vs log length")
    print("  pooled:", round(d[["score", "logw"]].corr().iloc[0, 1], 3))
    within_scen = d.groupby("scenario").apply(lambda x: x[["score", "logw"]].corr().iloc[0, 1], include_groups=False)
    print("  within-scenario median r:", round(within_scen.median(), 3),
          f"(neg in {(within_scen < 0).sum()}/{len(within_scen)} scenarios)")
    for fam in ["Borealis", "Gemma"]:
        x = d[d.family == fam]
        print(f"  {fam} only:", round(x[["score", "logw"]].corr().iloc[0, 1], 3))
    print()

    print("Borealis - Gemma score gap (points on 0-100), cluster SE by scenario")
    print(gap_models(d).to_string(), "\n")

    b, se, sdw = within_cell_length(d)
    print(f"Within model x scenario (reruns only): score ~ log length slope = {b:+.1f} (SE {se:.1f});"
          f" within-cell SD of log length = {sdw:.2f}\n")

    print("Length-matched gap (CEM, 5 within-scenario length bins)")
    for z, (gap, n, frac) in matched_gap(d).items():
        print(f"  {z}: gap {gap:+.1f}  matched n={n} ({frac:.0%} of runs)")
    print()

    # Length-only and keyword baselines: do they reproduce the model ranking?
    m = d.groupby("model").agg(score=("score", "mean"), words=("target_words", "mean"),
                               ulovlig=("ulovlig", "mean"), fam=("family", "first"), size=("size", "first"))
    print("Model-level: SA score vs mean length (Spearman):",
          round(m.score.corr(m.words, method="spearman"), 3),
          "| vs 'ulovlig' count:", round(m.score.corr(m.ulovlig, method="spearman"), 3))
    print(m.sort_values("score").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Qwen3.5-122B-A10B-Q4_K_S")
