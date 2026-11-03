"""Two-panel figure: response length by model, and Borealis-Gemma gap raw vs length-matched.

Usage: python make_figure.py runs.csv out.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import regressions as R

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def boot_gaps(d, n=300, seed=0):
    rng = np.random.default_rng(seed)
    scen = d.scenario.unique()
    raw = {z: [] for z in R.ORDER}
    cem = {z: [] for z in R.ORDER}
    for _ in range(n):
        pick = rng.choice(scen, len(scen), replace=True)
        bd = pd.concat([d[d.scenario == s].assign(scenario=f"{s}#{k}") for k, s in enumerate(pick)])
        m = bd.groupby(["size", "family"]).score.mean().unstack()
        for z in R.ORDER:
            raw[z].append(m.loc[z, "Borealis"] - m.loc[z, "Gemma"])
        for z, (g, _, _) in R.matched_gap(bd).items():
            cem[z].append(g)
    return raw, cem


def main(path, out):
    d = R.load(path, "Qwen3.5-122B-A10B-Q4_K_S")
    raw_b, cem_b = boot_gaps(d)
    m = d.groupby(["size", "family"]).score.mean().unstack()
    cem_pt = R.matched_gap(d)
    words = d.groupby(["size", "family"]).target_words.mean().unstack().reindex(R.ORDER)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    x = np.arange(len(R.ORDER))

    # Panel 1: length
    for fam, col, mk in [("Borealis", BLUE, "o"), ("Gemma", ORANGE, "s")]:
        a1.plot(x, words[fam], color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, label=fam)
        a1.annotate(fam, (x[-1], words[fam].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    a1.set_xticks(x, R.ORDER)
    a1.set_ylim(0, 2800)
    a1.set_ylabel("Target words per 5-turn conversation")
    a1.set_title("Borealis writes about a third as much", loc="left", fontsize=11, color=INK)
    a1.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6))

    # Panel 2: gap raw vs length-matched
    off = 0.13
    for lab, col, mk, est, bs, dx in [
        ("As reported", BLUE, "o", {z: m.loc[z, "Borealis"] - m.loc[z, "Gemma"] for z in R.ORDER}, raw_b, -off),
        ("Length-matched", ORANGE, "s", {z: cem_pt[z][0] for z in R.ORDER}, cem_b, off)]:
        pts = [est[z] for z in R.ORDER]
        lo = [np.nanpercentile(bs[z], 2.5) for z in R.ORDER]
        hi = [np.nanpercentile(bs[z], 97.5) for z in R.ORDER]
        a2.errorbar(x + dx, pts, yerr=[np.array(pts) - lo, np.array(hi) - pts], fmt=mk, color=col,
                    ms=8, mec=SURF, mew=2, elinewidth=2, capsize=0, label=lab)
    a2.axhline(0, color=INK2, lw=1)
    a2.set_xticks(x, R.ORDER)
    a2.set_ylabel("Borealis − Gemma score (points, 95% CI)")
    a2.set_title("At matched length, no detectable Borealis lead", loc="left", fontsize=11, color=INK)
    a2.legend(frameon=False, loc="upper right")

    for a in (a1, a2):
        a.set_facecolor(SURF)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("Model size")
    fig.text(0.01, 0.01, "Case-study data (§7), judge and auditor Qwen3.5-122B, 36 scenarios × 10 reruns per model. "
             "Length-matched = within-scenario length quintiles; CIs from a scenario-cluster bootstrap.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180, facecolor=SURF)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
