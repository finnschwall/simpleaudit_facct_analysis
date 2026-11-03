"""Gap by model size under four judge/construct settings. Usage: python make_rejudge_figure.py out.png"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import analyze_rejudge as A

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main(out):
    d = pd.read_csv("rejudged_binary_merged.csv")
    exp = pd.read_csv("expected_outcomes.csv")
    emap = dict(zip(exp.scenario_name, exp.expected_outcome))
    c = A.load_rejudged("rejudged", "safety__claude-haiku-5-5").rename(columns={"score_j": "generic_haiku"})
    c = c.merge(d[["family", "size", "run", "scenario_name", "generic_122b"]],
                on=["family", "size", "run", "scenario_name"])
    c["expected"] = c.scenario_name.map(emap)
    g = d[d.expected.isin(["answer", "refuse"])].copy()
    cg = c[c.expected.isin(["answer", "refuse"])]
    alt = dict(emap)
    for s in A.CONSTRUCT_CHOICES:
        alt[s] = "answer"
    g2 = d.assign(expected=d.scenario_name.map(alt))
    g2["outcome_score"] = [A.outcome_score(a, e) for a, e in zip(g2.abstained, g2.expected)]
    g2 = g2[g2.expected.isin(["answer", "refuse"])]

    series = [
        ("Generic rubric, 122B judge (paper)", "#2a78d6", "o", A.gap_with_ci(g, "generic_122b")),
        ("Generic rubric, Haiku judge (control)", "#eb6834", "s", A.gap_with_ci(cg, "generic_haiku")),
        ("Outcome-aware, Haiku judge", "#1baf7a", "D", A.gap_with_ci(g, "outcome_score")),
        ("Outcome-aware, 3 referral scenarios counted as should-answer", "#eda100", "^",
         A.gap_with_ci(g2, "outcome_score")),
    ]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURF)
    x = np.arange(len(A.ORDER))
    offs = np.linspace(-0.27, 0.27, len(series))
    for (lab, col, mk, gp), dx in zip(series, offs):
        pts = np.array([gp[z][0] for z in A.ORDER])
        lo = np.array([gp[z][1] for z in A.ORDER])
        hi = np.array([gp[z][2] for z in A.ORDER])
        ax.errorbar(x + dx, pts, yerr=[pts - lo, hi - pts], fmt=mk, color=col, ms=8, mec=SURF, mew=1.5,
                    elinewidth=2, capsize=0, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, A.ORDER)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Borealis − Gemma score (points, 95% CI)")
    ax.set_title("Grading refusals against the expected outcome shrinks Borealis's lead at 4B–27B",
                 loc="left", fontsize=11, color=INK)
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.text(0.01, 0.01, "Same 2,480 case-study transcripts (31 scenarios with a clear expected outcome × 8 models × 10 reruns). "
             "CIs: bootstrap over scenarios.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180, facecolor=SURF)


if __name__ == "__main__":
    main(sys.argv[1])
