"""Length vs score by judge, split by the stakeholder-defined expected outcome.

Each simulachat_<judge>.json is a separate single-turn run of the same 47 prompts,
scored by one judge config. Usage: python helsedir_check.py <dir_with_jsons_and_comparison.csv>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEV = {"ERROR": 0, "critical": 0, "high": 1, "medium": 2, "low": 3, "pass": 4}
SHORT = 300  # words; in the binary_abstention run every non-abstaining reply was >= 318 words


def load(root):
    exp = dict(pd.read_csv(os.path.join(root, "comparison.csv"))[["scenario_name", "expected_outcome"]].values)
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "simulachat_*.json"))):
        judge = os.path.basename(f)[len("simulachat_"):-5]
        for r in json.load(open(f, encoding="utf-8"))["results"]:
            reply = r["conversation"][-1]["content"]
            j = r.get("judgment") or {}
            rows.append(dict(judge=judge, scenario=r["scenario_name"], expected=exp[r["scenario_name"]],
                             sev=SEV[r["severity"]], words=len(reply.split()),
                             abstained_label=j.get("abstained")))
    d = pd.DataFrame(rows)
    d["short"] = d.words < SHORT
    return d


def boot_rho(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(x[b])) > 1 and len(set(y[b])) > 1:
            vals.append(spearmanr(x[b], y[b])[0])
    return np.percentile(vals, [2.5, 97.5])


def main(root):
    d = load(root)
    order = ["safety", "helsedir_no", "helsedir_no_rag", "abstention", "binary_abstention"]
    print("Spearman(score, reply length) within each expected outcome  [95% bootstrap CI]")
    print("  construct-following predicts: answer >= 0 (short = deflection), refuse < 0 (long = failed to decline)")
    print("  pure brevity preference predicts: negative in both\n")
    for j in order:
        x = d[d.judge == j]
        out = []
        for e in ["answer", "refuse"]:
            s = x[x.expected == e]
            rho = spearmanr(s.words, s.sev)[0]
            lo, hi = boot_rho(s.words.values, s.sev.values)
            out.append(f"{e}: {rho:+.2f} [{lo:+.2f},{hi:+.2f}] (n={len(s)})")
        print(f"{j:18s} " + "   ".join(out))

    print("\nMean score (0-4) by expected outcome x reply length (short < 300 words)")
    t = d.groupby(["judge", "expected", "short"]).sev.agg(["mean", "size"]).round(2)
    t = t.unstack(["expected", "short"])
    print(t.reindex(order).to_string())

    print("\nOverall score (0-100) per judge:", (d.groupby("judge").sev.mean() * 25).reindex(order).round(1).to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
