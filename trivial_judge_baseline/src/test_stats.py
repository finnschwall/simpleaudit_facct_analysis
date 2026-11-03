"""Brute-force checks of the pair-agreement statistic and its cluster bootstrap weighting.
Run: python trivial_judge_baseline/src/test_stats.py"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import PairAgreement, auroc  # noqa: E402


def brute(df):
    """Unordered pairs within scenario with different ref; tie in inst = 0.5."""
    num = den = 0.0
    rows = df.dropna(subset=["ref", "inst"]).to_dict("records")
    for a, b in itertools.combinations(rows, 2):
        if a["scenario"] != b["scenario"] or a["ref"] == b["ref"]:
            continue
        hi, lo = (a, b) if a["ref"] > b["ref"] else (b, a)
        den += 1
        num += 1.0 if hi["inst"] > lo["inst"] else 0.5 if hi["inst"] == lo["inst"] else 0.0
    return num / den, den


def main():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"scenario": rng.integers(0, 4, n), "cluster": rng.integers(0, 15, n),
                       "ref": rng.integers(0, 5, n).astype(float), "inst": rng.integers(0, 4, n).astype(float)})
    df.loc[3, "ref"] = np.nan
    df.loc[7, "inst"] = np.nan
    pa = PairAgreement(df, "ref", "inst")
    p = pa.point()
    b, nb = brute(df)
    assert abs(p["agreement"] - b) < 1e-12 and p["n_pairs"] == nb, (p, b, nb)

    # bootstrap weighting: weights w_c must equal explicitly duplicating clusters
    w = np.zeros(len(pa.clusters))
    w[0], w[2], w[5] = 1, 2, 3          # cluster 0 once, 2 twice, 5 three times
    num, den = 0.0, 0.0
    for blk in pa.blocks:
        ww = w[blk["codes"]]
        num += ww @ blk["A"] @ ww
        den += ww @ blk["D"] @ ww
    d = df.dropna(subset=["ref", "inst"])
    parts = []
    for k, c in enumerate(pa.clusters):
        for rep in range(int(w[k])):
            parts.append(d[d.cluster == c])
    dup = pd.concat(parts, ignore_index=True)
    bb, bnb = brute(dup)
    assert abs(num / den - bb) < 1e-12 and abs(den - bnb) < 1e-9, (num / den, bb, den, bnb)

    assert auroc([1, 2, 3], [0, 2]) == (1 + 1.5 + 2) / 6

    # keyword mechanics on toy strings (not a word list)
    from judges import phrase_strings, turn_matches
    ents = [{"en": "Foo", "nb": "Fuu"}, {"en": "Bar", "nb": "Bar"}, {"en": "Baz", "nb": "Foo"}]
    s, own = phrase_strings(ents, ("en", "nb"))
    assert s == ["Foo", "Fuu", "Bar", "Baz"] and own == [{0, 2}, {0}, {1}, {2}], (s, own)
    assert turn_matches("x Foo y", s, "substring") == {0}
    assert turn_matches("x foo y", s, "substring") == frozenset()          # case-sensitive
    assert turn_matches("  Bar is here", s, "prefix") == {2}
    assert turn_matches("here Bar", s, "prefix") == frozenset()
    removed0 = {i for i, o in enumerate(own) if o == {0}}
    assert removed0 == {1}       # "Foo" stays because entry 2 also contains it
    print("test_stats: all checks passed")


if __name__ == "__main__":
    main()
