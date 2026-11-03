"""Post hoc. Descriptive per-phrase hit rates of the primary keyword judge.

Read-only: loads the stored transcripts with the project's own loaders (src/data.py) and the
committed word list; writes only into results_posthoc/.

Outputs:
  phrase_hits_validation.csv  per string: share of target turns containing it, safe vs abliterated
                              (Q1 data: A in {M, L}, all six targets) and in S2 (safe, A in M/L/XL)
  phrase_hits_case.csv        per string: share of target turns containing it, Borealis vs Gemma
                              (matched sizes 1b/4b/12b/27b, all four packs)
  snippets.txt                a few context snippets for the most frequent strings
Run from the repository root:
  python trivial_judge_baseline/results_posthoc/phrase_hits.py > trivial_judge_baseline/results_posthoc/phrase_hits_output.txt
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TJB = HERE.parent
sys.path.insert(0, str(TJB / "src"))
from data import load_validation, load_case, target_turns, load_wordlist  # noqa: E402
from judges import phrase_strings, turn_matches  # noqa: E402

pd.set_option("display.width", 200)
entries, _ = load_wordlist()
strings, owners = phrase_strings(entries, ("en", "nb"))
lang_of = {}
for e in entries:
    lang_of.setdefault(e["en"], "en")
    lang_of.setdefault(e["nb"], "nb")


def turn_table(conv_df, conversations, keys):
    rows = []
    for _, r in conv_df.iterrows():
        for t_i, text in enumerate(target_turns(conversations[r.conv_id])):
            m = turn_matches(text, strings, "substring")
            row = {k: r[k] for k in keys}
            row.update(conv_id=r.conv_id, turn=t_i, n_hits=len(m), flagged=bool(m))
            for i in range(len(strings)):
                row[f"s{i}"] = i in m
            rows.append(row)
    return pd.DataFrame(rows)


def per_string(tt, group_col, groups):
    out = []
    for i, s in enumerate(strings):
        row = {"string": s, "lang": lang_of.get(s, "?"), "entries": sorted(owners[i])}
        for g in groups:
            sub = tt[tt[group_col] == g]
            hit = sub[f"s{i}"]
            row[f"rate_{g}"] = hit.mean()
            row[f"sole_{g}"] = (hit & (sub.n_hits == 1)).mean()   # only string matching the turn
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- validation
val, _, _, conversations = load_validation()
q1 = val[val.auditor.isin(["M", "L"])].copy()
q1["grp"] = q1.target_safe.map({True: "safe", False: "abliterated"})
tt = turn_table(q1, conversations, ["grp", "target", "auditor", "scenario"])
ps = per_string(tt, "grp", ["safe", "abliterated"])
s2 = val[val.target_safe & val.auditor.isin(["M", "L", "XL"])].copy()
s2["grp"] = "S2"
tt2 = turn_table(s2, conversations, ["grp", "target", "auditor", "scenario"])
ps2 = per_string(tt2, "grp", ["S2"])
ps = ps.merge(ps2[["string", "rate_S2", "sole_S2"]], on="string")
ps["diff_safe_minus_abl"] = ps.rate_safe - ps.rate_abliterated
ps = ps.sort_values("diff_safe_minus_abl", ascending=False)
ps.to_csv(HERE / "phrase_hits_validation.csv", index=False)
print("== Validation, Q1 data (A in M, L) ==")
print("target turns:", tt.groupby("grp").size().to_dict(),
      "flagged share:", tt.groupby("grp").flagged.mean().round(3).to_dict())
print(ps[(ps.rate_safe > 0.005) | (ps.rate_abliterated > 0.005)].round(3).to_string(index=False))

print("\nflagged share by scenario (Q1 data):")
print(tt.pivot_table(index="scenario", columns="grp", values="flagged", aggfunc="mean").round(3))
print("\nflagged share by auditor x group (Q1 data):")
print(tt.pivot_table(index="auditor", columns="grp", values="flagged", aggfunc="mean").round(3))
i_hei = strings.index("Hei!")
print("\n'Hei!' share by auditor x group:")
print(tt.pivot_table(index="auditor", columns="grp", values=f"s{i_hei}", aggfunc="mean").round(3))

for drop in [["ulovlig", "illegal"], ["Som en"], ["Hei!"], ["ulovlig", "illegal", "Som en", "Hei!", "Jeg er en"]]:
    idx = [strings.index(s) for s in drop if s in strings]
    keep = [f"s{i}" for i in range(len(strings)) if i not in idx]
    f = tt[keep].any(axis=1)
    print("without", drop, "-> flagged share:", f.groupby(tt.grp).mean().round(3).to_dict())

# ---------------------------------------------------------------- case study
case, case_convs = load_case()
case = case[~case.model.str.contains("270m")].copy()
case["family"] = case.model.map(lambda m: "borealis" if "borealis" in m.lower() else "gemma")
tc = turn_table(case, case_convs, ["family", "pack", "model"])
pc = per_string(tc, "family", ["borealis", "gemma"])
pc["diff_bor_minus_gem"] = pc.rate_borealis - pc.rate_gemma
pc = pc.sort_values("diff_bor_minus_gem")
pc.to_csv(HERE / "phrase_hits_case.csv", index=False)
print("\n== Case study (matched sizes) ==")
print("target turns:", tc.groupby("family").size().to_dict(),
      "flagged share:", tc.groupby("family").flagged.mean().round(3).to_dict())
print(tc.pivot_table(index="pack", columns="family", values="flagged", aggfunc="mean").round(3))
print(pc[(pc.rate_borealis > 0.005) | (pc.rate_gemma > 0.005)].round(3).to_string(index=False))

# ---------------------------------------------------------------- snippets
with open(HERE / "snippets.txt", "w", encoding="utf-8") as fh:
    for s in ["Som en", "ulovlig", "Hei!", "Jeg er en", "Beklager", "Jeg kan ikke"]:
        i = strings.index(s)
        fh.write(f"===== {s!r}\n")
        n = int(tt[f"s{i}"].sum())
        sub = tt[tt[f"s{i}"]].sample(n=min(5, n), random_state=20261031)
        for _, r in sub.iterrows():
            text = target_turns(conversations[r.conv_id])[r.turn]
            p = text.find(s)
            fh.write(f"[{r.grp} {r.target} A={r.auditor} {r.scenario}] ...{text[max(0, p - 120):p + 120]!r}...\n")
