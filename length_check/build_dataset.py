"""Build a per-(model, run, scenario) table from the SimpleAudit case-study records.

Usage: python build_dataset.py <procurement_case_study_dir> <out_csv>
"""
import glob
import json
import os
import re
import sys

import pandas as pd

SEV = {"ERROR": 0, "critical": 0, "high": 1, "medium": 2, "low": 3, "pass": 4}
WORD_RE = re.compile(r"\w+", re.UNICODE)


def parse_model(label):
    m = re.match(r"(NbAiLab_borealis|google_gemma-3)-([\d.]+[mb])", label)
    if not m:
        return None, None
    fam = "Borealis" if "borealis" in m.group(1) else "Gemma"
    return fam, m.group(2).upper()


def main(root, out_csv):
    rows = []
    for jdir in sorted(glob.glob(os.path.join(root, "judgeBy_*"))):
        judge = os.path.basename(jdir).replace("judgeBy_", "").replace("_split_by_scenario", "")
        for pack in sorted(os.listdir(jdir)):
            for label in sorted(os.listdir(os.path.join(jdir, pack))):
                fam, size = parse_model(label)
                for rf in sorted(glob.glob(os.path.join(jdir, pack, label, "run_*.json"))):
                    run = int(re.search(r"run_(\d+)", rf).group(1))
                    data = json.load(open(rf, encoding="utf-8"))
                    for r in data["results"]:
                        conv = r.get("conversation") or []
                        tgt = [m.get("content") or "" for m in conv if m.get("role") == "assistant"]
                        aud = [m.get("content") or "" for m in conv if m.get("role") == "user"]
                        tgt_text = "\n".join(tgt)
                        rows.append(dict(
                            judge=judge, pack=pack, model=label, family=fam, size=size, run=run,
                            scenario=f"{pack}::{r['scenario_name']}",
                            severity=r["severity"], sev=SEV.get(r["severity"]),
                            score=25 * SEV.get(r["severity"], 0),
                            critical=int(r["severity"] in ("critical", "ERROR")),
                            n_target_turns=len(tgt),
                            target_chars=len(tgt_text),
                            target_words=len(WORD_RE.findall(tgt_text)),
                            target_tokens=r.get("target_output_tokens"),
                            auditor_words=len(WORD_RE.findall("\n".join(aud))),
                            ulovlig=len(re.findall(r"ulovlig", tgt_text, flags=re.I)),
                        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.shape)
    print(df.groupby(["judge", "family", "size"]).size().unstack(0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
