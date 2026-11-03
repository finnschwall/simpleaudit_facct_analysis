"""Data loading for the trivial-judge baseline.

Reads the stored validation-sweep and case-study records. No model is called.
Validation: comparison_results/norwegian_safety/T={T}__A={A}/J={J}/experiment_results.json
Case study: <case_records>/<pack>/<model>/run_<i>.json  (Judge L records)
"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
TJB = SRC.parent                      # trivial_judge_baseline/
ROOT = TJB.parent                     # repository root
CFG = json.loads((TJB / "config.json").read_text(encoding="utf-8"))

TARGETS = ["XS", "S", "M", "XS_UNSAFE", "S_UNSAFE", "M_UNSAFE"]
SAFE_TARGETS = ["XS", "S", "M"]
AUDITORS = ["M", "L", "XL"]
JUDGES = ["XS", "S", "M", "L", "XL"]
ORD = CFG["severity_ordinal"]          # pass 0 ... critical 4
ERROR = CFG["error_verdict"]
ROLE = CFG["target_role"]              # "assistant"

CELL_RE = re.compile(r"^T=(?P<t>XS|S|M)(?P<u>_UNSAFE)?__A=(?P<a>XS|S|M|L|XL)(?P<au>_UNSAFE)?$")
JUDGE_RE = re.compile(r"^J=(?P<j>XS|S|M|L|XL)$")   # excludes J=XL2 and J=*_UNSAFE


def conv_hash(scenario_name, scenario_description, conversation):
    blob = json.dumps({"n": scenario_name, "d": scenario_description, "c": conversation},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def target_turns(conversation):
    """Contents of the target turns (role 'assistant'), as stored."""
    return [m["content"] for m in conversation if m["role"] == ROLE]


def load_validation():
    """Returns (convs, runs, checks).

    convs: one row per validation conversation (each conversation once, not per judge),
           with columns conv_id, dataset, cell, target, auditor, target_safe, target_size,
           run, scen_pos, scenario, scenario_description, cluster, sev_<J> for each judge
           file present in the cell (NaN where that judge was not run).
    runs:  one row per (cell, judge, run) with the stored run score summary.score.
    checks: list of dicts documenting the transcript-equality check per cell.
    conversations: dict conv_id -> list of {role, content}
    """
    root = ROOT / CFG["paths"]["validation_records"]
    conv_rows, run_rows, checks = [], [], []
    conversations = {}
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = CELL_RE.match(cell_dir.name)
        if not m or m["au"]:
            continue                                  # skips T=S_UNSAFE__A=S_UNSAFE
        target = m["t"] + (m["u"] or "")
        auditor = m["a"]
        if target not in TARGETS or auditor not in AUDITORS:
            continue
        judge_dirs = sorted((d for d in cell_dir.iterdir() if d.is_dir() and JUDGE_RE.match(d.name)),
                            key=lambda d: JUDGES.index(JUDGE_RE.match(d.name)["j"]))
        ref_hashes = None
        per_judge_sev = {}
        ref_conv = None
        for jd in judge_dirs:
            j = JUDGE_RE.match(jd.name)["j"]
            if not (jd / "experiment_results.json").exists():
                # incomplete judge run (only T=XS__A=L/J=S: 8 run files, no experiment_results.json);
                # not used by any analysis, recorded in the checks
                checks.append({"cell": cell_dir.name, "judge": j, "n_runs": None, "n_conversations": None,
                               "identical_to_first_judge": None, "first_judge": None,
                               "skipped": "no experiment_results.json"})
                continue
            d = json.loads((jd / "experiment_results.json").read_text(encoding="utf-8"))
            models = list(d["runs"])
            if len(models) != 1:
                raise ValueError(f"{jd}: expected one model, got {models}")
            runs = d["runs"][models[0]]
            hashes, sevs, convs = [], [], []
            for i, run in enumerate(runs):
                names = [r["scenario_name"] for r in run["results"]]
                if len(set(names)) != len(names):
                    raise ValueError(f"{jd} run {i}: duplicate scenario names")
                run_rows.append({"cell": cell_dir.name, "target": target, "auditor": auditor,
                                 "judge": j, "run": i, "score": float(run["summary"]["score"]),
                                 "n_results": len(run["results"]),
                                 "n_error": sum(r["severity"] == ERROR for r in run["results"])})
                for pos, r in enumerate(run["results"]):
                    hashes.append((i, pos, conv_hash(r["scenario_name"], r["scenario_description"],
                                                     r["conversation"])))
                    sevs.append(r["severity"])
                    convs.append((i, pos, r))
            if ref_hashes is None:
                ref_hashes, ref_conv, ref_judge = hashes, convs, j
                identical = True
            else:
                identical = hashes == ref_hashes
                if not identical:
                    raise AssertionError(f"{cell_dir.name}: J={j} conversations differ from J={ref_judge}")
            per_judge_sev[j] = sevs
            checks.append({"cell": cell_dir.name, "judge": j, "n_runs": len(runs),
                           "n_conversations": len(hashes), "identical_to_first_judge": identical,
                           "first_judge": ref_judge})
        for k, (i, pos, r) in enumerate(ref_conv):
            cid = f"val|T={target}|A={auditor}|run={i}|{r['scenario_name']}"
            if cid in conversations:
                raise ValueError(f"duplicate conversation id {cid}")
            conversations[cid] = r["conversation"]
            row = {"conv_id": cid, "dataset": "validation", "cell": cell_dir.name,
                   "target": target, "auditor": auditor,
                   "target_safe": not target.endswith("_UNSAFE"),
                   "target_size": target.replace("_UNSAFE", ""), "run": i, "scen_pos": pos,
                   "scenario": r["scenario_name"], "scenario_description": r["scenario_description"],
                   "cluster": f"val|T={target}|A={auditor}|run={i}"}
            for j, sevs in per_judge_sev.items():
                row[f"sev_{j}"] = sevs[k]
            conv_rows.append(row)
    convs = pd.DataFrame(conv_rows)
    for j in JUDGES:
        if f"sev_{j}" not in convs:
            convs[f"sev_{j}"] = pd.NA
    return convs, pd.DataFrame(run_rows), checks, conversations


def load_case():
    """Case-study Judge-L records. Returns (convs, conversations).

    convs: one row per conversation: conv_id, dataset, pack, model, run, scen_pos,
           scenario, scenario_description, cluster, sev_caseL, run_score_stored.
    """
    root = ROOT / CFG["paths"]["case_records"]
    rows, conversations = [], {}
    for pack_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in pack_dir.iterdir() if p.is_dir()):
            for f in sorted(model_dir.glob("run_*.json"), key=lambda p: int(re.search(r"run_(\d+)", p.name)[1])):
                i = int(re.search(r"run_(\d+)\.json$", f.name)[1])
                d = json.loads(f.read_text(encoding="utf-8"))
                names = [r["scenario_name"] for r in d["results"]]
                if len(set(names)) != len(names):
                    raise ValueError(f"{f}: duplicate scenario names")
                for pos, r in enumerate(d["results"]):
                    cid = f"case|{pack_dir.name}|{model_dir.name}|run={i}|{r['scenario_name']}"
                    conversations[cid] = r["conversation"]
                    rows.append({"conv_id": cid, "dataset": "case", "pack": pack_dir.name,
                                 "model": model_dir.name, "run": i, "scen_pos": pos,
                                 "scenario": r["scenario_name"],
                                 "scenario_description": r["scenario_description"],
                                 "cluster": f"case|{model_dir.name}|run={i}",
                                 "sev_caseL": r["severity"],
                                 "run_score_stored": float(d["summary"]["score"])})
    return pd.DataFrame(rows), conversations


def load_mapping_module():
    """Import mapping_sensitivity/severity_mapping_sensitivity.py (stdlib only, no side effects)."""
    path = ROOT / "mapping_sensitivity" / "severity_mapping_sensitivity.py"
    spec = importlib.util.spec_from_file_location("severity_mapping_sensitivity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safety_value(sev):
    """LLM verdict as a 'higher is safer' value: 4 - ordinal; NaN for ERROR or missing."""
    if not isinstance(sev, str) or sev == ERROR:
        return float("nan")
    return float(4 - ORD[sev])


def load_wordlist():
    """Returns (entries, path). entries: list of {"en": str, "nb": str} exactly as stored."""
    path = ROOT / CFG["paths"]["wordlist"]
    if not path.exists():
        return None, path
    d = json.loads(path.read_text(encoding="utf-8"))
    entries = d["phrases"]
    for e in entries:
        if not isinstance(e.get("en"), str) or not isinstance(e.get("nb"), str):
            raise ValueError(f"word-list entry without en/nb strings: {e}")
        if not e["en"] or not e["nb"]:
            raise ValueError(f"empty phrase string would match every turn: {e}")
    return entries, path
