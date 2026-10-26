import json
import os
import statistics
from pathlib import Path

BASE_DIR = Path(".")

scenarios = sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir())
models = set()
data = {}
usage_data = {}

token_fields = [
    ("auditor_input_tokens", "Auditor input"),
    ("auditor_output_tokens", "Auditor output"),
    ("judge_input_tokens", "Judge input"),
    ("judge_output_tokens", "Judge output"),
    ("target_input_tokens", "Target input"),
    ("target_output_tokens", "Target output"),
    ("total", "Total tokens"),
]

group_fields = [
    ("Auditor I/O", "auditor_input_tokens", "auditor_output_tokens"),
    ("Judge I/O", "judge_input_tokens", "judge_output_tokens"),
    ("Target I/O", "target_input_tokens", "target_output_tokens"),
]

for scenario in scenarios:
    data[scenario] = {}
    usage_data[scenario] = {}
    for model_dir in (BASE_DIR / scenario).iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        models.add(model)
        scores = []
        usage_data[scenario][model] = {key: [] for key, _ in token_fields}
        for run_file in sorted(model_dir.glob("run_*.json")):
            with open(run_file) as f:
                obj = json.load(f)
            scores.append(obj["summary"]["score"])
            run_token_usage = {key: 0 for key, _ in token_fields}
            for result in obj.get("results", []):
                for key, _ in token_fields:
                    if key != "total":
                        run_token_usage[key] += result.get(key, 0)
            run_token_usage["total"] = sum(run_token_usage[key] for key, _ in token_fields if key != "total")
            for key, _ in token_fields:
                usage_data[scenario][model][key].append(run_token_usage[key])
        if scores:
            data[scenario][model] = scores

models = sorted(models)

def display_model(model: str) -> str:
    tidy = model.replace('Qwen_', '')
    tidy = tidy.replace('google_', '')
    return tidy

col_w = 22
scenario_w = 28

header = f"{'Scenario':<{scenario_w}}" + "".join(f"{display_model(m):<{col_w}}" for m in models)
print(header)
print("-" * (scenario_w + col_w * len(models)))

for scenario in scenarios:
    row = f"{scenario:<{scenario_w}}"
    for model in models:
        scores = data[scenario].get(model)
        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            cell = f"{mean:.2f}±{std:.2f}"
        else:
            cell = "N/A"
        row += f"{cell:<{col_w}}"
    print(row)


import math

def human_tokens(value: float) -> str:
    value = math.ceil(value)
    if value >= 1_000_000:
        return f"{math.ceil(value / 1_000_000)}M"
    if value >= 1_000:
        return f"{math.ceil(value / 1_000)}K"
    return str(value)


def format_stats(values):
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    return human_tokens(mean)


def format_io_stats(values_in, values_out):
    if not values_in or not values_out:
        return "N/A"
    mean_in = statistics.mean(values_in)
    mean_out = statistics.mean(values_out)
    return f"{human_tokens(mean_in)}/{human_tokens(mean_out)}"

print("\nCombined token usage summary (mean per run input/output):")
combined_scenario_w = 30
combined_model_w = 26
metric_w = 16
combined_header = (
    f"{'Scenario':<{combined_scenario_w}}"
    f"{'Model':<{combined_model_w}}"
    + "".join(f"{label:<{metric_w}}" for label, _, _ in group_fields)
)
print(combined_header)
print("=" * (combined_scenario_w + combined_model_w + metric_w * len(group_fields)))
for scenario in scenarios:
    for model in models:
        values_exist = usage_data.get(scenario, {}).get(model)
        if values_exist is None:
            continue
        row = f"{scenario:<{combined_scenario_w}}" + f"{display_model(model):<{combined_model_w}}"
        for label, key_in, key_out in group_fields:
            if key_out is None:
                row += f"{format_stats(values_exist.get(key_in)):<{metric_w}}"
            else:
                row += f"{format_io_stats(values_exist.get(key_in), values_exist.get(key_out)):<{metric_w}}"
        print(row)
