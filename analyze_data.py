"""
analyze_data.py  —  SimpleAudit result analysis library
========================================================
Loads results from a single comparison_results/ directory and provides:

  - Data loading (safe and unsafe runs coexist in one DataFrame)
  - Filter helpers for slicing by safety, size, hierarchy
  - Variance decomposition (partial η²) to quantify J/A/T factor impact
  - Statistical tests per factor (Kruskal-Wallis + pairwise Mann-Whitney)
  - Plotting: bar chart, heatmap, variance decomp, severity profile,
               safe-vs-unsafe comparison, calibration scatter

The _UNSAFE suffix in directory names (e.g. T=XS_UNSAFE__A=M) is parsed
into per-role boolean columns so mixed experiments (safe judge + unsafe target,
etc.) are representable in a single DataFrame.

Typical usage
-------------
import analyze_data as ad

df = ad.load_comparison_results('comparison_results/norwegian_safety')
ad.show_available_combos(df)

# Filtering
df_safe        = ad.filter_all_safe(df)
df_big_judges  = ad.filter_min_judge_size(df, 'M')
df_strict      = ad.filter_strict_hierarchy(df)
df_unsafe_tgt  = ad.filter_by_safety(df, target_safe=False, judge_safe=True)

# Analysis
ad.plot_by_axis(df_safe, x='judge', fixed={'target': 'S', 'auditor': 'M'})
ad.plot_variance_decomposition(df_strict)
ad.plot_safety_split(df, x='target', split_by='target', fixed={'judge': 'L'})
"""

import json
import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, f_oneway, pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ─── Constants ────────────────────────────────────────────────────────────────

SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL']
FACTORS    = ['judge', 'auditor', 'target']
SAFETY_COLS = ['target_safe', 'auditor_safe', 'judge_safe']

SEVERITY_LEVELS = ['critical', 'high', 'medium', 'low', 'pass_count']
SEVERITY_COLORS = {
    'critical':   '#d62728',
    'high':       '#ff7f0e',
    'medium':     '#bcbd22',
    'low':        '#2ca02c',
    'pass_count': '#aec7e8',
}
SEVERITY_LABELS = {
    'critical':   'Critical',
    'high':       'High',
    'medium':     'Medium',
    'low':        'Low',
    'pass_count': 'Pass',
}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_key(key: str) -> tuple:
    """'XS_UNSAFE' → ('XS', False),  'XS' → ('XS', True)."""
    if key.endswith('_UNSAFE'):
        return key[:-7], False
    return key, True


def _fmt_key(size: str, safe: bool) -> str:
    """Format a size key for display; append '*' for unsafe."""
    return size if safe else f'{size}*'


def _apply_fixed(df: pd.DataFrame, fixed: dict) -> pd.DataFrame:
    """Filter df by any column values; raise a helpful error on empty result."""
    mask = pd.Series(True, index=df.index)
    for col, val in fixed.items():
        mask &= (df[col] == val)
    sub = df[mask]
    if sub.empty:
        fixed_str = ', '.join(f'{k}={v}' for k, v in fixed.items())
        avail = {f: sorted(df[f].dropna().unique().tolist()) for f in FACTORS}
        raise ValueError(
            f"No data for fixed={{{fixed_str}}}.\n"
            f"Available size values: {avail}\n"
            "Tip: run show_available_combos(df) to see all valid combinations."
        )
    return sub


def _active_factors(df: pd.DataFrame) -> list:
    """Return size factors that have ≥2 levels in df."""
    return [f for f in FACTORS if df[f].nunique() >= 2]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_comparison_results(base_dir: str) -> pd.DataFrame:
    """
    Load all SimpleAudit experiment results into a per-run DataFrame.

    Crawls  base_dir/T={t}__A={a}/J={j}/experiment_results.json

    The _UNSAFE suffix in directory names is parsed into per-role boolean
    columns (target_safe, auditor_safe, judge_safe), so all experiment
    variants coexist in a single DataFrame regardless of which combination
    of safe/unsafe models was used.

    Parameters
    ----------
    base_dir : path to the merged comparison_results directory

    Returns
    -------
    DataFrame — one row per run, columns:
      target, auditor, judge          — size keys (XS/S/M/L/XL), ordered categorical
      target_safe, auditor_safe, judge_safe — bool: True=safe, False=unsafe (_UNSAFE)
      target_model, judge_model        — full model name strings
      combo                            — raw directory name, e.g. "T=XS_UNSAFE__A=M"
      run_id                           — 0-based index within this combo×judge
      score                            — aggregate score (0–100)
      passed, failed, n_scenarios      — scenario-level counts
      critical, high, medium, low, pass_count  — severity counts per run
      judge/auditor/target input/output tokens
      rejudge                          — 1 if only the judge was re-run
    """
    base = Path(base_dir)
    rows = []

    for ta_dir in sorted(base.iterdir()):
        if not ta_dir.is_dir():
            continue
        m = re.match(r'T=(\w+)__A=(\w+)', ta_dir.name)
        if not m:
            continue
        target_key,  target_safe  = _parse_key(m.group(1))
        auditor_key, auditor_safe = _parse_key(m.group(2))

        if (target_key in SIZE_ORDER and auditor_key in SIZE_ORDER and
                SIZE_ORDER.index(target_key) >= SIZE_ORDER.index(auditor_key)):
            warnings.warn(
                f"Unexpected T≥A combo in '{ta_dir.name}'. "
                "By design the auditor should be larger than the target."
            )

        for j_dir in sorted(ta_dir.iterdir()):
            if not j_dir.is_dir():
                continue
            m2 = re.match(r'J=(\w+)', j_dir.name)
            if not m2:
                continue
            judge_key, judge_safe = _parse_key(m2.group(1))

            rf = j_dir / 'experiment_results.json'
            if not rf.exists():
                continue

            try:
                with open(rf) as f:
                    doc = json.load(f)
            except Exception as e:
                warnings.warn(f"Could not read {rf}: {e}")
                continue

            judge_model = doc.get('judge', {}).get('judge_model', '?')
            runs_dict   = doc.get('runs', {})

            for model_name, run_list in runs_dict.items():
                for run_id, run in enumerate(run_list):
                    summary = run.get('summary', {})
                    sev     = summary.get('severity_distribution', {})
                    tokens  = summary.get('token_usage', {})

                    rows.append({
                        'target':        target_key,
                        'auditor':       auditor_key,
                        'judge':         judge_key,
                        'target_safe':   target_safe,
                        'auditor_safe':  auditor_safe,
                        'judge_safe':    judge_safe,
                        'target_model':  model_name,
                        'judge_model':   judge_model,
                        'combo':         ta_dir.name,
                        'run_id':        run_id,
                        'score':         summary.get('score'),
                        'passed':        summary.get('passed', 0),
                        'failed':        summary.get('failed', 0),
                        'n_scenarios':   summary.get('total_scenarios', 0),
                        'critical':      sev.get('critical', 0),
                        'high':          sev.get('high', 0),
                        'medium':        sev.get('medium', 0),
                        'low':           sev.get('low', 0),
                        'pass_count':    summary.get('passed', 0),
                        'judge_input_tokens':    tokens.get('judge_input', 0),
                        'judge_output_tokens':   tokens.get('judge_output', 0),
                        'auditor_input_tokens':  tokens.get('auditor_input', 0),
                        'auditor_output_tokens': tokens.get('auditor_output', 0),
                        'target_input_tokens':   tokens.get('target_input', 0),
                        'target_output_tokens':  tokens.get('target_output', 0),
                        'rejudge': int(tokens.get('auditor_input', 0) == 0
                                       and tokens.get('target_input', 0) == 0),
                    })

    if not rows:
        raise ValueError(f"No experiment_results.json files found under {base_dir!r}")

    df = pd.DataFrame(rows)

    for col in FACTORS:
        df[col] = pd.Categorical(df[col], categories=SIZE_ORDER, ordered=True)

    return df.reset_index(drop=True)


# ─── Filter Helpers ───────────────────────────────────────────────────────────

def filter_by_safety(df: pd.DataFrame,
                     target_safe:  Optional[bool] = None,
                     auditor_safe: Optional[bool] = None,
                     judge_safe:   Optional[bool] = None) -> pd.DataFrame:
    """
    Filter rows by safety status of each role.  None = no constraint.

    Examples
    --------
    filter_by_safety(df, target_safe=False)          # only unsafe targets
    filter_by_safety(df, judge_safe=True, auditor_safe=True)  # safe judge+auditor
    filter_by_safety(df, target_safe=False, judge_safe=True)  # unsafe target, safe judge
    """
    mask = pd.Series(True, index=df.index)
    if target_safe  is not None: mask &= df['target_safe']  == target_safe
    if auditor_safe is not None: mask &= df['auditor_safe'] == auditor_safe
    if judge_safe   is not None: mask &= df['judge_safe']   == judge_safe
    return df[mask].copy()


def filter_all_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where all three roles are safe (no _UNSAFE models)."""
    return filter_by_safety(df, target_safe=True, auditor_safe=True, judge_safe=True)


def filter_strict_hierarchy_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where judge > target (prevents self-judging / upward-bias).
    Drop rows where the judge is the same size or smaller than the target.
    """
    return df[df['judge'] > df['target']].copy()

def filter_semistrict_hierarchy_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where judge >= target (prevents self-judging / upward-bias).
    Drop rows where the judge is smaller than the target.
    """
    return df[df['judge'] >= df['target']].copy()


def filter_min_judge_size(df: pd.DataFrame, min_size: str = 'M') -> pd.DataFrame:
    """
    Keep rows where judge size >= min_size.

    Example: filter_min_judge_size(df, 'M') drops XS and S judges.
    """
    if min_size not in SIZE_ORDER:
        raise ValueError(f"min_size must be one of {SIZE_ORDER}")
    valid = SIZE_ORDER[SIZE_ORDER.index(min_size):]
    return df[df['judge'].isin(valid)].copy()


def filter_by_size(df: pd.DataFrame, factor: str, sizes: list) -> pd.DataFrame:
    """
    Keep rows where *factor* is in *sizes*.

    Example: filter_by_size(df, 'judge', ['L', 'XL'])
    """
    if factor not in df.columns:
        raise ValueError(f"Unknown factor '{factor}'. Valid: {FACTORS}")
    return df[df[factor].isin(sizes)].copy()


# ─── Exploration Helpers ──────────────────────────────────────────────────────

def show_available_combos(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of all loaded (T, A, J) combinations,
    their safety status, run counts, and mean scores.

    Size keys marked with * are unsafe (_UNSAFE) model variants.
    Run this after loading to know which values to pass to fixed={}.
    """
    group_cols = FACTORS + SAFETY_COLS
    combos = (
        df.groupby(group_cols, observed=True)
          .agg(n_runs=('run_id', 'count'),
               mean_score=('score', 'mean'),
               std_score=('score', 'std'))
          .reset_index()
    )

    has_unsafe = not df[SAFETY_COLS].all().all()

    print('=' * 72)
    print(f'  Loaded {len(df)} rows  ({len(combos)} combinations)')
    if has_unsafe:
        print('  (* = unsafe/_UNSAFE model variant)')
    print('=' * 72)
    print(f'  {"Target":>8}  {"Auditor":>9}  {"Judge":>7}  '
          f'{"Runs":>5}  {"Mean Score":>10}  {"Std":>6}')
    print('  ' + '-' * 60)
    for _, row in combos.iterrows():
        t = _fmt_key(row.target,  row.target_safe)
        a = _fmt_key(row.auditor, row.auditor_safe)
        j = _fmt_key(row.judge,   row.judge_safe)
        print(f'  T={t:<5}  A={a:<6}  J={j:<5}  '
              f'{int(row.n_runs):>5}  {row.mean_score:>10.1f}  {row.std_score:>6.2f}')

    print('\n  Available size values (safe models):')
    safe_df = filter_all_safe(df)
    for f in FACTORS:
        vals = sorted(safe_df[f].dropna().unique().tolist())
        print(f'    {f:<8} = {vals}')

    if has_unsafe:
        print('\n  Available size values (unsafe* models — per role):')
        for f in FACTORS:
            unsafe_vals = sorted(
                df[~df[f'{f}_safe']][f].dropna().unique().tolist()
            )
            if unsafe_vals:
                print(f'    {f:<8}* = {unsafe_vals}')

    print(f'\n  Valid outcome names: {get_outcomes()}')
    print('=' * 72)


def get_outcomes() -> list:
    """Return valid outcome variable names for analysis functions."""
    return (
        ['score', 'passed', 'failed',
         'critical', 'high', 'medium', 'low', 'pass_count',
         'severity',   # special: runs analysis over all severity levels
         ]
        + ['judge_input_tokens', 'judge_output_tokens',
           'auditor_input_tokens', 'auditor_output_tokens',
           'target_input_tokens', 'target_output_tokens']
    )


# ─── Statistical Analysis ─────────────────────────────────────────────────────

def _variance_decomp_single(df: pd.DataFrame, outcome: str,
                            n_bootstrap: int = 0) -> pd.DataFrame:
    """
    Internal: partial η² via Type-II OLS for one outcome variable.
    Returns DataFrame with columns: factor, partial_eta2, F, p_value, n_levels, note.
    With n_bootstrap > 0 also adds eta2_lo, eta2_hi (95% percentile CI).
    """
    active = _active_factors(df)

    result_rows = []
    for f in FACTORS:
        if f not in active:
            result_rows.append({
                'factor': f, 'partial_eta2': float('nan'),
                'F': float('nan'), 'p_value': float('nan'),
                'n_levels': int(df[f].nunique()), 'note': 'single level'
            })

    if not active:
        return pd.DataFrame(result_rows)

    formula  = f'Q("{outcome}") ~ ' + ' + '.join(f'C({f})' for f in active)
    df_clean = df[active + [outcome]].dropna()

    try:
        model       = ols(formula, data=df_clean).fit()
        anova_table = anova_lm(model, typ=2)

        point_estimates = {}
        for f in active:
            key    = f'C({f})'
            ss_eff = anova_table.loc[key, 'sum_sq']
            ss_res = anova_table.loc['Residual', 'sum_sq']
            point_estimates[f] = float(ss_eff / (ss_eff + ss_res))

        boot_samples: dict[str, list[float]] = {f: [] for f in active}
        if n_bootstrap > 0:
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                idx     = rng.integers(0, len(df_clean), size=len(df_clean))
                df_boot = df_clean.iloc[idx]
                # skip samples that dropped a level (singular design matrix)
                if any(df_boot[f].nunique() < 2 for f in active):
                    continue
                try:
                    bt = anova_lm(ols(formula, data=df_boot).fit(), typ=2)
                    for f in active:
                        key = f'C({f})'
                        ss_e = bt.loc[key, 'sum_sq']
                        ss_r = bt.loc['Residual', 'sum_sq']
                        boot_samples[f].append(float(ss_e / (ss_e + ss_r)))
                except Exception:
                    pass

        for f in active:
            row = {
                'factor':       f,
                'partial_eta2': point_estimates[f],
                'F':            float(anova_table.loc[f'C({f})', 'F']),
                'p_value':      float(anova_table.loc[f'C({f})', 'PR(>F)']),
                'n_levels':     int(df[f].nunique()),
                'note':         '',
            }
            if n_bootstrap > 0 and boot_samples[f]:
                samples = np.array(boot_samples[f])
                row['eta2_lo'] = float(np.percentile(samples, 2.5))
                row['eta2_hi'] = float(np.percentile(samples, 97.5))
            result_rows.append(row)

    except Exception as e:
        for f in active:
            result_rows.append({
                'factor': f, 'partial_eta2': float('nan'),
                'F': float('nan'), 'p_value': float('nan'),
                'n_levels': int(df[f].nunique()), 'note': str(e)
            })

    return pd.DataFrame(result_rows)


def compute_variance_decomposition(df: pd.DataFrame,
                                   outcome: str = 'score',
                                   n_bootstrap: int = 0) -> pd.DataFrame:
    """
    Compute partial η² variance decomposition for each factor (judge/auditor/target).

    Partial η² = SS_factor / (SS_factor + SS_residual) — unique contribution of each
    factor controlling for all others.  0.06=small, 0.14=medium, 0.26=large effect.

    Parameters
    ----------
    df          : DataFrame from load_comparison_results() (optionally filtered)
    outcome     : 'score', any severity level, or 'severity' (runs all levels)
    n_bootstrap : number of bootstrap resamples for 95% CI on η² (0 = no CI)

    Returns
    -------
    DataFrame with factor, partial_eta2, F, p_value, n_levels, note.
    With n_bootstrap > 0: also eta2_lo, eta2_hi columns.
    When outcome='severity': also has 'severity_level' column.
    """
    if outcome == 'severity':
        frames = []
        for lvl in SEVERITY_LEVELS:
            frame = _variance_decomp_single(df, lvl, n_bootstrap=n_bootstrap)
            frame.insert(0, 'severity_level', lvl)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    return _variance_decomp_single(df, outcome, n_bootstrap=n_bootstrap)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_by_axis(df: pd.DataFrame,
                 x: str = 'judge',
                 fixed: dict = {},
                 outcome: str = 'score',
                 ci: bool = True) -> plt.Figure:
    """
    Bar chart: mean outcome by factor *x*, with optional fixed constraints.

    Parameters
    ----------
    x       : factor to put on x-axis: 'judge', 'auditor', or 'target'
    fixed   : dict of column→value to hold constant, e.g. {'target':'S', 'auditor':'M'}
    outcome : column to plot (see get_outcomes())
    ci      : show ±1 std error bars

    Examples
    --------
    plot_by_axis(df, x='judge', fixed={'target': 'S', 'auditor': 'M'})
    plot_by_axis(df, x='auditor', fixed={'judge': 'XL'}, outcome='critical')
    """
    sub = _apply_fixed(df, fixed) if fixed else df.copy()

    levels = [l for l in SIZE_ORDER if l in sub[x].values] \
             if x in FACTORS else sorted(sub[x].unique())

    grouped    = sub.groupby(x, observed=False)[outcome]
    means_all  = grouped.mean()
    stds_all   = grouped.std()
    counts_all = grouped.count()

    means  = [means_all.get(l, np.nan)  for l in levels]
    stds   = [stds_all.get(l, np.nan)   for l in levels]
    counts = [int(counts_all.get(l, 0)) for l in levels]

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.5 + 1), 5))
    xs = np.arange(len(levels))

    bar_colors = ['steelblue' if not np.isnan(m) else '#cccccc' for m in means]
    yerr       = [s if not np.isnan(s) else 0 for s in stds] if ci else None
    bar_heights = [m if not np.isnan(m) else 0 for m in means]

    ax.bar(xs, bar_heights, yerr=yerr, capsize=6,
           color=bar_colors, alpha=0.8,
           error_kw=dict(elinewidth=1.5, ecolor='black'))

    for i, (m, s, n) in enumerate(zip(means, stds, counts)):
        if np.isnan(m):
            ax.text(i, 2, 'n/a', ha='center', va='bottom', fontsize=9, color='gray')
        else:
            top = m + (s if not np.isnan(s) else 0)
            ax.text(i, top + 1.5, f'{m:.1f}\n±{s:.1f}\n(n={n})',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x[0].upper()}={l}' for l in levels])
    ax.set_xlabel(f'{x.capitalize()} model size', fontsize=11)
    ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=11)

    fixed_str = ', '.join(f'{k}={v}' for k, v in fixed.items()) if fixed else 'all data'
    ax.set_title(f'{outcome} by {x}  |  {fixed_str}', fontsize=12)

    valid = [(m, s) for m, s in zip(means, stds) if not np.isnan(m)]
    if valid:
        top = max(m + (s if not np.isnan(s) else 0) for m, s in valid)
        ax.set_ylim(0, top * 1.4)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame,
                 x: str = 'target',
                 y: str = 'auditor',
                 fixed: dict = {},
                 outcome: str = 'score', show_title: bool = True) -> plt.Figure:
    """
    2D heatmap: mean outcome for each (x, y) combination.
    Missing cells shown in gray with '×'.

    Parameters
    ----------
    x, y    : factors for the two axes (remaining factor goes in fixed or is marginalized)
    fixed   : dict of column→value, e.g. {'judge': 'XL'}
    show_title : whether to display the title

    Examples
    --------
    plot_heatmap(df, x='target', y='auditor', fixed={'judge': 'XL'})
    plot_heatmap(df, x='judge', y='target',   fixed={'auditor': 'M'})
    """
    sub = _apply_fixed(df, fixed) if fixed else df.copy()

    x_vals = [l for l in SIZE_ORDER if l in sub[x].values] \
             if x in FACTORS else sorted(sub[x].unique())
    y_vals = [l for l in SIZE_ORDER if l in sub[y].values] \
             if y in FACTORS else sorted(sub[y].unique())

    matrix   = pd.DataFrame(np.nan, index=y_vals, columns=x_vals, dtype=float)
    n_matrix = pd.DataFrame(0,      index=y_vals, columns=x_vals, dtype=int)

    for xi in x_vals:
        for yi in y_vals:
            cell = sub[(sub[x] == xi) & (sub[y] == yi)][outcome].dropna()
            if len(cell) > 0:
                matrix.loc[yi, xi]   = cell.mean()
                n_matrix.loc[yi, xi] = len(cell)

    fig, ax = plt.subplots(figsize=(max(5, len(x_vals) * 1.6 + 1),
                                    max(4, len(y_vals) * 1.3 + 1)))

    vmax = None
    sns.heatmap(
        matrix.astype(float), ax=ax, cmap='RdYlGn_r',
        mask=matrix.isna(), linewidths=0.5, linecolor='white',
        vmin=0, vmax=vmax, cbar_kws={'shrink': 0.8},
        annot=False,
    )

    for i, yi in enumerate(y_vals):
        for j, xi in enumerate(x_vals):
            val = matrix.loc[yi, xi]
            n   = n_matrix.loc[yi, xi]
            if pd.isna(val):
                ax.text(j + 0.5, i + 0.5, '×', ha='center', va='center',
                        fontsize=16, color='lightgray')
            else:
                text_color = 'white' if val > 65 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{val:.1f}\n(n={n})',
                        ha='center', va='center', fontsize=9, color=text_color)

    ax.set_xlabel(f'{x.capitalize()} size', fontsize=11)
    ax.set_ylabel(f'{y.capitalize()} size', fontsize=11)
    fixed_str = ', '.join(f'{k}={v}' for k, v in fixed.items()) if fixed else 'all data'
    if show_title:
        ax.set_title(f'{outcome} — {x} × {y}  |  {fixed_str}', fontsize=12)
    plt.tight_layout()
    return fig


def _draw_eta2_bars(result: pd.DataFrame, ax: plt.Axes, title: str = '') -> None:
    """Draw partial η² horizontal bars onto ax (internal helper)."""
    factor_rows = result[result['factor'].isin(FACTORS)].copy()

    if factor_rows['partial_eta2'].isna().all():
        ax.text(0.5, 0.5, 'Insufficient data\n(need ≥2 levels per factor)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray', style='italic')
        ax.set_title(title, fontsize=10)
        return

    factor_rows = factor_rows.sort_values('partial_eta2', ascending=True, na_position='first')
    colors_map  = {'judge': '#4878CF', 'auditor': '#D65F5F', 'target': '#6ACC65'}

    has_ci = 'eta2_lo' in factor_rows.columns and 'eta2_hi' in factor_rows.columns

    for i, row in enumerate(factor_rows.itertuples()):
        eta2 = row.partial_eta2
        if not np.isnan(eta2):
            ax.barh(i, eta2, color=colors_map.get(row.factor, 'gray'), alpha=0.85)
            if has_ci and not np.isnan(row.eta2_lo) and not np.isnan(row.eta2_hi):
                ax.errorbar(eta2, i,
                            xerr=[[eta2 - row.eta2_lo], [row.eta2_hi - eta2]],
                            fmt='none', color='black', capsize=4, linewidth=1.2)
            label_x = row.eta2_hi + 0.01 if has_ci and not np.isnan(row.eta2_hi) else eta2 + 0.01
            ax.text(label_x, i, f'{eta2:.3f}', va='center', fontsize=9)
        else:
            note = row.note if hasattr(row, 'note') else ''
            ax.text(0.02, i, f'n/a  ({note})', va='center', fontsize=8, color='gray')

    ax.set_yticks(range(len(factor_rows)))
    ax.set_yticklabels([r.factor.capitalize() for r in factor_rows.itertuples()])
    ax.set_xlabel('Partial η²', fontsize=10)
    ax.set_xlim(0, 1.0)
    # for thresh, label, ls in [(0.06, 'small', ':'), (0.14, 'medium', '--'), (0.26, 'large', '-.')]:
    #     ax.axvline(thresh, color='gray', linestyle=ls, alpha=0.45, linewidth=1)
    #     ax.text(thresh + 0.005, len(factor_rows) - 0.05, label,
    #             fontsize=7, color='gray', va='top')
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.3)


def plot_variance_decomposition(df: pd.DataFrame,
                                outcome: str = 'score',
                                n_bootstrap: int = 0, show_title: bool = True) -> plt.Figure:
    """
    Horizontal bar chart: partial η² for judge, auditor, and target.

    Partial η² measures unique variance explained by each factor controlling
    for the others (0=no effect, 1=explains everything).
    Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant.

    Parameters
    ----------
    n_bootstrap : bootstrap resamples for 95% CI whiskers (0 = no CI, ~1000 recommended)

    If outcome='severity', produces a faceted plot — one panel per severity level.
    """
    result = compute_variance_decomposition(df, outcome, n_bootstrap=n_bootstrap)

    if outcome == 'severity':
        n_lvls = len(SEVERITY_LEVELS)
        fig, axes = plt.subplots(1, n_lvls,
                                 figsize=(3.5 * n_lvls, 4), sharey=True)
        for ax, lvl in zip(axes, SEVERITY_LEVELS):
            sub = result[result['severity_level'] == lvl]
            _draw_eta2_bars(sub, ax, title=SEVERITY_LABELS.get(lvl, lvl))
        if show_title:
            fig.suptitle('Partial η² by factor — severity levels', fontsize=13, y=1.01)
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if show_title:
        _draw_eta2_bars(result, ax, title=f'Variance decomposition — {outcome}')
    else:
        _draw_eta2_bars(result, ax)
    plt.tight_layout()
    return fig


def plot_severity_profile(df: pd.DataFrame,
                          group_by: str = 'judge',
                          fixed: dict = {}) -> plt.Figure:
    """
    Stacked bar chart: mean severity distribution per group.

    Two settings with the same mean score can differ in severity profile
    (one finds 'critical' issues, another finds only 'medium'). A matching
    stack shape indicates genuine agreement, not just score coincidence.

    Parameters
    ----------
    group_by : factor to group by on x-axis ('judge', 'auditor', 'target')
    fixed    : constraints on other factors

    Example
    -------
    plot_severity_profile(df, group_by='judge', fixed={'target':'S', 'auditor':'M'})
    """
    sub = _apply_fixed(df, fixed) if fixed else df.copy()

    levels = [l for l in SIZE_ORDER if l in sub[group_by].values] \
             if group_by in FACTORS else sorted(sub[group_by].unique())

    sev_means = {
        lvl: {sev: sub[sub[group_by] == lvl][sev].mean()
              for sev in SEVERITY_LEVELS}
        for lvl in levels
    }

    fig, (ax_stack, ax_score) = plt.subplots(
        2, 1, figsize=(max(6, len(levels) * 1.5 + 1), 7),
        gridspec_kw={'height_ratios': [3, 1]},
    )

    xs = np.arange(len(levels))
    bottoms = np.zeros(len(levels))

    for sev in SEVERITY_LEVELS:
        vals  = np.array([sev_means[l][sev] for l in levels])
        label = SEVERITY_LABELS.get(sev, sev)
        ax_stack.bar(xs, vals, bottom=bottoms,
                     label=label, color=SEVERITY_COLORS[sev], alpha=0.85)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.4:
                ax_stack.text(i, b + v / 2, f'{v:.1f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if sev == 'critical' else 'black')
        bottoms += vals

    ax_stack.set_xticks(xs)
    ax_stack.set_xticklabels([f'{group_by[0].upper()}={l}' for l in levels])
    ax_stack.set_ylabel('Mean count per run', fontsize=11)
    ax_stack.legend(loc='upper right', fontsize=9)
    ax_stack.grid(axis='y', linestyle='--', alpha=0.3)

    score_means = [sub[sub[group_by] == l]['score'].mean() for l in levels]
    score_stds  = [sub[sub[group_by] == l]['score'].std()  for l in levels]
    ax_score.bar(xs, score_means, yerr=score_stds, capsize=5,
                 color='steelblue', alpha=0.75,
                 error_kw=dict(elinewidth=1.5, ecolor='black'))
    for i, (m, s) in enumerate(zip(score_means, score_stds)):
        ax_score.text(i, m + s + 0.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8)
    ax_score.set_xticks(xs)
    ax_score.set_xticklabels([f'{group_by[0].upper()}={l}' for l in levels])
    ax_score.set_ylabel('Mean score', fontsize=10)
    ax_score.set_ylim(0, max(score_means) * 1.5 if score_means else 100)
    ax_score.grid(axis='y', linestyle='--', alpha=0.3)

    fixed_str = ', '.join(f'{k}={v}' for k, v in fixed.items()) if fixed else 'all data'
    fig.suptitle(f'Severity profile by {group_by}  |  {fixed_str}', fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


# ─── Safe vs Unsafe Comparison ────────────────────────────────────────────────

def add_safety_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Tag a DataFrame with a 'safety' column ('safe' or 'unsafe').
    Useful for manually combining two pre-filtered slices."""
    df = df.copy()
    df['safety'] = label
    return df


def plot_safety_split(df: pd.DataFrame,
                      x: str = 'target',
                      split_by: str = 'target',
                      fixed: dict = {},
                      outcome: str = 'score') -> plt.Figure:
    """
    Grouped bar chart comparing safe vs unsafe variants for the *split_by* role.

    Works directly on the merged DataFrame — no need to load two separate files.

    Parameters
    ----------
    df       : DataFrame from load_comparison_results()
    x        : factor to put on x-axis ('judge', 'auditor', or 'target')
    split_by : which role to split on ('target', 'auditor', or 'judge')
    fixed    : dict of column→value, e.g. {'judge': 'M', 'auditor': 'M'}
    outcome  : column to plot

    Example
    -------
    plot_safety_split(df, x='target', split_by='target',
                      fixed={'auditor': 'M', 'judge': 'L'})
    """
    safe_col = f'{split_by}_safe'
    if safe_col not in df.columns:
        raise ValueError(f"'{safe_col}' not in df. split_by must be one of {FACTORS}")

    df_safe   = add_safety_label(df[df[safe_col] == True],  'safe')
    df_unsafe = add_safety_label(df[df[safe_col] == False], 'unsafe')
    df_combined = pd.concat([df_safe, df_unsafe], ignore_index=True)

    return plot_safe_unsafe_comparison(df_combined, x=x, fixed=fixed, outcome=outcome)


def plot_safe_unsafe_comparison(df_combined: pd.DataFrame,
                                x: str = 'target',
                                fixed: dict = {},
                                outcome: str = 'score') -> plt.Figure:
    """
    Grouped bar chart: safe (blue) vs unsafe (red) side-by-side for each level of *x*.

    Expects a 'safety' column ('safe'/'unsafe') on df_combined.
    Build it with add_safety_label() or use the convenience wrapper plot_safety_split().

    Parameters
    ----------
    df_combined : concat of safe + unsafe DataFrames, each tagged with 'safety' column
    x           : factor to put on x-axis ('judge', 'auditor', or 'target')
    fixed       : dict of column→value, e.g. {'judge': 'M'}
    outcome     : column to plot

    Example
    -------
    df_s = ad.add_safety_label(ad.filter_by_safety(df, target_safe=True),  'safe')
    df_u = ad.add_safety_label(ad.filter_by_safety(df, target_safe=False), 'unsafe')
    ad.plot_safe_unsafe_comparison(pd.concat([df_s, df_u]), x='target',
                                    fixed={'auditor': 'M', 'judge': 'L'})
    """
    sub = _apply_fixed(df_combined, fixed) if fixed else df_combined.copy()

    levels = [l for l in SIZE_ORDER if l in sub[x].values] \
             if x in FACTORS else sorted(sub[x].unique())

    safety_labels = ['safe', 'unsafe']
    colors        = {'safe': '#4878CF', 'unsafe': '#D65F5F'}
    bar_width     = 0.35
    xs            = np.arange(len(levels))

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 2 + 1), 5))

    for i_s, slabel in enumerate(safety_labels):
        sub_s = sub[sub['safety'] == slabel]
        means, stds = [], []
        for lvl in levels:
            vals = sub_s[sub_s[x] == lvl][outcome].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std()   if len(vals) else np.nan)

        offsets = xs + (i_s - 0.5) * bar_width
        bar_h   = [m if not np.isnan(m) else 0 for m in means]
        yerr    = [s if not np.isnan(s) else 0 for s in stds]

        ax.bar(offsets, bar_h, bar_width,
               yerr=yerr, capsize=5,
               color=colors[slabel], alpha=0.8,
               label=slabel.capitalize(),
               error_kw=dict(elinewidth=1.5, ecolor='black'))

        for j, (m, s) in enumerate(zip(means, stds)):
            if np.isnan(m):
                ax.text(offsets[j], 1, 'n/a',
                        ha='center', va='bottom', fontsize=8, color='gray')
            else:
                top = m + (s if not np.isnan(s) else 0)
                ax.text(offsets[j], top + 1.5, f'{m:.1f}',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x[0].upper()}={l}' for l in levels])
    ax.set_xlabel(f'{x.capitalize()} model size', fontsize=11)
    ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=11)
    fixed_str = ', '.join(f'{k}={v}' for k, v in fixed.items()) if fixed else 'all data'
    ax.set_title(f'{outcome} — safe vs unsafe  |  {fixed_str}', fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# ─── Scenario-Level Analysis ──────────────────────────────────────────────────

SEVERITY_ORDER   = ['pass', 'low', 'medium', 'high', 'critical']
SEVERITY_ORDINAL = {s: i for i, s in enumerate(SEVERITY_ORDER)}


def load_scenario_results(base_dir: str) -> pd.DataFrame:
    """
    Load per-scenario, per-run data from experiment_results.json files.

    Returns one row per (target, auditor, judge, run_id, scenario_name).
    The _UNSAFE suffix is parsed into target_safe, auditor_safe, judge_safe columns.

    Columns
    -------
    target, auditor, judge   — size keys, ordered categorical
    target_safe, auditor_safe, judge_safe — bool safety flags
    run_id                   — 0-based run index
    scenario_name            — e.g. 'GDPR and Personvern Compliance'
    severity                 — categorical: pass < low < medium < high < critical
    severity_ordinal         — integer 0–4 (pass=0, critical=4)
    n_issues                 — length of issues_found list
    passed_scenario          — bool (severity == 'pass')

    Rows where the judge returned an ERROR are dropped with a warning.
    """
    base = Path(base_dir)
    rows = []
    n_errors = 0

    for ta_dir in sorted(base.iterdir()):
        if not ta_dir.is_dir():
            continue
        m = re.match(r'T=(\w+)__A=(\w+)', ta_dir.name)
        if not m:
            continue
        target_key,  target_safe  = _parse_key(m.group(1))
        auditor_key, auditor_safe = _parse_key(m.group(2))

        for j_dir in sorted(ta_dir.iterdir()):
            if not j_dir.is_dir():
                continue
            m2 = re.match(r'J=(\w+)', j_dir.name)
            if not m2:
                continue
            judge_key, judge_safe = _parse_key(m2.group(1))

            rf = j_dir / 'experiment_results.json'
            if not rf.exists():
                continue

            try:
                with open(rf) as f:
                    doc = json.load(f)
            except Exception as e:
                warnings.warn(f"Could not read {rf}: {e}")
                continue

            runs_dict = doc.get('runs', {})
            for model_name, run_list in runs_dict.items():
                for run_id, run in enumerate(run_list):
                    for scen in run.get('results', []):
                        sev = scen.get('severity', 'pass')
                        if sev == 'ERROR':
                            n_errors += 1
                            continue
                        rows.append({
                            'target':          target_key,
                            'auditor':         auditor_key,
                            'judge':           judge_key,
                            'target_safe':     target_safe,
                            'auditor_safe':    auditor_safe,
                            'judge_safe':      judge_safe,
                            'run_id':          run_id,
                            'scenario_name':   scen.get('scenario_name', '?'),
                            'severity':        sev,
                            'severity_ordinal': SEVERITY_ORDINAL.get(sev, 0),
                            'n_issues':        len(scen.get('issues_found', [])),
                            'passed_scenario': sev == 'pass',
                        })

    if n_errors:
        warnings.warn(f"Dropped {n_errors} ERROR rows from scenario results.")

    if not rows:
        raise ValueError(f"No scenario data found under {base_dir!r}")

    df = pd.DataFrame(rows)
    for col in FACTORS:
        df[col] = pd.Categorical(df[col], categories=SIZE_ORDER, ordered=True)
    df['severity'] = pd.Categorical(df['severity'],
                                    categories=SEVERITY_ORDER, ordered=True)
    return df.reset_index(drop=True)


def compute_severity_confusion(df_scen: pd.DataFrame,
                               judge_a: str,
                               judge_b: str,
                               fixed: dict = {}) -> pd.DataFrame:
    """
    Match scenarios judged by both judge_a and judge_b on identical
    (target, auditor, run_id, scenario_name) keys, then compare severity.

    Returns a merged DataFrame with columns:
      sev_a, ord_a       — severity / ordinal from judge_a (reference)
      sev_b, ord_b       — severity / ordinal from judge_b
      ordinal_diff       — ord_b − ord_a  (negative = judge_b rates lower)
      agreement          — exact severity match
      adjacent           — |ordinal_diff| ≤ 1 (tolerable)
      critical_miss      — one judge says critical/high (≥3), other says low/pass (≤1)
    """
    sub = _apply_fixed(df_scen, fixed) if fixed else df_scen.copy()

    cols = ['target', 'auditor', 'run_id', 'scenario_name',
            'severity', 'severity_ordinal']

    df_a = (sub[sub['judge'] == judge_a][cols]
            .rename(columns={'severity': 'sev_a', 'severity_ordinal': 'ord_a'}))
    df_b = (sub[sub['judge'] == judge_b][cols]
            .rename(columns={'severity': 'sev_b', 'severity_ordinal': 'ord_b'}))

    merged = pd.merge(df_a, df_b,
                      on=['target', 'auditor', 'run_id', 'scenario_name'],
                      how='inner')

    merged['ordinal_diff']    = merged['ord_b'] - merged['ord_a']
    merged['agreement']       = merged['sev_a'] == merged['sev_b']
    merged['adjacent']        = merged['ordinal_diff'].abs() <= 1
    merged['true_adjacent']   = merged['ordinal_diff'].abs() == 1
    merged['critical_miss']   = (
        ((merged['ord_a'] >= 3) & (merged['ord_b'] <= 1)) |
        ((merged['ord_b'] >= 3) & (merged['ord_a'] <= 1))
    )
    # wrong by more than ±1 but not spanning the critical/high ↔ low/pass gap
    merged['non_critical_miss'] = ~merged['agreement'] & ~merged['true_adjacent'] & ~merged['critical_miss']
    return merged


_RATE_COLS = [
    'exact_agreement_rate',
    'true_adjacent_rate',
    'non_critical_miss_rate',
    'critical_miss_rate',
    'downgrade_rate',
    'upgrade_rate',
]

def _compute_rates(merged: pd.DataFrame) -> dict:
    return {
        'exact_agreement_rate':   float(merged['agreement'].mean()),
        'true_adjacent_rate':     float(merged['true_adjacent'].mean()),
        'non_critical_miss_rate': float(merged['non_critical_miss'].mean()),
        'critical_miss_rate':     float(merged['critical_miss'].mean()),
        'downgrade_rate':         float((merged['ordinal_diff'] < 0).mean()),
        'upgrade_rate':           float((merged['ordinal_diff'] > 0).mean()),
    }


def _compute_self_consistency_row(df_scen: pd.DataFrame,
                                   reference: str,
                                   n_bootstrap: int) -> dict:
    """
    Build all within-group run pairs for the reference judge and compute the
    same agreement metrics used in _compute_rates.  Each (target, auditor,
    scenario_name) group contributes C(k, 2) pairs where k = number of runs.
    downgrade/upgrade are included for API symmetry but are ~equal by construction.
    """
    ref_data = df_scen[df_scen['judge'] == reference][
        ['target', 'auditor', 'run_id', 'scenario_name', 'severity', 'severity_ordinal']
    ]
    pair_rows = []
    for _, grp in ref_data.groupby(['target', 'auditor', 'scenario_name']):
        runs = grp[['severity', 'severity_ordinal']].values  # shape (k, 2)
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                pair_rows.append({'sev_a': runs[i, 0], 'ord_a': runs[i, 1],
                                  'sev_b': runs[j, 0], 'ord_b': runs[j, 1]})

    if not pair_rows:
        return {}

    pairs = pd.DataFrame(pair_rows)
    pairs = pairs.astype({'ord_a': int, 'ord_b': int})
    pairs['ordinal_diff']      = pairs['ord_b'] - pairs['ord_a']
    pairs['agreement']         = pairs['sev_a'] == pairs['sev_b']
    pairs['true_adjacent']     = pairs['ordinal_diff'].abs() == 1
    pairs['critical_miss']     = (
        ((pairs['ord_a'] >= 3) & (pairs['ord_b'] <= 1)) |
        ((pairs['ord_b'] >= 3) & (pairs['ord_a'] <= 1))
    )
    pairs['non_critical_miss'] = ~pairs['agreement'] & ~pairs['true_adjacent'] & ~pairs['critical_miss']

    n   = len(pairs)
    row = {'judge': f'{reference} (self)', 'n_matched_scenarios': n,
           **_compute_rates(pairs)}

    if n_bootstrap > 0:
        boot = np.array([
            list(_compute_rates(pairs.sample(n, replace=True)).values())
            for _ in range(n_bootstrap)
        ])
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        for k, col in enumerate(_RATE_COLS):
            row[col + '_lo'] = float(lo[k])
            row[col + '_hi'] = float(hi[k])

    return row


def compute_critical_miss_rates(df_scen: pd.DataFrame,
                                reference: str = 'XL',
                                n_bootstrap: int = 1000,
                                include_self_consistency: bool = False) -> pd.DataFrame:
    """
    For each judge (other than *reference*), compute scenario-level agreement
    rates against the reference judge on matched (T, A, run_id, scenario) tuples.

    Returns DataFrame with columns:
      judge, n_matched_scenarios
      exact_agreement_rate     — fraction of exactly matching severities
      true_adjacent_rate       — fraction off by exactly ±1 severity level (excludes exact)
      non_critical_miss_rate   — fraction wrong by >1 but not spanning critical/high ↔ low/pass
      critical_miss_rate       — fraction where one says ≥high, other says ≤low
      downgrade_rate           — judge rates lower than reference
      upgrade_rate             — judge rates higher than reference

    When n_bootstrap > 0, also adds <col>_lo and <col>_hi columns with 95% CI bounds
    computed by resampling matched pairs with replacement.

    When include_self_consistency=True, appends a '{reference} (self)' row
    computed from all pairwise run comparisons of the reference judge on the
    same (target, auditor, scenario) — a natural upper-bound baseline.
    """
    results = []
    judges  = [j for j in SIZE_ORDER
               if j in df_scen['judge'].values and j != reference]

    for j in judges:
        merged = compute_severity_confusion(df_scen, reference, j)
        if merged.empty:
            continue
        n   = len(merged)
        row = {'judge': j, 'n_matched_scenarios': n, **_compute_rates(merged)}

        if n_bootstrap > 0:
            boot = np.array([
                list(_compute_rates(merged.sample(n, replace=True)).values())
                for _ in range(n_bootstrap)
            ])  # shape (n_bootstrap, n_rate_cols)
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
            for k, col in enumerate(_RATE_COLS):
                row[col + '_lo'] = float(lo[k])
                row[col + '_hi'] = float(hi[k])

        results.append(row)

    if include_self_consistency:
        self_row = _compute_self_consistency_row(df_scen, reference, n_bootstrap)
        if self_row:
            results.append(self_row)

    return pd.DataFrame(results)


def compute_scenario_profiles(df_scen: pd.DataFrame,
                              fixed: dict = {}) -> pd.DataFrame:
    """
    For each scenario, compute aggregate severity statistics across all
    (target, auditor, judge, run_id) combinations in the filtered data.

    Returns DataFrame (sorted by mean_ordinal descending) with:
      scenario_name, mean_ordinal, std_ordinal,
      modal_severity, pct_critical_high, pct_pass, n_observations
    """
    sub = _apply_fixed(df_scen, fixed) if fixed else df_scen.copy()
    results = []
    for scen, grp in sub.groupby('scenario_name'):
        ord_vals = grp['severity_ordinal'].values
        results.append({
            'scenario_name':     scen,
            'mean_ordinal':      float(np.mean(ord_vals)),
            'std_ordinal':       float(np.std(ord_vals)),
            'modal_severity':    grp['severity'].mode().iloc[0],
            'pct_critical_high': float((grp['severity_ordinal'] >= 3).mean()),
            'pct_pass':          float((grp['severity'] == 'pass').mean()),
            'n_observations':    len(grp),
        })
    return (pd.DataFrame(results)
            .sort_values('mean_ordinal', ascending=False)
            .reset_index(drop=True))


def plot_critical_miss_summary(df_scen: pd.DataFrame,
                               reference: str = 'XL',
                               n_bootstrap: int = 1000,
                               include_self_judge: bool = False) -> plt.Figure:
    """
    Grouped bar chart: exact / off-by-±1 / non-critical-miss / critical-miss rates
    for each judge compared to the reference judge at the scenario level.
    Pass n_bootstrap=0 to omit error bars.

    When include_self_judge=True, appends a '{reference} (self)' bar group showing
    the reference judge's own run-to-run consistency — a natural upper-bound baseline.
    That group is rendered with hatching and separated by a vertical dashed line.
    """
    rates = compute_critical_miss_rates(df_scen, reference, n_bootstrap=n_bootstrap,
                                        include_self_consistency=include_self_judge)
    has_ci = n_bootstrap > 0

    if rates.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'No data (need J={reference} and ≥1 other judge)',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return fig

    judges    = rates['judge'].tolist()
    self_label = f'{reference} (self)'
    is_self   = [j == self_label for j in judges]
    xs        = np.arange(len(judges))
    w         = 0.20

    fig, ax = plt.subplots(figsize=(max(8, len(judges) * 2.5 + 1), 5))

    bars = [
        ('exact_agreement_rate',   'Exact match',          '#2ca02c'),
        ('true_adjacent_rate',     'Off by ±1 level',      '#bcbd22'),
        ('non_critical_miss_rate', 'Wrong (non-critical)', '#ff7f0e'),
        ('critical_miss_rate',     'Critical miss',        '#d62728'),
    ]
    for i_col, (col, label, color) in enumerate(bars):
        offset = (i_col - 1.5) * w
        vals = rates[col].values
        yerr = None
        if has_ci:
            yerr = np.array([
                vals - rates[col + '_lo'].values,
                rates[col + '_hi'].values - vals,
            ])
        for idx, (xi, v, self_bar) in enumerate(zip(xs + offset, vals, is_self)):
            this_yerr = ([[yerr[0][idx]], [yerr[1][idx]]] if has_ci else None)
            hatch = '///' if self_bar else None
            ax.bar(xi, v, w, color=color, alpha=0.82, hatch=hatch,
                   label=label if idx == 0 else '_nolegend_',
                   yerr=this_yerr,
                   error_kw=dict(ecolor='black', capsize=3, linewidth=1))
            top = v + (yerr[1][idx] if has_ci else 0) + 0.015
            ax.text(xi, top, f'{v:.0%}',
                    ha='center', va='bottom', fontsize=8)

    # separator line before the self-consistency group
    if include_self_judge and any(is_self):
        self_idx = next(i for i, s in enumerate(is_self) if s)
        ax.axvline(self_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xticks(xs)
    ns = rates['n_matched_scenarios'].values
    xlabels = []
    for j, n in zip(judges, ns):
        if j == self_label:
            xlabels.append(f'J={reference} (self)\n(N={n:,} pairs)')
        else:
            xlabels.append(f'J={j}\n(N={n:,})')
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    title = f'Scenario-level agreement vs reference J={reference}'
    if has_ci:
        title += f'  (95% CI, B={n_bootstrap})'
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def _compute_cross_run_confusion(df_scen: pd.DataFrame,
                                  judge_ref: str,
                                  judge_other: str) -> pd.DataFrame:
    """
    For each (target, auditor, scenario_name), cross-join all runs judged by
    judge_ref with all runs judged by judge_other, keeping only pairs where
    run_id differs (different conversations).  When judge_ref == judge_other,
    symmetric pairs are deduplicated so each unordered pair appears once.

    Returns a DataFrame with the same schema as compute_severity_confusion
    (sev_a/ord_a from judge_ref, sev_b/ord_b from judge_other, ordinal_diff,
    agreement, true_adjacent, critical_miss, non_critical_miss).
    """
    cols = ['target', 'auditor', 'run_id', 'scenario_name', 'severity', 'severity_ordinal']

    df_ref = df_scen[df_scen['judge'] == judge_ref][cols].rename(
        columns={'run_id': 'run_id_a', 'severity': 'sev_a', 'severity_ordinal': 'ord_a'})
    df_oth = df_scen[df_scen['judge'] == judge_other][cols].rename(
        columns={'run_id': 'run_id_b', 'severity': 'sev_b', 'severity_ordinal': 'ord_b'})

    merged = pd.merge(df_ref, df_oth, on=['target', 'auditor', 'scenario_name'], how='inner')
    merged = merged[merged['run_id_a'] != merged['run_id_b']]

    if judge_ref == judge_other:
        merged = merged[merged['run_id_a'] < merged['run_id_b']]

    merged = merged.astype({'ord_a': int, 'ord_b': int})
    merged['ordinal_diff']      = merged['ord_b'] - merged['ord_a']
    merged['agreement']         = merged['sev_a'] == merged['sev_b']
    merged['true_adjacent']     = merged['ordinal_diff'].abs() == 1
    merged['critical_miss']     = (
        ((merged['ord_a'] >= 3) & (merged['ord_b'] <= 1)) |
        ((merged['ord_b'] >= 3) & (merged['ord_a'] <= 1))
    )
    merged['non_critical_miss'] = ~merged['agreement'] & ~merged['true_adjacent'] & ~merged['critical_miss']
    return merged


def compute_critical_miss_rates2(df_scen: pd.DataFrame,
                                  reference: str = 'XL',
                                  n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Unified cross-run version of compute_critical_miss_rates.

    Unlike compute_critical_miss_rates, which compares each judge J against the
    reference judge on the *same* conversation (row-wise join on run_id), this
    function uses *cross-run* pairs throughout: for every (target, auditor,
    scenario_name) group it pairs every reference run with every other-judge run
    that has a *different* run_id.  The reference's own self-consistency row uses
    the same cross-run logic (all C(k,2) run pairs within the reference judge).

    This puts all judges — including the self-consistency baseline — on the same
    footing: every comparison spans two *different* conversations for the same
    experimental setting.  The self-consistency row therefore acts as a noise
    floor capturing conversation-level stochasticity, and any judge whose bars
    sit above that floor shows disagreement beyond what conversation randomness
    alone explains.

    Methodological trade-off: cross-run pairs for other judges confound judge
    disagreement with conversation variability, so apparent disagreement rates
    will be higher than in compute_critical_miss_rates.  The gain is that the
    self-consistency baseline is directly interpretable alongside the other bars.
    """
    results = []
    judges = [j for j in SIZE_ORDER if j in df_scen['judge'].values and j != reference]

    for j in judges:
        pairs = _compute_cross_run_confusion(df_scen, reference, j)
        if pairs.empty:
            continue
        n   = len(pairs)
        row = {'judge': j, 'n_matched_scenarios': n, **_compute_rates(pairs)}
        if n_bootstrap > 0:
            boot = np.array([
                list(_compute_rates(pairs.sample(n, replace=True)).values())
                for _ in range(n_bootstrap)
            ])
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
            for k, col in enumerate(_RATE_COLS):
                row[col + '_lo'] = float(lo[k])
                row[col + '_hi'] = float(hi[k])
        results.append(row)

    self_pairs = _compute_cross_run_confusion(df_scen, reference, reference)
    if not self_pairs.empty:
        n   = len(self_pairs)
        row = {'judge': f'{reference} (self)', 'n_matched_scenarios': n,
               **_compute_rates(self_pairs)}
        if n_bootstrap > 0:
            boot = np.array([
                list(_compute_rates(self_pairs.sample(n, replace=True)).values())
                for _ in range(n_bootstrap)
            ])
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
            for k, col in enumerate(_RATE_COLS):
                row[col + '_lo'] = float(lo[k])
                row[col + '_hi'] = float(hi[k])
        results.append(row)

    return pd.DataFrame(results)


def plot_critical_miss_summary2(df_scen: pd.DataFrame,
                                 reference: str = 'XL',
                                 n_bootstrap: int = 1000, show_title: bool = True) -> plt.Figure:
    """
    Grouped bar chart of scenario-level agreement rates — unified cross-run variant.

    Differs from plot_critical_miss_summary in one key respect: all comparisons
    (including the self-consistency baseline) use *cross-run* pairs, i.e. two
    different conversations for the same (target, auditor, scenario) setting.
    The original function compares other judges against the reference on the
    *same* conversation (row-wise on run_id), while the self-consistency group
    uses cross-run pairs — making the two types of bars methodologically
    incomparable.  Here every bar, including the self-consistency group, is
    cross-run, so they share a common unit of analysis.

    The self-consistency bar (always included, separated by a dashed line) acts
    as a noise floor: it quantifies how much ratings vary purely from
    conversation-level stochasticity.  Judges whose bars visibly exceed the
    self-consistency floor show systematic disagreement beyond that noise.
    """
    rates = compute_critical_miss_rates2(df_scen, reference, n_bootstrap=n_bootstrap)
    has_ci = n_bootstrap > 0

    if rates.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'No data (need J={reference} and ≥1 other judge)',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return fig

    judges     = rates['judge'].tolist()
    self_label = f'{reference} (self)'
    xs         = np.arange(len(judges))
    w          = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(judges) * 2.5 + 1), 5))

    bars = [
        ('exact_agreement_rate',   'Exact match',          '#2ca02c'),
        ('true_adjacent_rate',     'Off by ±1 level',      '#bcbd22'),
        ('non_critical_miss_rate', 'Wrong (non-critical)', '#ff7f0e'),
        ('critical_miss_rate',     'Critical miss',        '#d62728'),
    ]
    for i_col, (col, label, color) in enumerate(bars):
        offset = (i_col - 1.5) * w
        vals = rates[col].values
        yerr = None
        if has_ci:
            yerr = np.array([
                vals - rates[col + '_lo'].values,
                rates[col + '_hi'].values - vals,
            ])
        for idx, (xi, v) in enumerate(zip(xs + offset, vals)):
            this_yerr = ([[yerr[0][idx]], [yerr[1][idx]]] if has_ci else None)
            ax.bar(xi, v, w, color=color, alpha=0.82,
                   label=label if idx == 0 else '_nolegend_',
                   yerr=this_yerr,
                   error_kw=dict(ecolor='black', capsize=3, linewidth=1))
            top = v + (yerr[1][idx] if has_ci else 0) + 0.015
            ax.text(xi, top, f'{v:.0%}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(xs)
    ns = rates['n_matched_scenarios'].values
    xlabels = []
    for j, n in zip(judges, ns):
        if j == self_label:
            xlabels.append(f'J={reference} (self)\n(N={n:,} pairs)')
        else:
            xlabels.append(f'J={j}\n(N={n:,} pairs)')
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    title = f'Scenario-level agreement vs J={reference} — cross-run pairs'
    if has_ci:
        title += f'  (95% CI, B={n_bootstrap})'
    if show_title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# ─── Reproducibility analysis ─────────────────────────────────────────────────

def compute_run_agreement(df_scen: pd.DataFrame, fixed: dict = {}) -> pd.DataFrame:
    """
    Pairwise cross-run severity agreement within each (target, auditor, judge) cell.

    Forms all C(n, 2) run pairs per scenario and computes agreement rates.
    Returns one row per (target, auditor, judge, target_safe) cell.
    Reuses _compute_cross_run_confusion and _compute_rates.

    Columns: target, auditor, judge, target_safe, n_pairs + _RATE_COLS.
    """
    df = _apply_fixed(df_scen, fixed)
    results = []
    for (judge, target_safe), group_df in df.groupby(
            ['judge', 'target_safe'], observed=True):
        pairs = _compute_cross_run_confusion(group_df, judge, judge)
        if pairs.empty:
            continue
        for (target, auditor), cell_pairs in pairs.groupby(
                ['target', 'auditor'], observed=True):
            rates = _compute_rates(cell_pairs)
            rates.update({
                'target':      target,
                'auditor':     auditor,
                'judge':       str(judge),
                'target_safe': target_safe,
                'n_pairs':     len(cell_pairs),
            })
            results.append(rates)
    return pd.DataFrame(results)


def plot_run_agreement_summary(
        agreement_df: pd.DataFrame,
        fixed: dict = {},
        figsize: tuple = (11, 5)) -> plt.Figure:
    """
    Two-panel stacked bar chart of run-level agreement rates (safe / unsafe).

    Bars stacked as: exact match / ±1 adjacent / non-critical miss / critical miss.
    Pass fixed={'judge': 'L', 'auditor': 'L'} to subset before plotting.
    Returns Figure.
    """
    df = agreement_df.copy()
    for col, val in fixed.items():
        df = df[df[col] == val]

    stack_keys = [
        ('exact_agreement_rate',   'Exact match',        '#2ca02c'),
        ('true_adjacent_rate',     '±1 (adjacent)',      '#98df8a'),
        ('non_critical_miss_rate', 'Non-critical miss',  '#ff7f0e'),
        ('critical_miss_rate',     'Critical miss',      '#d62728'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, safe_val, panel_title in zip(
            axes, [True, False], ['Safe targets', 'Unsafe targets']):
        sub = df[df['target_safe'] == safe_val]
        if sub.empty:
            ax.set_title(f'{panel_title} — no data', fontsize=11)
            continue

        sizes = [s for s in SIZE_ORDER if s in sub['target'].values]
        xs = np.arange(len(sizes))
        bottoms = np.zeros(len(sizes))

        for key, label, color in stack_keys:
            vals = np.array([
                float(sub[sub['target'] == sz][key].mean())
                if not sub[sub['target'] == sz].empty else 0.0
                for sz in sizes
            ])
            ax.bar(xs, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0.08:
                    ax.text(xi, b + v / 2, f'{v:.0%}',
                            ha='center', va='center', fontsize=9,
                            color='white', fontweight='bold')
            bottoms += vals

        ax.set_xticks(xs)
        ax.set_xticklabels([f'T={s}' for s in sizes])
        ax.set_title(panel_title, fontsize=11)
        ax.set_ylabel('Rate', fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Run-level scenario agreement (cross-run pairs)', fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def compute_stability_curve(
        df_scen: pd.DataFrame,
        fixed: dict = {},
        min_runs: int = 8,
        n_bootstrap: int = 300,
        rng_seed: int = 42) -> pd.DataFrame:
    """
    Ordinal profile stability: MAD of k-run mean severity vs full-run reference.

    For each (target, auditor, judge, target_safe) cell with >= min_runs runs,
    bootstraps k-run subsamples (without replacement) and measures how far the
    per-scenario mean severity_ordinal deviates from the full-run reference.

    Returns long DataFrame:
      target, auditor, judge, target_safe, k, n_runs, mean_mad, ci_lo, ci_hi
    MAD is in ordinal units (0–4 scale).
    """
    df = _apply_fixed(df_scen, fixed)
    rng = np.random.default_rng(rng_seed)
    results = []

    for (target, auditor, judge, target_safe), cell in df.groupby(
            ['target', 'auditor', 'judge', 'target_safe'], observed=True):
        run_ids = cell['run_id'].unique()
        n_runs = len(run_ids)
        if n_runs < min_runs:
            continue

        pivot = (cell.pivot_table(index='run_id', columns='scenario_name',
                                  values='severity_ordinal', aggfunc='mean')
                     .reindex(run_ids))
        mat = pivot.values.astype(float)   # (n_runs, n_scenarios)
        ref = mat.mean(axis=0)             # (n_scenarios,)

        for k in range(1, n_runs):
            # Vectorised bootstrap: (n_bootstrap, k) index matrix
            idx = np.array([rng.choice(n_runs, k, replace=False)
                            for _ in range(n_bootstrap)])   # (n_bootstrap, k)
            samples = mat[idx].mean(axis=1)                # (n_bootstrap, n_scenarios)
            mads = np.abs(samples - ref).mean(axis=1)      # (n_bootstrap,)
            results.append({
                'target':      target,
                'auditor':     auditor,
                'judge':       str(judge),
                'target_safe': target_safe,
                'k':           k,
                'n_runs':      n_runs,
                'mean_mad':    float(mads.mean()),
                'ci_lo':       float(np.percentile(mads, 2.5)),
                'ci_hi':       float(np.percentile(mads, 97.5)),
            })

    return pd.DataFrame(results)


def compute_stability_curve_score(
        df_raw: pd.DataFrame,
        fixed: dict = {},
        min_runs: int = 8,
        n_bootstrap: int = 300,
        rng_seed: int = 42) -> pd.DataFrame:
    """
    Score stability: MAD of k-run mean score vs full-run reference.

    For each (target, auditor, judge, target_safe) cell with >= min_runs runs,
    bootstraps k-run subsamples (without replacement) and measures how far the
    mean score deviates from the full-run reference mean.

    Returns long DataFrame:
      target, auditor, judge, target_safe, k, n_runs, mean_mad, ci_lo, ci_hi
    MAD is in score units (0–100 scale).
    """
    df = _apply_fixed(df_raw, fixed)
    rng = np.random.default_rng(rng_seed)
    results = []

    for (target, auditor, judge, target_safe), cell in df.groupby(
            ['target', 'auditor', 'judge', 'target_safe'], observed=True):
        scores = cell['score'].dropna().values
        n_runs = len(scores)
        if n_runs < min_runs:
            continue

        ref = scores.mean()

        for k in range(1, n_runs):
            idx = np.array([rng.choice(n_runs, k, replace=False)
                            for _ in range(n_bootstrap)])   # (n_bootstrap, k)
            sample_means = scores[idx].mean(axis=1)         # (n_bootstrap,)
            mads = np.abs(sample_means - ref)               # (n_bootstrap,)
            results.append({
                'target':      target,
                'auditor':     auditor,
                'judge':       str(judge),
                'target_safe': target_safe,
                'k':           k,
                'n_runs':      n_runs,
                'mean_mad':    float(mads.mean()),
                'ci_lo':       float(np.percentile(mads, 2.5)),
                'ci_hi':       float(np.percentile(mads, 97.5)),
            })

    return pd.DataFrame(results)


def plot_stability_curve(stability_df: pd.DataFrame,
                          figsize: tuple = (8, 5),
                          ylabel: str = 'Mean ordinal MAD  (0–4 scale)',
                          title: str = 'Ordinal severity profile: convergence with run count') -> plt.Figure:
    """
    Line plot of MAD vs k runs, split by target safety.

    Shows mean MAD across cells with ±1 std shaded band (variability across
    configurations). Works for both ordinal-scale and score-scale stability DFs.
    Returns Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {True: '#4878CF', False: '#C44E52'}
    labels = {True: 'Safe targets', False: 'Unsafe targets'}

    for safe_val in [True, False]:
        sub = stability_df[stability_df['target_safe'] == safe_val]
        if sub.empty:
            continue
        grp = sub.groupby('k')['mean_mad']
        k_vals  = grp.mean().index.values
        means   = grp.mean().values
        stds    = grp.std(ddof=1).fillna(0).values

        ax.plot(k_vals, means, color=colors[safe_val],
                label=labels[safe_val], linewidth=2, marker='o', markersize=5)
        ax.fill_between(k_vals,
                        np.maximum(0, means - stds),
                        means + stds,
                        color=colors[safe_val], alpha=0.15)

    ax.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax.set_xlabel('Runs used (k)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def compute_detection_stability(
        df_raw: pd.DataFrame,
        fixed: dict = {},
        n_bootstrap: int = 500,
        rng_seed: int = 42) -> pd.DataFrame:
    """
    AUROC = P(safe_run > unsafe_run) as a function of k runs used.

    Uses run-level score as the ranking signal (consistent with the main AUROC
    analysis; AUROC is rank-based so absolute values are irrelevant). Pools
    all target sizes within each (auditor, judge) configuration, matching the
    existing main-paper AUROC analysis (which mixes XS/S/M safe vs unsafe).

    Bootstraps k-run subsamples from each pool to show how AUROC confidence
    tightens as k increases. k = number of runs sampled from each pool.

    Returns DataFrame: auditor, judge, k, auroc, ci_lo, ci_hi, n_safe, n_unsafe.
    """
    df = _apply_fixed(df_raw, fixed)
    rng = np.random.default_rng(rng_seed)

    results = []
    for (auditor, judge), grp in df.groupby(['auditor', 'judge'], observed=True):
        safe_v   = grp[ grp['target_safe']]['score'].values
        unsafe_v = grp[~grp['target_safe']]['score'].values
        if len(safe_v) < 3 or len(unsafe_v) < 3:
            continue
        max_k = min(len(safe_v), len(unsafe_v))
        for k in range(1, max_k + 1):
            s_samp = rng.choice(safe_v,   (n_bootstrap, k), replace=True)
            u_samp = rng.choice(unsafe_v, (n_bootstrap, k), replace=True)
            # AUROC = P(safe_score > unsafe_score): safe models score higher.
            s3 = s_samp[:, :, None]
            u3 = u_samp[:, None, :]
            aurocs = ((s3 > u3).astype(float) + 0.5 * (s3 == u3).astype(float)
                      ).mean(axis=(1, 2))
            results.append({
                'auditor':   str(auditor),
                'judge':     str(judge),
                'k':         k,
                'auroc':     float(aurocs.mean()),
                'ci_lo':     float(np.percentile(aurocs, 2.5)),
                'ci_hi':     float(np.percentile(aurocs, 97.5)),
                'n_safe':    len(safe_v),
                'n_unsafe':  len(unsafe_v),
            })

    return pd.DataFrame(results)


def plot_detection_stability(detection_df: pd.DataFrame,
                              figsize: tuple = (8, 5)) -> plt.Figure:
    """
    Line plot of AUROC vs k runs used, one line per (auditor, judge) config.

    AUROC = P(safe_score > unsafe_score), pooled across target sizes within
    each config (matching the existing main-paper AUROC analysis). Shaded
    band is the 95% bootstrap CI; narrowing band shows growing confidence.
    Returns Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    configs = detection_df[['auditor', 'judge']].drop_duplicates()
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (_, row) in enumerate(configs.iterrows()):
        sub = detection_df[(detection_df['auditor'] == row['auditor']) &
                           (detection_df['judge']   == row['judge'])]
        color = palette[i % len(palette)]
        label = f'A={row["auditor"]} J={row["judge"]}'
        ax.plot(sub['k'], sub['auroc'], color=color, linewidth=2,
                marker='o', markersize=4, label=label)
        ax.fill_between(sub['k'], sub['ci_lo'], sub['ci_hi'],
                        color=color, alpha=0.15)

    ax.axhline(0.5, color='grey', linestyle=':', linewidth=1.5, label='Chance (0.5)')
    ax.set_xlabel('Runs used (k)', fontsize=11)
    ax.set_ylabel('AUROC   P(safe > unsafe)', fontsize=11)
    ax.set_title('Detection reliability vs number of runs  (pooled across target sizes)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


# ─── CLI sanity check ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else 'comparison_results/norwegian_safety'
    print(f"Loading from: {base}")
    df = load_comparison_results(base)
    print(f"Loaded {len(df)} rows.\n")
    show_available_combos(df)
