#!/usr/bin/env python3
"""
Generate Production-Level LaTeX Tables (ECS_corr only)

Creates one publication-ready table: ECS_corr (Lin's CCC) vs Cost.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

STUDY_GROUPS = {
    "Cognition": ["study_001", "study_002", "study_003", "study_004"],
    "Strategic": ["study_009", "study_010", "study_011", "study_012"],
    "Social": ["study_005", "study_006", "study_007", "study_008"],
}

METHOD_DISPLAY_MAP = {
    "v1-empty": "A1",
    "v2-human": "A2",
    "v3-human-plus-demo": "A3",
    "v4-background": "A4",
}


def format_model_name(base_model: str) -> str:
    """Format model name for publication."""
    name = base_model.lower()
    if 'qwen' in name:
        return "Qwen3 Next 80b"
    if 'grok' in name:
        return "Grok 4.1 Fast"
    if 'deepseek' in name:
        return "DeepSeek V3.2"
    if 'claude' in name and 'haiku' in name:
        return "Claude Haiku 4.5"
    if 'gpt' in name:
        if 'oss' in name:
            return "GPT OSS 120b" if '120' in name else "GPT OSS"
        if '5' in name and 'nano' in name:
            return "GPT 5 Nano"
        return "GPT"
    for prefix in ['mistralai_', 'deepseek_', 'openai_', 'x_ai_', 'google_', 'anthropic_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suf in ['_none', '_preview', '_fast', '_v3.2', '_v4.1', '_5', 'none', 'preview']:
        if name.endswith(suf):
            name = name[:-len(suf)]
    name = name.replace('_', ' ').strip()
    result = ' '.join(w.capitalize() for w in name.split())
    if 'oss' in result.lower():
        result = result.replace('Oss', 'OSS').replace('oss', 'OSS')
    return result


def escape_latex(text: str) -> str:
    for char, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#')]:
        text = text.replace(char, rep)
    return text


def get_color_cell(val_str: str, rank: int = -1, is_cost: bool = False) -> str:
    if '$' not in val_str and val_str not in ['--', '---', '{--}', '{---}']:
        val_str = f"${val_str}$" if val_str and val_str[0].isdigit() or (val_str and val_str[0] in '-+') else val_str
    if rank == 0:
        return f"\\cellcolor{{best1}}{{{val_str}}}"
    if rank == 1:
        return f"\\cellcolor{{best2}}{{{val_str}}}"
    if rank == 2:
        return f"\\cellcolor{{best3}}{{{val_str}}}"
    if rank == 3:
        return f"\\cellcolor{{best4}}{{{val_str}}}"
    if rank == -1:
        return f"\\cellcolor{{worst1}}{{{val_str}}}"
    if rank == -2:
        return f"\\cellcolor{{worst2}}{{{val_str}}}"
    if rank == -3:
        return f"\\cellcolor{{worst3}}{{{val_str}}}"
    if rank == -4:
        return f"\\cellcolor{{worst4}}{{{val_str}}}"
    return val_str


def get_rank(value: float, sorted_values: list, is_cost: bool = False) -> int:
    if value is None or value not in sorted_values:
        return -999
    idx = sorted_values.index(value)
    if is_cost:
        if idx == 0:
            return 0
        if idx == 1:
            return 1
        if idx == 2:
            return 2
        if idx == 3:
            return 3
        if idx == len(sorted_values) - 1:
            return -1
        if idx == len(sorted_values) - 2:
            return -2
        if idx == len(sorted_values) - 3:
            return -3
        if idx == len(sorted_values) - 4:
            return -4
    else:
        if idx == 0:
            return 0
        if idx == 1:
            return 1
        if idx == 2:
            return 2
        if idx == 3:
            return 3
        if idx == len(sorted_values) - 1:
            return -1
        if idx == len(sorted_values) - 2:
            return -2
        if idx == len(sorted_values) - 3:
            return -3
        if idx == len(sorted_values) - 4:
            return -4
    return -999


def parse_model_method(model_key: str) -> Tuple[str, str]:
    parts = model_key.rsplit('_', 1)
    if len(parts) == 2:
        m = parts[1]
        if m.startswith('v') or m.startswith('empty') or m == 'legacy' or m == 'example-v4':
            return parts[0], m
    if len(parts) == 2:
        return parts[0], parts[1]
    return model_key, "unknown"


def load_benchmark_data(summary_json_path: Path) -> Dict:
    with open(summary_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def organize_data_by_model_method(data: Dict, random_alignment_stats: Dict = None, results_dir: Path = None) -> Dict:
    """Organize by base model and method. Filter: skip when no ECS and no tokens."""
    organized = defaultdict(lambda: defaultdict(dict))
    if results_dir is None:
        results_dir = Path("results")

    for model_key, model_data in data.get("models", {}).items():
        base_model, method = parse_model_method(model_key)
        summary = model_data.get("summary", {})
        total_usage = summary.get("total_usage", {})
        total_tokens = total_usage.get("total_tokens", 0)
        ecs_overall = summary.get("ecs_overall") or summary.get("average_ecs")
        if total_tokens == 0 and ecs_overall is None:
            continue
        if method == 'example-v4':
            continue
        if method == 'legacy' and base_model in organized and any(m.startswith('v') for m in organized[base_model].keys()):
            continue
        organized[base_model][method] = {
            "summary": summary,
            "studies": model_data.get("studies", []),
        }
    return organized


def generate_latex_table_ecs_cost(organized_data: Dict) -> str:
    """Single table: Model, Method, ECS (CCC), Cost."""
    rows_data = []
    for base_model in sorted(organized_data.keys()):
        if 'temp' in base_model.lower():
            continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        for method in sorted_methods:
            m_data = methods[method]
            summary = m_data.get("summary", {})
            ecs = summary.get("ecs_overall") or summary.get("average_ecs")
            cost = summary.get("total_usage", {}).get("total_cost", 0.0)
            rows_data.append({
                "base_model": base_model,
                "method": method,
                "ecs": ecs,
                "cost": cost,
            })

    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows

    ecs_vals = sorted(list(set([r["ecs"] for r in regular_rows if r["ecs"] is not None])), reverse=True)
    cost_vals = sorted(list(set([r["cost"] for r in regular_rows if r["cost"] > 0])))

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Model Performance: ECS\\_corr (Lin's Concordance Correlation Coefficient) vs Cost. ECS measures agreement between human and agent effect profiles (Cohen's d-equivalent). Best ECS highlighted in teal, worst in salmon; best (lowest) cost in teal.}",
        "\\label{tab:ecs-cost}",
        "\\begin{tabular}{@{}llcc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Method} & \\textbf{ECS (CCC)} & \\textbf{Cost (\\$)} \\\\",
        "\\midrule",
    ]
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        n_span = sum(1 for x in rows_data_sorted if x["base_model"] == r["base_model"])
        m_cell = f"\\multirow{{{n_span}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed:
            lines.append("\\midrule")
        curr_model = r["base_model"]
        ecs_val = r["ecs"]
        ecs_str = f"{ecs_val:.3f}" if ecs_val is not None else "---"
        rank_ecs = -999 if is_mixed else (get_rank(ecs_val, ecs_vals, is_cost=False) if ecs_val is not None else -999)
        rank_cost = -999 if is_mixed else get_rank(r["cost"], cost_vals, is_cost=True)
        lines.append(" & ".join([
            m_cell,
            meth_disp,
            get_color_cell(ecs_str, rank_ecs),
            get_color_cell(f"{r['cost']:.4f}", rank_cost, is_cost=True),
        ]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="results/benchmark_summary.json")
    parser.add_argument("--results-dir", "--output-dir", dest="results_dir", default="results")
    parser.add_argument("--output-latex", default="results/production_tables.tex")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"Error: Summary JSON not found: {summary_path}. Run generate_results_table.py --format all --output results/benchmark_summary first.")
        return 1

    data = load_benchmark_data(summary_path)
    organized = organize_data_by_model_method(data, results_dir=Path(args.results_dir))

    out_path = Path(args.output_latex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{siunitx}\n\\usepackage[table]{xcolor}\n\n")
        f.write("\\definecolor{best1}{HTML}{B2DFDB}\n\\definecolor{best2}{HTML}{CDEAE8}\n\\definecolor{best3}{HTML}{E0F2F1}\n\\definecolor{best4}{HTML}{F1F8F7}\n")
        f.write("\\definecolor{worst1}{HTML}{FFCDD2}\n\\definecolor{worst2}{HTML}{F8E1E3}\n\\definecolor{worst3}{HTML}{FCEBED}\n\\definecolor{worst4}{HTML}{FFF8F9}\n\n")
        f.write("\\begin{document}\n\n")
        f.write(generate_latex_table_ecs_cost(organized) + "\n\n")
        f.write("\\end{document}\n")

    print(f"Production table written to {out_path}")


if __name__ == "__main__":
    main()
