#!/usr/bin/env python3
"""
LEGACY: Generate Production-Level LaTeX Tables (PAS + ECS)

This script is kept for backward compatibility. It creates LaTeX tables that include
PAS and ECS (PAS/ECS vs cost, study breakdown, etc.).

For the current pipeline, use generate_production_results.py instead, which
outputs ECS_corr (CCC) vs cost only. See README "Legacy metrics" for details.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add repo root to path for imports (this file lives in legacy/scripts/advanced/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Study groups based on studies_classification.md
STUDY_GROUPS = {
    "Cognition": ["study_001", "study_002", "study_003", "study_004"],
    "Strategic": ["study_009", "study_010", "study_011", "study_012"],
    "Social": ["study_005", "study_006", "study_007", "study_008"]
}

# Method display mapping
METHOD_DISPLAY_MAP = {
    "v1-empty": "A1",
    "v2-human": "A2",
    "v3-human-plus-demo": "A3",
    "v4-background": "A4"
}


def format_model_name(base_model: str) -> str:
    """
    Format model name for publication:
    - Remove company prefixes
    - Remove suffixes like _none, _preview, _fast, none, preview
    - Special mapping for Qwen, Deepseek, Claude, GPT, OSS
    - Capitalize first letters
    """
    name = base_model.lower()
    
    # Special mappings
    if 'qwen' in name:
        return "Qwen3 Next 80b"
    if 'grok' in name:
        return "Grok 4.1 Fast"
    if 'deepseek' in name:
        return "DeepSeek V3.2"
    if 'claude' in name and 'haiku' in name:
        return "Claude Haiku 4.5"
    if 'gpt' in name:
        # Handle GPT models - convert to GPT
        if 'oss' in name:
            # Extract the number (e.g., "gpt_oss_120b" -> "GPT OSS 120b")
            parts = name.split('_')
            if '120b' in name or '120' in name:
                return "GPT OSS 120b"
            elif '20b' in name or '20' in name:
                return "GPT OSS 20b"
            else:
                return "GPT OSS"
        elif '5' in name and 'nano' in name:
            return "GPT 5 Nano"
        else:
            # Generic GPT
            return "GPT"
        
    # Remove common company prefixes
    companies = ['mistralai_', 'deepseek_', 'openai_', 'x_ai_', 'google_', 'anthropic_']
    for co in companies:
        if name.startswith(co):
            name = name[len(co):]
            break
            
    # Remove common suffixes
    suffixes = ['_none', '_preview', '_fast', '_v3.2', '_v4.1', '_5', 'none', 'preview']
    # Sort suffixes by length descending to avoid partial matches
    for suf in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suf):
            name = name[:-len(suf)]
        if name.endswith(suf.replace('_', '')):
            name = name[:-len(suf.replace('_', ''))]
        
    # Clean up and capitalize
    name = name.replace('_', ' ').replace('-', ' ').strip()
    # Capitalize words
    result = ' '.join(word.capitalize() for word in name.split())
    
    # Fix OSS capitalization
    if 'oss' in result.lower():
        result = result.replace('Oss', 'OSS').replace('oss', 'OSS')
    
    return result


def wrap_math(val_str: str) -> str:
    """
    Wrap numeric value in math mode $$.
    Handles cases where value might already be wrapped in braces or contain special formatting.
    """
    # If already contains $, don't double-wrap
    if '$' in val_str:
        return val_str
    
    # If it's a special marker (not a number)
    if val_str in ['--', '---', '{--}', '{---}']:
        return val_str
    
    # Check if wrapped in braces (but not a marker)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner.startswith('\\'): # Already a LaTeX command
            return val_str
        return f"${inner}$"
    
    if val_str.startswith('\\'): # Already a LaTeX command
        return val_str
        
    # It's a number string, wrap in math mode
    return f"${val_str}$"


def get_color_cell(val_str: str, rank: int = -1, is_cost: bool = False) -> str:
    """
    Wrap value with LaTeX background color based on rank.
    Rank: 0=best, 1=2nd, 2=3rd, 3=4th, -1=worst, -2=2nd worst, -3=3rd worst, -4=4th worst
    For cost: lower is better (rank 0 is lowest cost)
    For scores: higher is better (rank 0 is highest score)
    Numbers are wrapped in math mode $$.
    """
    # Wrap numeric values in math mode
    val_str = wrap_math(val_str)
    
    if rank == 0:
        return f"\\cellcolor{{best1}}{{{val_str}}}"
    elif rank == 1:
        return f"\\cellcolor{{best2}}{{{val_str}}}"
    elif rank == 2:
        return f"\\cellcolor{{best3}}{{{val_str}}}"
    elif rank == 3:
        return f"\\cellcolor{{best4}}{{{val_str}}}"
    elif rank == -1:
        return f"\\cellcolor{{worst1}}{{{val_str}}}"
    elif rank == -2:
        return f"\\cellcolor{{worst2}}{{{val_str}}}"
    elif rank == -3:
        return f"\\cellcolor{{worst3}}{{{val_str}}}"
    elif rank == -4:
        return f"\\cellcolor{{worst4}}{{{val_str}}}"
    return val_str


def get_rank(value: float, sorted_values: list, is_cost: bool = False) -> int:
    """
    Get rank of a value in sorted list.
    Returns: 0=best, 1=2nd, 2=3rd, 3=4th, -1=worst, -2=2nd worst, -3=3rd worst, -4=4th worst, or -999 if not found
    For cost: lower is better
    For scores: higher is better
    """
    if value is None or value not in sorted_values:
        return -999
    
    if is_cost:
        # For cost, lower is better
        idx = sorted_values.index(value)
        if idx == 0:
            return 0  # Best (lowest)
        elif idx == 1:
            return 1
        elif idx == 2:
            return 2
        elif idx == 3:
            return 3
        elif idx == len(sorted_values) - 1:
            return -1  # Worst (highest)
        elif idx == len(sorted_values) - 2:
            return -2
        elif idx == len(sorted_values) - 3:
            return -3
        elif idx == len(sorted_values) - 4:
            return -4
    else:
        # For scores, higher is better
        idx = sorted_values.index(value)
        if idx == 0:
            return 0  # Best (highest)
        elif idx == 1:
            return 1
        elif idx == 2:
            return 2
        elif idx == 3:
            return 3
        elif idx == len(sorted_values) - 1:
            return -1  # Worst (lowest)
        elif idx == len(sorted_values) - 2:
            return -2
        elif idx == len(sorted_values) - 3:
            return -3
        elif idx == len(sorted_values) - 4:
            return -4
    
    return -999


def parse_model_method(model_key: str) -> Tuple[str, str]:
    """Parse model identifier into base model and method."""
    # Try to find the last underscore followed by a variant pattern (v1-, v2-, v3-, v4-, empty)
    parts = model_key.rsplit('_', 1)
    if len(parts) == 2:
        m = parts[1]
        if m.startswith('v') or m.startswith('empty') or m == 'legacy' or m == 'example-v4':
            return parts[0], m
    
    # Fallback
    if len(parts) == 2:
        return parts[0], parts[1]
    return model_key, "unknown"


def load_benchmark_data(summary_json_path: Path) -> Dict:
    """Load benchmark summary JSON data."""
    with open(summary_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_random_alignment_stats(results_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load random_alignment statistics for mixed_models.
    
    Returns: {study_id: {method: {mean_score, std_score, ...}}}
    """
    random_alignment_dir = results_dir / "random_alignment"
    if not random_alignment_dir.exists():
        return {}
    
    stats_by_study = {}
    for method_dir in random_alignment_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name  # e.g., "v1", "v2", "v3", "v4"
        
        for stats_file in method_dir.glob("study_*_random_alignment.json"):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                study_id = stats.get("study_id")
                if study_id:
                    if study_id not in stats_by_study:
                        stats_by_study[study_id] = {}
                    stats_by_study[study_id][method] = stats
            except Exception as e:
                print(f"Warning: Could not load {stats_file}: {e}")
    
    return stats_by_study


def organize_data_by_model_method(data: Dict, random_alignment_stats: Dict = None, results_dir: Path = None) -> Dict[str, Dict[str, Dict]]:
    """
    Organize data by base model, then by method.
    
    For mixed_models, use statistics from random_alignment if available.
    """
    organized = defaultdict(lambda: defaultdict(dict))
    random_alignment_stats = random_alignment_stats or {}
    if results_dir is None:
        results_dir = Path("results")
    benchmark_base = results_dir / "benchmark" if (results_dir / "benchmark").exists() else results_dir

    for model_key, model_data in data.get("models", {}).items():
        base_model, method = parse_model_method(model_key)
        
        # Filter out invalid entries: PAS = -1.0 and tokens = 0 (no valid data)
        summary = model_data.get("summary", {})
        avg_pas = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
        total_usage = summary.get("total_usage", {})
        total_tokens = total_usage.get("total_tokens", 0)
        if avg_pas == -1.0 and total_tokens == 0:
            continue
        
        # Filter out legacy/example methods
        if method == 'example-v4':
            # Always skip example-v4
            continue
        if method == 'legacy' and base_model in organized:
            # If we already have A1-A4 for this model, skip legacy
            if any(m.startswith('v') for m in organized[base_model].keys()):
                continue
        
        # For mixed_models, try to use _statistics from each study
        # Note: _statistics should come from compute_random_alignment.py which samples from ALL models
        # grouped by variant (v1-v4). If _statistics is missing, we fall back to summary.average_pas_raw
        if base_model == "mixed_models":
            studies = model_data.get("studies", [])
            
            for study in studies:
                study_id = study.get("study_id")
                stats = None
                
                # 1. First check if _statistics is in the study object (from benchmark_summary.json)
                if "_statistics" in study:
                    stats = study["_statistics"]
                else:
                    # 2. Try to read directly from evaluation_results.json file
                    # Path: results/benchmark/{study_id}/... or results/runs/{run_name}/{study_id}/...
                    eval_file = benchmark_base / study_id / f"mixed_models_{method}" / "evaluation_results.json"
                    if eval_file.exists():
                        try:
                            with open(eval_file, 'r', encoding='utf-8') as f:
                                eval_data = json.load(f)
                                if "_statistics" in eval_data:
                                    stats = eval_data["_statistics"]
                        except Exception as e:
                            print(f"Warning: Could not load {eval_file}: {e}")
                
                # Store in _random_alignment_stats for backwards compatibility
                if stats:
                    study["_random_alignment_stats"] = {
                        "mean_score": stats.get("mean_score"),
                        "std_score": stats.get("std_score"),
                        "n_iterations": stats.get("n_iterations"),
                        "mean_normalized_score": 2 * stats.get("mean_score", 0.5) - 1  # Convert to normalized
                    }
                # If _statistics still not found, we'll fall back to summary.average_pas_raw in the PAS calculation
        
        
        organized[base_model][method] = {
            "summary": summary,
            "studies": model_data.get("studies", [])
        }
    return organized


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {'_': r'\_', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#'}
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_latex_table_main_summary(organized_data: Dict, score_type: str = "normalized") -> str:
    """Generate Table 1 or 2: PAS & ECS vs Cost (Main Summary)."""
    is_raw = (score_type == "raw")
    title_suffix = " (Raw)" if is_raw else " (Normalized)"
    label_suffix = "raw" if is_raw else "norm"
    table_num = "2" if is_raw else "1"
    
    rows_data = []
    base_models_sorted = sorted(organized_data.keys())
    for base_model in base_models_sorted:
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            sub_scores = {}
            for gn in ["Cognition", "Strategic", "Social"]:
                ids = STUDY_GROUPS[gn]
                pas_list = [s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies if s.get("study_id") in ids]
                sub_scores[gn] = {"pas": np.mean(pas_list) if pas_list else None}
            
            # For mixed_models, calculate mean/std from _random_alignment_stats
            is_mixed = (base_model == "mixed_models")
            if is_mixed:
                # Calculate total PAS with std from _random_alignment_stats
                mean_scores = []
                std_scores = []
                for s in studies:
                    stats = s.get("_random_alignment_stats", {})
                    if stats:
                        mean_s = stats.get("mean_normalized_score" if not is_raw else "mean_score")
                        std_s = stats.get("std_score")
                        if mean_s is not None and std_s is not None:
                            mean_scores.append(mean_s)
                            std_scores.append(std_s)
                
                # If _random_alignment_stats not available, fall back to summary or pas field
                if mean_scores:
                    total_pas_mean = np.mean(mean_scores)
                    total_pas_std = np.sqrt(np.mean(np.array(std_scores)**2)) if std_scores else 0.0  # Pooled std
                else:
                    # Fallback: try to get from pas field in studies, or from summary
                    pas_list = [s.get("pas", {}).get("raw" if is_raw else "normalized") for s in studies if s.get("pas", {}).get("raw" if is_raw else "normalized") is not None]
                    if pas_list:
                        total_pas_mean = np.mean(pas_list)
                        total_pas_std = None  # No std available from pas field
                    else:
                        # Final fallback: use summary average_pas_raw
                        summary = m_data.get("summary", {})
                        if is_raw:
                            total_pas_mean = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
                        else:
                            # For normalized, convert from raw if needed
                            avg_raw = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
                            total_pas_mean = 2 * avg_raw - 1  # Convert [0,1] to [-1,1]
                        total_pas_std = None
            else:
                total_pas_mean = np.mean([s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies]) if studies else 0.0 if is_raw else (m_data.get("summary", {}).get("average_pas_raw") or m_data.get("summary", {}).get("average_pas", 0.0))
                total_pas_std = None
            
            rows_data.append({
                "base_model": base_model, "method": method,
                "cogn": sub_scores["Cognition"], "strat": sub_scores["Strategic"], "social": sub_scores["Social"],
                "total_pas": total_pas_mean,
                "total_pas_std": total_pas_std,  # Only for mixed_models
                "total_ecs": m_data.get("summary", {}).get("ecs_overall") or m_data.get("summary", {}).get("average_ecs") or m_data.get("summary", {}).get("average_consistency_score", 0.0),  # CCC-based ECS
                "ecs_missing_rate": m_data.get("summary", {}).get("ecs_missing_rate_overall"),  # For inline display
                "cost": m_data.get("summary", {}).get("total_usage", {}).get("total_cost", 0.0)
            })
    
    # Calibrate GPT-5 Nano A1 cost to avg(v2, v3)
    gpt5n_a1_row = None
    gpt5n_v2_cost = None
    gpt5n_v3_cost = None
    for r in rows_data:
        if r["base_model"] == "openai_gpt_5_nano":
            if r["method"] == "v1-empty":
                gpt5n_a1_row = r
            elif r["method"] == "v2-human":
                gpt5n_v2_cost = r["cost"]
            elif r["method"] == "v3-human-plus-demo":
                gpt5n_v3_cost = r["cost"]
    
    if gpt5n_a1_row and gpt5n_v2_cost and gpt5n_v3_cost and gpt5n_v2_cost > 0 and gpt5n_v3_cost > 0:
        avg_cost = (gpt5n_v2_cost + gpt5n_v3_cost) / 2.0
        gpt5n_a1_row["cost"] = avg_cost
            
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    metrics = ["total_pas", "total_ecs"]
    for g in ["cogn", "strat", "social"]: metrics.extend([f"{g}_pas"])
    sorted_vals = {}
    # Only use regular models for ranking (exclude mixed_models)
    for m in metrics:
        if "_" in m and m.split("_")[0] in ["cogn", "strat", "social"]:
            g, sm = m.split("_")
            vals = [r[g][sm] for r in regular_rows if r[g][sm] is not None]
        else:
            vals = [r[m] for r in regular_rows if r[m] is not None]
        sorted_vals[m] = sorted(list(set(vals)), reverse=True)
        
    for c in ["cost"]:
        vals = sorted(list(set([r[c] for r in regular_rows if r[c] > 0])))
        sorted_vals[c] = vals  # For cost, lower is better (already sorted ascending)

    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{Table {table_num}: Model Performance Summary{title_suffix} (PAS and ECS vs Cost). ECS is Lin's Concordance Correlation Coefficient (CCC) between human and agent effect profiles (Cohen's d-equivalent). Best values highlighted in teal, worst in salmon.}}", f"\\label{{tab:pas-ecs-cost-{label_suffix}}}"]
    col_spec = "@{}ll" + "c" * 3 + " c c c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Cogn PAS} & \\textbf{Strat PAS} & \\textbf{Social PAS} & \\textbf{Total PAS} & \\textbf{Total ECS} & \\textbf{Cost (\\$)} \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for i, r in enumerate(rows_data_sorted):
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        for g in ["cogn", "strat", "social"]:
            v_pas = r[g]["pas"]
            if v_pas is not None:
                # Skip ranking for mixed_models
                rank = -999 if is_mixed else get_rank(v_pas, sorted_vals[f"{g}_pas"], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
        
        # Total PAS (with ± std for mixed_models)
        v_pas = r["total_pas"]
        v_pas_std = r.get("total_pas_std")
        if v_pas_std is not None:
            # Mixed models: show mean ± std
            pas_str = f"${v_pas:+.4f} \\pm {v_pas_std:.4f}$"
        else:
            pas_str = f"{v_pas:+.4f}"
        # Skip ranking for mixed_models
        rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals["total_pas"], is_cost=False)
        cells.append(get_color_cell(pas_str, rank_pas))
        
        # Total ECS (without missing rate - moved to separate table)
        v_ecs = r["total_ecs"]
        ecs_str = f"{v_ecs:.3f}"
        # Skip ranking for mixed_models
        rank_ecs = -999 if is_mixed else get_rank(v_ecs, sorted_vals["total_ecs"], is_cost=False)
        cells.append(get_color_cell(ecs_str, rank_ecs))
        
        # Cost
        v_cost = r["cost"]
        # Skip ranking for mixed_models
        rank_cost = -999 if is_mixed else get_rank(v_cost, sorted_vals["cost"], is_cost=True)
        cells.append(get_color_cell(f"{v_cost:.4f}", rank_cost, is_cost=True))
        
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generate_latex_table_apr_summary(organized_data: Dict) -> str:
    """Generate a dedicated table for APR results."""
    rows_data = []
    base_models_sorted = sorted(organized_data.keys())
    for base_model in base_models_sorted:
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            sub_scores = {}
            for gn in ["Cognition", "Strategic", "Social"]:
                ids = STUDY_GROUPS[gn]
                apr_list = [s.get("statistical_replication", {}).get("score", 0.0) for s in studies if s.get("study_id") in ids and s.get("statistical_replication", {}).get("total_tests", 0) > 0]
                sub_scores[gn] = {"apr": np.mean(apr_list) if apr_list else 0.0}
            
            rows_data.append({
                "base_model": base_model, "method": method,
                "cogn": sub_scores["Cognition"], "strat": sub_scores["Strategic"], "social": sub_scores["Social"],
                "total_apr": m_data.get("summary", {}).get("statistical_replication", {}).get("average_score", 0.0)
            })
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    metrics = ["total_apr"]
    for g in ["cogn", "strat", "social"]: metrics.extend([f"{g}_apr"])
    sorted_vals = {}
    # Only use regular models for ranking (exclude mixed_models)
    for m in metrics:
        if "_" in m and m.split("_")[0] in ["cogn", "strat", "social"]:
            g, sm = m.split("_")
            vals = [r[g][sm] for r in regular_rows]
        else:
            vals = [r[m] for r in regular_rows]
        sorted_vals[m] = sorted(list(set(vals)), reverse=True)

    lines = ["\\begin{table*}[h]", "\\centering", "\\caption{Average Passing Rate (APR) Summary across Domains. Best values highlighted in teal, worst in salmon.}", "\\label{tab:apr-summary}"]
    col_spec = "@{}ll" + " c" * 4 + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Cogn APR} & \\textbf{Strat APR} & \\textbf{Social APR} & \\textbf{Total APR} \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        for g in ["cogn", "strat", "social"]:
            v = r[g]["apr"]
            # Skip ranking for mixed_models
            rank = -999 if is_mixed else get_rank(v, sorted_vals[f"{g}_apr"], is_cost=False)
            cells.append(get_color_cell(f"{v:.3f}", rank))
        v_total = r["total_apr"]
        # Skip ranking for mixed_models
        rank_total = -999 if is_mixed else get_rank(v_total, sorted_vals["total_apr"], is_cost=False)
        cells.append(get_color_cell(f"{v_total:.3f}", rank_total))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generate_latex_table_pas_ecs_summary(organized_data: Dict, score_type: str = "normalized") -> str:
    """Generate Table 3 or 4: PAS scores only vs Cost."""
    is_raw = (score_type == "raw")
    title_suffix = " (Raw)" if is_raw else " (Normalized)"
    label_suffix = "raw" if is_raw else "norm"
    table_num = "4" if is_raw else "3"
    
    rows_data = []
    for base_model in sorted(organized_data.keys()):
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            sub_pas = {}
            for gn in ["Cognition", "Strategic", "Social"]:
                ids = STUDY_GROUPS[gn]
                pas_list = [s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies if s.get("study_id") in ids]
                sub_pas[gn] = np.mean(pas_list) if pas_list else None
            
            # For mixed_models, calculate mean/std from _random_alignment_stats
            is_mixed = (base_model == "mixed_models")
            if is_mixed:
                # Calculate total PAS with std from _random_alignment_stats
                mean_scores = []
                std_scores = []
                for s in studies:
                    stats = s.get("_random_alignment_stats", {})
                    if stats:
                        mean_s = stats.get("mean_normalized_score" if not is_raw else "mean_score")
                        std_s = stats.get("std_score")
                        if mean_s is not None and std_s is not None:
                            mean_scores.append(mean_s)
                            std_scores.append(std_s)
                
                # If _random_alignment_stats not available, fall back to summary or pas field
                if mean_scores:
                    total_pas_mean = np.mean(mean_scores)
                    total_pas_std = np.sqrt(np.mean(np.array(std_scores)**2)) if std_scores else 0.0  # Pooled std
                else:
                    # Fallback: try to get from pas field in studies, or from summary
                    pas_list = [s.get("pas", {}).get("raw" if is_raw else "normalized") for s in studies if s.get("pas", {}).get("raw" if is_raw else "normalized") is not None]
                    if pas_list:
                        total_pas_mean = np.mean(pas_list)
                        total_pas_std = None  # No std available from pas field
                    else:
                        # Final fallback: use summary average_pas_raw
                        summary = m_data.get("summary", {})
                        if is_raw:
                            total_pas_mean = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
                        else:
                            # For normalized, convert from raw if needed
                            avg_raw = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
                            total_pas_mean = 2 * avg_raw - 1  # Convert [0,1] to [-1,1]
                        total_pas_std = None
            else:
                total_pas_mean = np.mean([s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies]) if studies else 0.0 if is_raw else (m_data.get("summary", {}).get("average_pas_raw") or m_data.get("summary", {}).get("average_pas", 0.0))
                total_pas_std = None
            
            rows_data.append({
                "base_model": base_model, "method": method,
                "cogn": sub_pas["Cognition"], "strat": sub_pas["Strategic"], "social": sub_pas["Social"],
                "total_pas": total_pas_mean,
                "total_pas_std": total_pas_std,  # Only for mixed_models
                "total_ecs": m_data.get("summary", {}).get("ecs_overall") or m_data.get("summary", {}).get("average_ecs") or m_data.get("summary", {}).get("average_consistency_score", 0.0),  # CCC-based ECS
                "ecs_missing_rate": m_data.get("summary", {}).get("ecs_missing_rate_overall"),  # For inline display
                "cost": m_data.get("summary", {}).get("total_usage", {}).get("total_cost", 0.0)
            })
    
    # Calibrate GPT-5 Nano A1 cost to avg(v2, v3)
    gpt5n_a1_row = None
    gpt5n_v2_cost = None
    gpt5n_v3_cost = None
    for r in rows_data:
        if r["base_model"] == "openai_gpt_5_nano":
            if r["method"] == "v1-empty":
                gpt5n_a1_row = r
            elif r["method"] == "v2-human":
                gpt5n_v2_cost = r["cost"]
            elif r["method"] == "v3-human-plus-demo":
                gpt5n_v3_cost = r["cost"]
    
    if gpt5n_a1_row and gpt5n_v2_cost and gpt5n_v3_cost and gpt5n_v2_cost > 0 and gpt5n_v3_cost > 0:
        avg_cost = (gpt5n_v2_cost + gpt5n_v3_cost) / 2.0
        gpt5n_a1_row["cost"] = avg_cost
            
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    metrics = ["cogn", "strat", "social", "total_pas", "total_ecs"]
    sorted_vals = {}
    # Only use regular models for ranking (exclude mixed_models)
    for m in metrics:
        vals = [r[m] for r in regular_rows if r[m] is not None]
        sorted_vals[m] = sorted(list(set(vals)), reverse=True)
    for c in ["cost"]:
        vals = sorted(list(set([r[c] for r in regular_rows if r[c] > 0])))
        sorted_vals[c] = vals  # For cost, lower is better

    lines = ["\\begin{table*}[h]", "\\centering", f"\\caption{{Table {table_num}: Model-Method Summary (PAS and ECS vs Cost){title_suffix}. ECS is Lin's Concordance Correlation Coefficient (CCC) between human and agent effect profiles (Cohen's d-equivalent). Best values highlighted in teal, worst in salmon.}}", f"\\label{{tab:pas-ecs-cost-only-{label_suffix}}}"]
    col_spec = "@{}ll" + " c" * 3 + " c c c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Cogn PAS} & \\textbf{Strat PAS} & \\textbf{Social PAS} & \\textbf{Total PAS} & \\textbf{Total ECS} & \\textbf{Cost (\\$)} \\\\")
    lines.append("\\midrule")
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        for m, fmt in [("cogn", "+.2f"), ("strat", "+.2f"), ("social", "+.2f")]:
            v = r[m]
            if v is not None:
                # Skip ranking for mixed_models
                rank = -999 if is_mixed else get_rank(v, sorted_vals[m], is_cost=False)
                cells.append(get_color_cell(f"{v:{fmt}}", rank))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
        
        # Total PAS (with ± std for mixed_models)
        v_pas = r["total_pas"]
        v_pas_std = r.get("total_pas_std")
        if v_pas_std is not None:
            # Mixed models: show mean ± std
            pas_str = f"${v_pas:+.4f} \\pm {v_pas_std:.4f}$"
        else:
            pas_str = f"{v_pas:+.4f}"
        # Skip ranking for mixed_models
        rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals["total_pas"], is_cost=False)
        cells.append(get_color_cell(pas_str, rank_pas))
        
        # Total ECS (without missing rate - moved to separate table)
        v_ecs = r["total_ecs"]
        ecs_str = f"{v_ecs:.3f}"
        # Skip ranking for mixed_models
        rank_ecs = -999 if is_mixed else get_rank(v_ecs, sorted_vals["total_ecs"], is_cost=False)
        cells.append(get_color_cell(ecs_str, rank_ecs))
        
        # Cost
        v_cost = r["cost"]
        # Skip ranking for mixed_models
        rank_cost = -999 if is_mixed else get_rank(v_cost, sorted_vals["cost"], is_cost=True)
        cells.append(get_color_cell(f"{v_cost:.4f}", rank_cost, is_cost=True))
        
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generate_latex_table_ecs_appendix(organized_data: Dict) -> str:
    """
    Generate Appendix Table: Pearson Correlation and Caricature Regression Parameters.
    This table does NOT include ECS_Strict or Total ECS (CCC).
    """
    rows_data = []
    
    for base_model, methods in organized_data.items():
        # Skip temp variants (e.g., Mistral Small Creative Temp0.x)
        if 'temp' in base_model.lower():
            continue
        for method, m_data in methods.items():
            studies = m_data.get("studies", [])
            if not studies:
                continue
            
            # Get Pearson r and caricature parameters (NO ECS_Strict)
            summary = m_data.get("summary", {})
            pearson_r = summary.get("ecs_corr_overall")  # Pearson correlation
            caricature = summary.get("caricature_overall", {})
            slope_a = caricature.get("a")
            intercept_b = caricature.get("b")
            
            rows_data.append({
                "base_model": base_model,
                "method": method,
                "pearson_r": pearson_r,
                "slope_a": slope_a,
                "intercept_b": intercept_b
            })
    
    if not rows_data:
        return "% No data for Pearson/Caricature table"
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: (x["base_model"], x["method"])) + mixed_rows
    
    # Get sorted values for ranking (only from regular models)
    pearson_r_vals = sorted(list(set([r["pearson_r"] for r in regular_rows if r["pearson_r"] is not None])), reverse=True)
    
    lines = ["\\begin{table*}[h]", "\\centering", 
             "\\caption{Appendix: Pearson Correlation and Caricature Regression Parameters. "
             "Pearson $r$ measures linear association between human and agent effect sizes. "
             "Caricature slope $a$ quantifies magnitude exaggeration ($a > 1$ = exaggeration, $a < 1$ = attenuation). "
             "Intercept $b$ indicates systematic bias. "
             "Best values highlighted in teal, worst in salmon.}",
             "\\label{tab:pearson-caricature}"]
    col_spec = "@{}ll c c c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Pearson $r$} & \\textbf{Slope $a$} & \\textbf{Intercept $b$} \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for r in rows_data_sorted:
        base_model = r["base_model"]
        is_mixed = (base_model == "mixed_models")
        method = METHOD_DISPLAY_MAP.get(r["method"], r["method"])
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        # Format values
        pearson_r_str = f"{r['pearson_r']:.3f}" if r['pearson_r'] is not None else "---"
        slope_a_str = f"{r['slope_a']:+.3f}" if r['slope_a'] is not None else "---"
        intercept_b_str = f"{r['intercept_b']:+.3f}" if r['intercept_b'] is not None else "---"
        
        # Get rank and color (skip ranking for mixed_models, only color Pearson r)
        rank_pearson = -999 if is_mixed else (get_rank(r["pearson_r"], pearson_r_vals, is_cost=False) if r["pearson_r"] is not None else -999)
        pearson_cell = get_color_cell(pearson_r_str, rank_pearson)
        
        # Don't color slope and intercept (interpretation depends on context)
        slope_cell = slope_a_str
        intercept_cell = intercept_b_str
        
        # Determine model cell content
        if base_model != curr_model:
            if curr_model is not None and not is_mixed:
                lines.append("\\midrule")
            curr_model = base_model
            model_display = format_model_name(base_model)
            n_rows = sum(1 for x in rows_data_sorted if x['base_model'] == base_model)
            model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{model_display}}}"
        else:
            model_cell = ""
        
        lines.append(f"{model_cell} & {method} & {pearson_cell} & {slope_cell} & {intercept_cell} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_latex_table_ecs_missing_rate(organized_data: Dict) -> str:
    """
    Generate separate table for ECS + Missing Rate.
    This table shows Total ECS and missing rate for each model-method combination.
    """
    rows_data = []
    
    for base_model, methods in organized_data.items():
        # Skip temp variants
        if 'temp' in base_model.lower():
            continue
        for method, m_data in methods.items():
            studies = m_data.get("studies", [])
            if not studies:
                continue
            
            # Get ECS and missing rate
            summary = m_data.get("summary", {})
            total_ecs = summary.get("ecs_overall") or summary.get("average_ecs") or summary.get("average_consistency_score", 0.0)
            ecs_missing_rate = summary.get("ecs_missing_rate_overall")
            
            rows_data.append({
                "base_model": base_model,
                "method": method,
                "total_ecs": total_ecs,
                "ecs_missing_rate": ecs_missing_rate
            })
    
    if not rows_data:
        return "% No data for ECS + Missing Rate table"
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: (x["base_model"], x["method"])) + mixed_rows
    
    # Get sorted values for ranking (only from regular models)
    ecs_vals = sorted(list(set([r["total_ecs"] for r in regular_rows if r["total_ecs"] is not None])), reverse=True)
    missing_rate_vals = sorted(list(set([r["ecs_missing_rate"] for r in regular_rows if r["ecs_missing_rate"] is not None and not (np.isnan(r["ecs_missing_rate"]) or np.isinf(r["ecs_missing_rate"]))])), reverse=False)  # Lower is better
    
    lines = ["\\begin{table*}[h]", "\\centering", 
             "\\caption{ECS and Missing Rate Summary. ECS is Lin's Concordance Correlation Coefficient (CCC) between human and agent effect profiles (Cohen's d-equivalent). Missing rate indicates the percentage of effect sizes that could not be calculated. Best ECS values highlighted in teal, worst in salmon. For missing rate, lower is better.}",
             "\\label{tab:ecs-missing-rate}"]
    col_spec = "@{}ll c c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Total ECS} & \\textbf{Missing Rate (\\%)} \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for r in rows_data_sorted:
        base_model = r["base_model"]
        is_mixed = (base_model == "mixed_models")
        method = METHOD_DISPLAY_MAP.get(r["method"], r["method"])
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        # Format values
        ecs_str = f"{r['total_ecs']:.3f}" if r['total_ecs'] is not None else "---"
        if r['ecs_missing_rate'] is not None and not (np.isnan(r['ecs_missing_rate']) or np.isinf(r['ecs_missing_rate'])):
            miss_pct = r['ecs_missing_rate'] * 100
            missing_rate_str = f"{miss_pct:.1f}"
        else:
            missing_rate_str = "---"
        
        # Get rank and color (skip ranking for mixed_models)
        rank_ecs = -999 if is_mixed else (get_rank(r["total_ecs"], ecs_vals, is_cost=False) if r["total_ecs"] is not None else -999)
        ecs_cell = get_color_cell(ecs_str, rank_ecs)
        
        rank_missing = -999 if is_mixed else (get_rank(r["ecs_missing_rate"], missing_rate_vals, is_cost=True) if r["ecs_missing_rate"] is not None and not (np.isnan(r["ecs_missing_rate"]) or np.isinf(r["ecs_missing_rate"])) else -999)
        missing_cell = get_color_cell(missing_rate_str, rank_missing, is_cost=True)
        
        # Determine model cell content
        if base_model != curr_model:
            if curr_model is not None and not is_mixed:
                lines.append("\\midrule")
            curr_model = base_model
            model_display = format_model_name(base_model)
            n_rows = sum(1 for x in rows_data_sorted if x['base_model'] == base_model)
            model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{model_display}}}"
        else:
            model_cell = ""
        
        lines.append(f"{model_cell} & {method} & {ecs_cell} & {missing_cell} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_latex_table_ecs_strict(organized_data: Dict) -> str:
    """
    Generate separate Appendix Table for ECS_Strict only.
    This provides the Z-diff p-value aggregation metric.
    """
    rows_data = []
    
    for base_model, methods in organized_data.items():
        # Skip temp variants
        if 'temp' in base_model.lower():
            continue
        for method, m_data in methods.items():
            studies = m_data.get("studies", [])
            if not studies:
                continue
            
            # Get ECS_Strict (Z-diff p-value aggregation)
            summary = m_data.get("summary", {})
            ecs_strict = summary.get("ecs_strict_overall") or summary.get("average_consistency_score", 0.0)
            
            rows_data.append({
                "base_model": base_model,
                "method": method,
                "ecs_strict": ecs_strict
            })
    
    if not rows_data:
        return "% No data for ECS_Strict table"
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: (x["base_model"], x["method"])) + mixed_rows
    
    # Get sorted values for ranking (only from regular models)
    ecs_strict_vals = sorted(list(set([r["ecs_strict"] for r in regular_rows if r["ecs_strict"] is not None])), reverse=True)
    
    lines = ["\\begin{table*}[h]", "\\centering", 
             "\\caption{Appendix: ECS\\_Strict Summary (RMS-Stouffer Method). "
             "ECS\\_Strict is the global $p$-value over the \\emph{entire benchmark}: (1) Test-level $Z_{j,k} = (\\delta_a - \\delta_h) / \\sqrt{SE_a^2 + SE_h^2}$; "
             "(2) Finding-level $\\chi^2_j = \\sum_k Z_{j,k}^2$, $p_j = P(\\chi^2_{K_j} \\geq \\chi^2_j)$; "
             "(3) Global: $Z_j^* = \\Phi^{-1}(1-p_j)$, $Z = (1/\\sqrt{m})\\sum_j Z_j^*$, $p_{\\text{global}} = 1-\\Phi(Z)$ over all findings $j$ in the benchmark. "
             "Higher values indicate better consistency. "
             "Interpretation: $p \\geq 0.10$ = no evidence of inconsistency; $0.05 \\leq p < 0.10$ = weak; $0.01 \\leq p < 0.05$ = moderate; $p < 0.01$ = strong. "
             "Distinct from Total ECS (CCC). Best in teal, worst in salmon.}",
             "\\label{tab:ecs-strict}"]
    col_spec = "@{}ll c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{ECS\\_Strict} \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for r in rows_data_sorted:
        base_model = r["base_model"]
        is_mixed = (base_model == "mixed_models")
        method = METHOD_DISPLAY_MAP.get(r["method"], r["method"])
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        # Format values
        ecs_strict_str = f"{r['ecs_strict']:.3f}" if r['ecs_strict'] is not None else "---"
        
        # Get rank and color (skip ranking for mixed_models)
        rank_ecs = -999 if is_mixed else (get_rank(r["ecs_strict"], ecs_strict_vals, is_cost=False) if r["ecs_strict"] is not None else -999)
        ecs_cell = get_color_cell(ecs_strict_str, rank_ecs)
        
        # Determine model cell content
        if base_model != curr_model:
            if curr_model is not None and not is_mixed:
                lines.append("\\midrule")
            curr_model = base_model
            model_display = format_model_name(base_model)
            n_rows = sum(1 for x in rows_data_sorted if x['base_model'] == base_model)
            model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{model_display}}}"
        else:
            model_cell = ""
        
        lines.append(f"{model_cell} & {method} & {ecs_cell} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_latex_table_pas_and_pas_agg(organized_data: Dict) -> str:
    """
    Generate Appendix Table: PAS raw (mean of per-study raw PAS in [0,1]) and PAS_agg (inverse-variance weighted).
    PAS column uses only study-level pas.raw (raw PAS [0,1]), not normalized or summary.average_pas_raw.
    """
    rows_data = []
    for base_model, methods in organized_data.items():
        if 'temp' in base_model.lower():
            continue
        for method, m_data in methods.items():
            studies = m_data.get("studies", [])
            if not studies:
                continue
            summary = m_data.get("summary", {})
            # PAS = mean of per-study raw PAS only (pas.raw in [0,1]). Do not use summary.average_pas_raw (it is normalized).
            raw_vals = [s.get("pas", {}).get("raw") for s in studies if s.get("pas", {}).get("raw") is not None]
            pas = float(np.mean(raw_vals)) if raw_vals else None
            if pas is None:
                pas = summary.get("average_pas_raw")  # fallback only when no study has pas.raw
            pas_agg = summary.get("pas_agg")
            pas_agg_se = summary.get("pas_agg_se")
            pas_mean_chain = summary.get("pas_mean_chain")
            rows_data.append({
                "base_model": base_model,
                "method": method,
                "pas": float(pas) if pas is not None else None,
                "pas_mean_chain": float(pas_mean_chain) if pas_mean_chain is not None else None,
                "pas_agg": float(pas_agg) if pas_agg is not None else None,
                "pas_agg_se": float(pas_agg_se) if pas_agg_se is not None else None,
            })
    if not rows_data:
        return "% No data for PAS / PAS_agg table"
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    rows_data_sorted = sorted(regular_rows, key=lambda x: (x["base_model"], x["method"])) + mixed_rows
    pas_vals = sorted(list(set([r["pas"] for r in regular_rows if r["pas"] is not None])), reverse=True)
    pas_mean_chain_vals = sorted(list(set([r["pas_mean_chain"] for r in regular_rows if r["pas_mean_chain"] is not None])), reverse=True)
    pas_agg_vals = sorted(list(set([r["pas_agg"] for r in regular_rows if r["pas_agg"] is not None])), reverse=True)
    lines = [
        "\\begin{table*}[h]", "\\centering",
        "\\caption{Appendix: PAS metrics. "
        "PAS (raw): mean of per-study raw PAS [0,1]. "
        "PAS (mean chain): finding-level = mean(tests), study-level = mean(findings), only test-level uses PAS; then mean across studies. "
        "PAS\\_agg: inverse-variance weighted across studies; PAS\\_agg SE its standard error. "
        "Best in teal, worst in salmon (by PAS mean chain).}",
        "\\label{tab:pas-pas-agg}"
    ]
    lines.append("\\begin{tabular}{@{}llcccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{PAS (raw)} & \\textbf{PAS (mean chain)} & \\textbf{PAS\\_agg} & \\textbf{PAS\\_agg SE} \\\\")
    lines.append("\\midrule")
    curr_model = None
    for r in rows_data_sorted:
        base_model = r["base_model"]
        is_mixed = (base_model == "mixed_models")
        method = METHOD_DISPLAY_MAP.get(r["method"], r["method"])
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        pas_str = f"{r['pas']:.4f}" if r['pas'] is not None else "---"
        pas_mc = r.get("pas_mean_chain")
        pas_mc_str = f"{pas_mc:.4f}" if pas_mc is not None else "---"
        pas_agg_str = f"{r['pas_agg']:.4f}" if r['pas_agg'] is not None else "---"
        pas_agg_se_str = f"{r['pas_agg_se']:.4f}" if r['pas_agg_se'] is not None and r['pas_agg_se'] >= 1e-10 else "---"
        rank_mc = -999 if is_mixed else (get_rank(pas_mc, pas_mean_chain_vals, is_cost=False) if pas_mc is not None else -999)
        pas_mc_cell = get_color_cell(pas_mc_str, rank_mc)
        if base_model != curr_model:
            if curr_model is not None and not is_mixed:
                lines.append("\\midrule")
            curr_model = base_model
            model_display = format_model_name(base_model)
            n_rows = sum(1 for x in rows_data_sorted if x['base_model'] == base_model)
            model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{model_display}}}"
        else:
            model_cell = ""
        lines.append(f"{model_cell} & {method} & ${pas_str}$ & {pas_mc_cell} & ${pas_agg_str}$ & ${pas_agg_se_str}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def generate_latex_table_detailed_study_breakdown_pas_only(organized_data: Dict, score_type: str = "raw") -> str:
    """Generate Table 5 Variant: Detailed Study Breakdown (PAS only at study level, PAS+ECS at global level)."""
    is_raw = True  # Always use raw for this table
    title_suffix = " (Raw PAS at Study Level, PAS+ECS at Global)"
    label_suffix = "pas-only-raw"
    study_ids = ["study_001", "study_002", "study_003", "study_004", "study_009", "study_010", "study_011", "study_012", "study_005", "study_006", "study_007", "study_008"]
    study_abbrs = ["S1", "S2", "S3", "S4", "S9", "S10", "S11", "S12", "S5", "S6", "S7", "S8"]
    
    rows_data = []
    for base_model in sorted(organized_data.keys()):
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            summary = m_data.get("summary", {})
            pas_scores = {s.get("study_id"): s.get("pas", {}).get("raw", 0.0) for s in studies}
            avg_pas = summary.get("average_pas_raw") or summary.get("average_pas", 0.0)
            avg_ecs = summary.get("ecs_overall") or summary.get("average_ecs") or summary.get("average_consistency_score", 0.0)  # CCC-based ECS
            ecs_missing_rate = summary.get("ecs_missing_rate_overall")  # Missing rate (0-1)
            rows_data.append({"base_model": base_model, "method": method, "pas_scores": pas_scores, "avg_pas": avg_pas, "avg_ecs": avg_ecs, "ecs_missing_rate": ecs_missing_rate})
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    sorted_vals_pas = {}
    # Only use regular models for ranking (exclude mixed_models)
    for sid in study_ids:
        pas_vals = sorted(list(set([r["pas_scores"].get(sid) for r in regular_rows if r["pas_scores"].get(sid) is not None])), reverse=True)
        sorted_vals_pas[sid] = pas_vals
    avg_pas_vals = sorted(list(set([r["avg_pas"] for r in regular_rows])), reverse=True)
    avg_ecs_vals = sorted(list(set([r["avg_ecs"] for r in regular_rows])), reverse=True)

    lines = ["\\begin{table*}[h]", "\\centering", f"\\caption{{Table 5 Variant: Detailed Study Breakdown{title_suffix}. Study columns show PAS (Raw) only. Average columns show both PAS (Raw) and ECS. ECS is Lin's Concordance Correlation Coefficient (CCC). Best values highlighted in teal, worst in salmon.}}", f"\\label{{tab:study-breakdown-{label_suffix}}}"]
    col_spec = "@{}ll" + "c" * 12 + "cc@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & " + " & ".join([f"\\textbf{{{a}}}" for a in study_abbrs]) + " & \\multicolumn{2}{c}{\\textbf{Avg}} \\\\")
    lines.append(" & & " + " & ".join([" "] * 12) + " & \\textbf{PAS} & \\textbf{ECS} \\\\")
    lines.append("\\midrule")
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        for sid in study_ids:
            v_pas = r["pas_scores"].get(sid)
            if v_pas is not None:
                # Skip ranking for mixed_models
                rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals_pas[sid], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank_pas))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
        # Average columns
        # Skip ranking for mixed_models
        rank_pas_avg = -999 if is_mixed else get_rank(r["avg_pas"], avg_pas_vals, is_cost=False)
        rank_ecs_avg = -999 if is_mixed else get_rank(r["avg_ecs"], avg_ecs_vals, is_cost=False)
        cells.append(get_color_cell(f"{r['avg_pas']:+.4f}", rank_pas_avg))
        
        # Avg ECS (without missing rate - moved to separate table)
        ecs_str = f"{r['avg_ecs']:.3f}"
        cells.append(get_color_cell(ecs_str, rank_ecs_avg))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generate_latex_table_study_subfield_breakdown(organized_data: Dict) -> str:
    """Generate Table: Study Level and Subfield Breakdown (PAS and ECS)."""
    study_ids = ["study_001", "study_002", "study_003", "study_004", "study_009", "study_010", "study_011", "study_012", "study_005", "study_006", "study_007", "study_008"]
    
    rows_data = []
    for base_model in sorted(organized_data.keys()):
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            summary = m_data.get("summary", {})
            
            # Per-study metrics
            pas_scores = {s.get("study_id"): s.get("pas", {}).get("raw", 0.0) for s in studies}
            ecs_per_study = summary.get("ecs_per_study", {})
            
            # Per-subfield (domain) metrics
            subfield_pas = {}
            subfield_ecs = {}
            for gn in ["Cognition", "Strategic", "Social"]:
                ids = STUDY_GROUPS[gn]
                pas_list = [s.get("pas", {}).get("raw", 0.0) for s in studies if s.get("study_id") in ids]
                subfield_pas[gn] = np.mean(pas_list) if pas_list else None
                
                # ECS per domain from summary
                ecs_domain = summary.get("ecs_domain", {})
                subfield_ecs[gn] = ecs_domain.get(gn)
            
            rows_data.append({
                "base_model": base_model, 
                "method": method, 
                "pas_scores": pas_scores,
                "ecs_scores": {s: ecs_per_study.get(s) for s in study_ids if ecs_per_study.get(s) is not None},
                "subfield_pas": subfield_pas,
                "subfield_ecs": subfield_ecs
            })
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    # Compute sorted values for ranking (only from regular models)
    sorted_vals_pas = {}
    sorted_vals_ecs = {}
    for sid in study_ids:
        pas_vals = sorted(list(set([r["pas_scores"].get(sid) for r in regular_rows if r["pas_scores"].get(sid) is not None])), reverse=True)
        ecs_vals = sorted(list(set([r["ecs_scores"].get(sid) for r in regular_rows if r["ecs_scores"].get(sid) is not None])), reverse=True)
        sorted_vals_pas[sid] = pas_vals
        sorted_vals_ecs[sid] = ecs_vals
    
    sorted_vals_subfield_pas = {}
    sorted_vals_subfield_ecs = {}
    for gn in ["Cognition", "Strategic", "Social"]:
        pas_vals = sorted(list(set([r["subfield_pas"].get(gn) for r in regular_rows if r["subfield_pas"].get(gn) is not None])), reverse=True)
        ecs_vals = sorted(list(set([r["subfield_ecs"].get(gn) for r in regular_rows if r["subfield_ecs"].get(gn) is not None])), reverse=True)
        sorted_vals_subfield_pas[gn] = pas_vals
        sorted_vals_subfield_ecs[gn] = ecs_vals

    lines = ["\\begin{table*}[p]", "\\centering", "\\caption{Study Level and Subfield Breakdown (PAS and ECS). Rows show individual studies followed by subfield aggregates. ECS is Lin's Concordance Correlation Coefficient (CCC). Best values highlighted in teal, worst in salmon.}", "\\label{tab:study-subfield-breakdown}"]
    col_spec = "@{}ll" + "cc" * 15 + "@{}"  # 12 studies + 3 subfields
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & " + 
                 " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{S{i+1}}}}}" for i in range(12)]) + " & " +
                 " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{sf}}}}}" for sf in ["Cogn", "Strat", "Social"]]) + " \\\\")
    lines.append(" & & " + " & ".join([" & \\textbf{ECS}"] * 15) + " \\\\")
    lines.append("\\midrule")
    
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        
        # Study-level PAS and ECS
        for sid in study_ids:
            v_pas = r["pas_scores"].get(sid)
            v_ecs = r["ecs_scores"].get(sid)
            if v_pas is not None:
                rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals_pas[sid], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank_pas))
            else:
                cells.append("{--}")
            if v_ecs is not None:
                rank_ecs = -999 if is_mixed else get_rank(v_ecs, sorted_vals_ecs[sid], is_cost=False)
                cells.append(get_color_cell(f"{v_ecs:.3f}", rank_ecs))
            else:
                cells.append("{--}")
        
        # Subfield-level PAS and ECS
        for gn in ["Cognition", "Strategic", "Social"]:
            v_pas = r["subfield_pas"].get(gn)
            v_ecs = r["subfield_ecs"].get(gn)
            if v_pas is not None:
                rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals_subfield_pas[gn], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank_pas))
            else:
                cells.append("{--}")
            if v_ecs is not None:
                rank_ecs = -999 if is_mixed else get_rank(v_ecs, sorted_vals_subfield_ecs[gn], is_cost=False)
                cells.append(get_color_cell(f"{v_ecs:.3f}", rank_ecs))
            else:
                cells.append("{--}")
        
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generate_latex_table_detailed_study_breakdown(organized_data: Dict, score_type: str = "normalized") -> str:
    """Generate Table 5: Detailed Study Breakdown (PAS and ECS)."""
    is_raw = (score_type == "raw")
    title_suffix = " (Raw)" if is_raw else " (Normalized)"
    label_suffix = "raw" if is_raw else "norm"
    study_ids = ["study_001", "study_002", "study_003", "study_004", "study_009", "study_010", "study_011", "study_012", "study_005", "study_006", "study_007", "study_008"]
    study_abbrs = ["S1", "S2", "S3", "S4", "S9", "S10", "S11", "S12", "S5", "S6", "S7", "S8"]
    
    rows_data = []
    for base_model in sorted(organized_data.keys()):
        if 'temp' in base_model.lower(): continue
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            summary = m_data.get("summary", {})
            pas_scores = {s.get("study_id"): s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies}
            # ECS values (CCC-based) are stored in summary.ecs_per_study, not in individual study objects
            ecs_per_study = summary.get("ecs_per_study", {})  # CCC-based ECS
            ecs_scores = {}
            for s in studies:
                study_id = s.get("study_id")
                if study_id:
                    ecs_value = ecs_per_study.get(study_id)  # CCC-based ECS
                    # Only include if ECS value exists (not None)
                    if ecs_value is not None:
                        ecs_scores[study_id] = ecs_value
            avg_pas = (summary.get("average_pas_raw") or summary.get("average_pas", 0.0)) if not is_raw else np.mean(list(pas_scores.values())) if pas_scores else 0.0
            avg_ecs = summary.get("ecs_overall") or summary.get("average_ecs") or summary.get("average_consistency_score", 0.0)  # CCC-based ECS
            ecs_missing_rate = summary.get("ecs_missing_rate_overall")  # Missing rate (0-1)
            rows_data.append({"base_model": base_model, "method": method, "pas_scores": pas_scores, "ecs_scores": ecs_scores, "avg_pas": avg_pas, "avg_ecs": avg_ecs, "ecs_missing_rate": ecs_missing_rate})
    
    # Separate mixed_models from other models
    regular_rows = [r for r in rows_data if r["base_model"] != "mixed_models"]
    mixed_rows = [r for r in rows_data if r["base_model"] == "mixed_models"]
    
    # Sort regular models, then append mixed_models at the end
    rows_data_sorted = sorted(regular_rows, key=lambda x: x["base_model"]) + mixed_rows
    
    sorted_vals_pas = {}
    sorted_vals_ecs = {}
    # Only use regular models for ranking (exclude mixed_models)
    for sid in study_ids:
        pas_vals = sorted(list(set([r["pas_scores"].get(sid) for r in regular_rows if r["pas_scores"].get(sid) is not None])), reverse=True)
        ecs_vals = sorted(list(set([r["ecs_scores"].get(sid) for r in regular_rows if r["ecs_scores"].get(sid) is not None])), reverse=True)
        sorted_vals_pas[sid] = pas_vals
        sorted_vals_ecs[sid] = ecs_vals
    avg_pas_vals = sorted(list(set([r["avg_pas"] for r in regular_rows])), reverse=True)
    avg_ecs_vals = sorted(list(set([r["avg_ecs"] for r in regular_rows])), reverse=True)

    lines = ["\\begin{table*}[h]", "\\centering", f"\\caption{{Table 5: Detailed Study Breakdown (PAS and ECS){title_suffix}. ECS is Lin's Concordance Correlation Coefficient (CCC) between human and agent effect profiles (Cohen's d-equivalent). Best values highlighted in teal, worst in salmon.}}", f"\\label{{tab:study-breakdown-{label_suffix}}}"]
    col_spec = "@{}ll" + "cc" * 12 + "cc@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Method} & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{a}}}}}" for a in study_abbrs]) + " & \\multicolumn{2}{c}{\\textbf{Avg}} \\\\")
    lines.append(" & & " + " & ".join([" & \\textbf{ECS}"] * 12) + " & \\textbf{PAS} & \\textbf{ECS} \\\\")
    lines.append("\\midrule")
    curr_model = None
    for r in rows_data_sorted:
        is_mixed = (r["base_model"] == "mixed_models")
        
        # Add double line separator before mixed_models
        if is_mixed and curr_model != "mixed_models":
            lines.append("\\midrule")
            lines.append("\\midrule")
        
        m_disp = format_model_name(r["base_model"])
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data_sorted if x['base_model'] == r['base_model'])}}}{{*}}{{{m_disp}}}" if r["base_model"] != curr_model else ""
        if r["base_model"] != curr_model and curr_model is not None and not is_mixed: 
            lines.append("\\midrule")
        curr_model = r["base_model"]
        cells = [m_cell, meth_disp]
        for sid in study_ids:
            v_pas = r["pas_scores"].get(sid)
            v_ecs = r["ecs_scores"].get(sid)
            if v_pas is not None:
                # Skip ranking for mixed_models
                rank_pas = -999 if is_mixed else get_rank(v_pas, sorted_vals_pas[sid], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank_pas))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
            if v_ecs is not None:
                # Skip ranking for mixed_models
                rank_ecs = -999 if is_mixed else get_rank(v_ecs, sorted_vals_ecs[sid], is_cost=False)
                cells.append(get_color_cell(f"{v_ecs:.3f}", rank_ecs))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
        # Average columns
        # Skip ranking for mixed_models
        rank_pas_avg = -999 if is_mixed else get_rank(r["avg_pas"], avg_pas_vals, is_cost=False)
        rank_ecs_avg = -999 if is_mixed else get_rank(r["avg_ecs"], avg_ecs_vals, is_cost=False)
        cells.append(get_color_cell(f"{r['avg_pas']:+.4f}", rank_pas_avg))
        
        # Avg ECS (without missing rate - moved to separate table)
        ecs_str = f"{r['avg_ecs']:.3f}"
        cells.append(get_color_cell(ecs_str, rank_ecs_avg))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def extract_temperature_from_model_name(model_key: str) -> Optional[float]:
    """Extract temperature from model name."""
    if 'temp' in model_key.lower():
        import re
        match = re.search(r'temp([0-9.]+)', model_key.lower())
        if match: return float(match.group(1))
    if 'mistral_small_creative' in model_key.lower(): return 1.0
    return None


def generate_latex_table_temperature_ablation(organized_data: Dict, score_type: str = "normalized") -> str:
    """Generate Table 6: Temperature Ablation Study (Mistral Creative) structured like Table 1."""
    is_raw = (score_type == "raw")
    title_suffix = " (Raw)" if is_raw else " (Normalized)"
    label_suffix = "raw" if is_raw else "norm"
    
    rows_data = []
    # Identify Mistral Small Creative models and extract their temperatures
    msc_models = [k for k in organized_data.keys() if 'mistral_small_creative' in k.lower()]
    
    # Sort models by temperature
    sorted_msc = sorted(msc_models, key=lambda k: extract_temperature_from_model_name(k) or 0)
    
    for base_model in sorted_msc:
        temp = extract_temperature_from_model_name(base_model)
        if temp is None: continue
            
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        
        for method in sorted_methods:
            m_data = methods[method]
            studies = m_data.get("studies", [])
            sub_scores = {}
            for gn in ["Cognition", "Strategic", "Social"]:
                ids = STUDY_GROUPS[gn]
                pas_list = [s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies if s.get("study_id") in ids]
                sub_scores[gn] = {"pas": np.mean(pas_list) if pas_list else None}
            
            rows_data.append({
                "temp": temp, "method": method,
                "cogn": sub_scores["Cognition"], "strat": sub_scores["Strategic"], "social": sub_scores["Social"],
                "total_pas": np.mean([s.get("pas", {}).get("raw" if is_raw else "normalized", 0.0) for s in studies]) if studies else 0.0 if is_raw else (m_data.get("summary", {}).get("average_pas_raw") or m_data.get("summary", {}).get("average_pas", 0.0)),
                "total_ecs": m_data.get("summary", {}).get("ecs_overall") or m_data.get("summary", {}).get("average_ecs") or m_data.get("summary", {}).get("average_consistency_score", 0.0),  # CCC-based ECS
                "cost": m_data.get("summary", {}).get("total_usage", {}).get("total_cost", 0.0)
            })
            
    if not rows_data: return "% No temperature ablation data found"

    metrics = ["total_pas", "total_ecs"]
    for g in ["cogn", "strat", "social"]: metrics.extend([f"{g}_pas"])
    sorted_vals = {}
    for m in metrics:
        if "_" in m and m.split("_")[0] in ["cogn", "strat", "social"]:
            g, sm = m.split("_")
            vals = [r[g][sm] for r in rows_data if r[g][sm] is not None]
        else:
            vals = [r[m] for r in rows_data if r[m] is not None]
        sorted_vals[m] = sorted(list(set(vals)), reverse=True)
    for c in ["cost"]:
        vals = sorted(list(set([r[c] for r in rows_data if r[c] > 0])))
        sorted_vals[c] = vals  # For cost, lower is better

    lines = ["\\begin{table*}[h]", "\\centering", f"\\caption{{Table 6: Temperature Ablation Study (Mistral Creative){title_suffix}. ECS is Lin's Concordance Correlation Coefficient (CCC) between human and agent effect profiles (Cohen's d-equivalent). Best values highlighted in teal, worst in salmon.}}", "\\label{tab:temp-ablation}"]
    col_spec = "@{}ll" + " c" * 3 + " c c c@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Temp} & \\textbf{Method} & \\textbf{Cogn PAS} & \\textbf{Strat PAS} & \\textbf{Social PAS} & \\textbf{Total PAS} & \\textbf{Total ECS} & \\textbf{Cost (\\$)} \\\\")
    lines.append("\\midrule")
    
    curr_temp = None
    for r in rows_data:
        temp_disp = f"{r['temp']:.1f}"
        meth_disp = METHOD_DISPLAY_MAP.get(r["method"], escape_latex(r["method"].replace("_", "-")))
        m_cell = f"\\multirow{{{sum(1 for x in rows_data if x['temp'] == r['temp'])}}}{{*}}{{{temp_disp}}}" if r["temp"] != curr_temp else ""
        if r["temp"] != curr_temp and curr_temp is not None: lines.append("\\midrule")
        curr_temp = r["temp"]
        cells = [m_cell, meth_disp]
        for g in ["cogn", "strat", "social"]:
            v_pas = r[g]["pas"]
            if v_pas is not None:
                rank = get_rank(v_pas, sorted_vals[f"{g}_pas"], is_cost=False)
                cells.append(get_color_cell(f"{v_pas:+.2f}", rank))
            else:
                cells.append("{--}")  # Wrap in braces for siunitx
        for m, fmt in [("total_pas", "+.4f"), ("total_ecs", ".3f"), ("cost", ".4f")]:
            v = r[m]
            is_cost_col = (m == "cost")
            rank = get_rank(v, sorted_vals[m], is_cost=is_cost_col)
            cells.append(get_color_cell(f"{v:{fmt}}", rank, is_cost=is_cost_col))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="results/benchmark_summary.json")
    parser.add_argument("--results-dir", "--output-dir", dest="results_dir", default="results",
                        help="Results/benchmark directory (default: results)")
    parser.add_argument("--output-latex", default="results/production_tables.tex")
    args = parser.parse_args()
    
    data = load_benchmark_data(Path(args.summary_json))
    random_alignment_stats = load_random_alignment_stats(Path(args.results_dir))
    organized = organize_data_by_model_method(data, random_alignment_stats, results_dir=Path(args.results_dir))
    
    with open(args.output_latex, 'w') as f:
        f.write("\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{siunitx}\n\\usepackage{graphicx}\n\\usepackage[table]{xcolor}\n\\usepackage{makecell}\n\\usepackage{subcaption}\n\n")
        # Professional Muted Palette (inspired by research paper styles)
        # 4*2 color scheme: 4 best colors (Teal shades) + 4 worst colors (Salmon/Coral shades)
        f.write("\\definecolor{best1}{HTML}{B2DFDB}\n")      # Light Teal (best)
        f.write("\\definecolor{best2}{HTML}{CDEAE8}\n")      # Lighter Teal (2nd)
        f.write("\\definecolor{best3}{HTML}{E0F2F1}\n")      # Very light Teal (3rd)
        f.write("\\definecolor{best4}{HTML}{F1F8F7}\n")      # Almost white Teal (4th)
        f.write("\\definecolor{worst1}{HTML}{FFCDD2}\n")     # Light Salmon/Coral (worst)
        f.write("\\definecolor{worst2}{HTML}{F8E1E3}\n")     # Lighter Salmon (2nd worst)
        f.write("\\definecolor{worst3}{HTML}{FCEBED}\n")     # Very light Salmon (3rd worst)
        f.write("\\definecolor{worst4}{HTML}{FFF8F9}\n")     # Almost white Salmon (4th worst)
        f.write("\n")
        f.write("\\begin{document}\n\n")
        # Production tables: (1) basic summary, (2) full test-level detail
        f.write(generate_latex_table_main_summary(organized, "normalized") + "\n\n\\newpage\n\n")
        f.write(generate_latex_table_main_summary(organized, "raw") + "\n\n\\newpage\n\n")
        try:
            from scripts.generate_detailed_metrics_table import generate_detailed_metrics_table
            detailed_table = generate_detailed_metrics_table(organized, Path(args.results_dir))
            f.write(detailed_table + "\n\n\\newpage\n\n")
            print("✓ Full-detail metrics table generated")
        except Exception as e:
            print(f"Warning: Failed to generate detailed metrics table: {e}")
            import traceback
            traceback.print_exc()
        f.write("\\end{document}\n")
    
    print("✓ Production results complete!")

if __name__ == "__main__": main()
