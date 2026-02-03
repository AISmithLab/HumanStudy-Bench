#!/usr/bin/env python3
"""
HumanStudy-Bench Results Table Generator (ECS_corr only)

Aggregates benchmark results and outputs ECS_corr (Lin's CCC) only.
Generates benchmark_summary.json, markdown summary table, and CSV.
"""

import json
import argparse
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_usage_from_individual_data(benchmark_data: dict) -> dict:
    """Calculate usage stats from individual_data if usage_stats is missing or zero."""
    usage_stats = benchmark_data.get('usage_stats', {})
    if usage_stats.get('total_tokens', 0) > 0:
        return usage_stats
    individual_data = benchmark_data.get('individual_data', [])
    if not individual_data:
        return usage_stats
    is_flat = len(individual_data) > 0 and 'responses' not in individual_data[0]
    total_prompt_tokens = total_completion_tokens = total_tokens = 0
    total_cost = 0.0
    participant_count = 0
    if is_flat:
        seen = set()
        for resp in individual_data:
            pid = resp.get('participant_id')
            if pid not in seen:
                seen.add(pid)
                participant_count += 1
            usage = resp.get('usage', {})
            if usage and usage.get('total_tokens', 0):
                total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                total_completion_tokens += usage.get('completion_tokens', 0) or 0
                total_tokens += usage.get('total_tokens', 0) or 0
                total_cost += usage.get('cost', 0.0) or 0.0
    else:
        participant_count = len(individual_data)
        for participant in individual_data:
            u = participant.get('usage', {})
            if u and u.get('total_tokens', 0):
                total_prompt_tokens += u.get('prompt_tokens', 0) or 0
                total_completion_tokens += u.get('completion_tokens', 0) or 0
                total_tokens += u.get('total_tokens', 0) or 0
                total_cost += u.get('cost', 0.0) or 0.0
            else:
                for resp in participant.get('responses', []):
                    u = resp.get('usage', {})
                    if u:
                        total_prompt_tokens += u.get('prompt_tokens', 0) or 0
                        total_completion_tokens += u.get('completion_tokens', 0) or 0
                        total_tokens += u.get('total_tokens', 0) or 0
                        total_cost += u.get('cost', 0.0) or 0.0
    if participant_count > 0 and total_tokens > 0:
        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": float(total_cost),
            "avg_tokens_per_participant": float(total_tokens / participant_count),
            "avg_cost_per_participant": float(total_cost / participant_count),
        }
    return usage_stats


def load_all_results(results_dir: Path) -> dict:
    """Load all benchmark results. Returns {model_config: {study_id: study_result}}."""
    all_results = defaultdict(dict)
    if results_dir.name != "results" and (results_dir / results_dir.name).exists():
        search_base = results_dir / results_dir.name
    elif "benchmark" in str(results_dir):
        search_base = results_dir
    else:
        subdirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in ["benchmark", "benchmark_baseline", "runs"]]
        search_base = subdirs[0] if subdirs else results_dir

    for study_dir in search_base.iterdir():
        if not study_dir.is_dir() or not study_dir.name.startswith("study_"):
            continue
        study_id = study_dir.name
        for config_dir in study_dir.iterdir():
            if not config_dir.is_dir():
                continue
            benchmark_file = config_dir / "full_benchmark.json"
            eval_file = config_dir / "evaluation_results.json"
            if not benchmark_file.exists():
                continue
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as fp:
                    benchmark_data = json.load(fp)
                eval_data = {}
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r', encoding='utf-8') as fp:
                            eval_data = json.load(fp)
                    except Exception:
                        pass
                usage_stats = calculate_usage_from_individual_data(benchmark_data)
                config_name = config_dir.name
                study_result = {
                    'study_id': study_id,
                    'title': benchmark_data.get('title', 'N/A'),
                    'model': benchmark_data.get('model', 'unknown'),
                    'system_prompt_preset': benchmark_data.get('system_prompt_preset', 'unknown'),
                    'n_participants': len(benchmark_data.get('individual_data', [])),
                    'usage_stats': usage_stats,
                    'pas_result': eval_data,
                    '_mtime': benchmark_file.stat().st_mtime,
                    '_config': config_name,
                }
                if study_id not in all_results[config_name] or study_result['_mtime'] > all_results[config_name][study_id].get('_mtime', 0):
                    all_results[config_name][study_id] = study_result
            except Exception as e:
                print(f"Warning: Could not load {benchmark_file}: {e}")

    return all_results


def _get_ecs_corr(study_result: dict) -> float:
    """Get ECS_corr (CCC) for a study from pas_result."""
    pr = study_result.get('pas_result', {})
    ecs = pr.get('ecs_corr')
    if ecs is not None:
        return float(ecs)
    ecs = pr.get('ecs_corr_study')
    if ecs is not None:
        return float(ecs)
    details = pr.get('ecs_corr_details', {})
    per_study = details.get('ecs_per_study', {})
    sid = study_result.get('study_id')
    if sid and sid in per_study and per_study[sid] is not None:
        return float(per_study[sid])
    return np.nan


def generate_summary_table(results: dict, **kwargs) -> str:
    """Generate summary table: Study | Title | ECS (CCC) | Fail % | Cost | Status."""
    lines = []
    for model, studies in sorted(results.items()):
        lines.append(f"\n{'='*100}")
        lines.append(f"Model: {model}")
        lines.append(f"{'='*100}\n")
        lines.append("| Study | Title | ECS (CCC) | Raw Fail % | Final Fail % | Total Tokens | Total Cost | Status |")
        lines.append("|-------|-------|-----------|------------|--------------|--------------|------------|--------|")
        total_cost = 0.0
        total_tokens = 0
        count = 0
        for sid in sorted(studies.keys()):
            s = studies[sid]
            ecs = _get_ecs_corr(s)
            pr = s.get('pas_result', {})
            fr = pr.get('failure_rates', {})
            raw_fail = fr.get('raw_failure_rate')
            final_fail = fr.get('final_failure_rate')
            usage = s.get('usage_stats', {}) or pr.get('usage_stats', {})
            tot_tok = usage.get('total_tokens', 0)
            tot_cost = usage.get('total_cost', 0.0)
            total_tokens += tot_tok
            total_cost += tot_cost
            count += 1
            ecs_str = f"{ecs:.4f}" if not np.isnan(ecs) else "N/A"
            raw_str = f"{raw_fail:.1f}%" if raw_fail is not None else "--"
            final_str = f"{final_fail:.1f}%" if final_fail is not None else "--"
            status = "pass" if (not np.isnan(ecs) and ecs > 0) else "fail"
            title = (s.get('title', 'N/A')[:30]).replace('|', ' ')
            lines.append(f"| {sid} | {title} | {ecs_str} | {raw_str} | {final_str} | {tot_tok} | {tot_cost:.4f} | {status} |")
        if count:
            lines.append(f"\n**Summary**: {count} studies | Total Cost: ${total_cost:.4f} | Total Tokens: {total_tokens:,}\n")
    return "\n".join(lines) if lines else "No results."


def generate_json_summary(results: dict, **kwargs) -> dict:
    """Build benchmark_summary.json with ECS_corr (CCC) only."""
    from src.evaluation.stats_lib import compute_ecs_corr

    STUDY_GROUPS = {
        "Cognition": ["study_001", "study_002", "study_003", "study_004"],
        "Strategic": ["study_009", "study_010", "study_011", "study_012"],
        "Social": ["study_005", "study_006", "study_007", "study_008"],
    }

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "format": "humanstudy_bench_summary_ecs_only",
        },
        "models": {},
    }

    for model, studies in sorted(results.items()):
        all_test_results = []
        for sid in sorted(studies.keys()):
            s = studies[sid]
            pr = s.get('pas_result', {})
            for t in pr.get('test_results', []):
                t_copy = dict(t)
                t_copy['study_id'] = sid
                all_test_results.append(t_copy)

        overall_ecs = None
        ecs_per_study = {}
        ecs_domain = {}
        caricature_overall = {'a': None, 'b': None}
        n_tests_total = len(all_test_results)
        n_tests_valid = 0
        ecs_missing_rate_overall = None

        if all_test_results:
            ecs_result = compute_ecs_corr(all_test_results, study_groups=STUDY_GROUPS)
            overall_ecs = ecs_result.get('ecs_overall')
            ecs_per_study = ecs_result.get('ecs_per_study', {})
            ecs_domain = ecs_result.get('ecs_domain', {})
            caricature_overall = ecs_result.get('caricature_overall', {'a': None, 'b': None})
            n_tests_valid = ecs_result.get('n_tests_overall', 0)
            if n_tests_total > 0:
                ecs_missing_rate_overall = 1.0 - (n_tests_valid / n_tests_total)

        total_input_tokens = total_output_tokens = total_tokens = 0
        total_cost = 0.0
        model_studies = []
        for sid in sorted(studies.keys()):
            s = studies[sid]
            pr = s.get('pas_result', {})
            usage = s.get('usage_stats', {}) or pr.get('usage_stats', {})
            ti = usage.get('total_prompt_tokens', 0)
            to = usage.get('total_completion_tokens', 0)
            tt = usage.get('total_tokens', 0)
            tc = usage.get('total_cost', 0.0)
            total_input_tokens += ti
            total_output_tokens += to
            total_tokens += tt
            total_cost += tc
            ecs_study = _get_ecs_corr(s)
            model_studies.append({
                "study_id": sid,
                "title": s.get('title', 'N/A'),
                "ecs_corr": float(ecs_study) if not np.isnan(ecs_study) else None,
                "usage": {"total_prompt_tokens": ti, "total_completion_tokens": to, "total_tokens": tt, "total_cost": tc},
                "failure_rates": pr.get('failure_rates', {}),
                "n_participants": s.get('n_participants', 0),
            })

        output["models"][model] = {
            "summary": {
                "ecs_overall": overall_ecs,
                "average_ecs": float(overall_ecs) if overall_ecs is not None else None,
                "ecs_per_study": ecs_per_study,
                "ecs_domain": ecs_domain,
                "caricature_overall": caricature_overall,
                "ecs_missing_rate_overall": ecs_missing_rate_overall,
                "ecs_n_tests_total": n_tests_total,
                "ecs_n_tests_valid": n_tests_valid,
                "total_usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_tokens,
                    "total_cost": float(total_cost),
                },
            },
            "studies": model_studies,
        }

    return output


def generate_csv(results: dict) -> str:
    """CSV with model, study_id, title, ecs_corr, status, cost."""
    lines = ["model,study_id,title,ecs_corr,status,n_participants,test_count,total_tokens,total_cost"]
    for model, studies in sorted(results.items()):
        for sid in sorted(studies.keys()):
            s = studies[sid]
            ecs = _get_ecs_corr(s)
            pr = s.get('pas_result', {})
            usage = s.get('usage_stats', {}) or pr.get('usage_stats', {})
            total_tokens = usage.get('total_tokens', 0)
            total_cost = usage.get('total_cost', 0.0)
            test_count = len(pr.get('test_results', []))
            ecs_str = f"{ecs:.4f}" if not np.isnan(ecs) else ""
            status = "pass" if (not np.isnan(ecs) and ecs > 0) else "fail"
            title = (s.get('title', 'N/A')[:50]).replace(',', ';')
            lines.append(f"{model},{sid},{title},{ecs_str},{status},{s.get('n_participants', 0)},{test_count},{total_tokens},{total_cost:.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate HumanStudy-Bench results (ECS_corr only)")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--format", choices=['summary', 'detailed', 'csv', 'json', 'all'], default='all', help="Output format")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return 1

    results = load_all_results(results_dir)
    if not results:
        print("No results found!")
        return 1

    output_lines = [
        "# HumanStudy-Bench Results Report (ECS_corr only)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    if args.format in ['summary', 'all']:
        output_lines.append("\n# SUMMARY TABLE\n")
        output_lines.append(generate_summary_table(results))

    if args.format in ['detailed', 'all']:
        output_lines.append("\n# STUDY LIST (ECS_corr)\n")
        for model, studies in sorted(results.items()):
            output_lines.append(f"\n## {model}\n")
            for sid in sorted(studies.keys()):
                s = studies[sid]
                ecs = _get_ecs_corr(s)
                ecs_str = f"{ecs:.4f}" if not np.isnan(ecs) else "N/A"
                output_lines.append(f"- {sid}: ECS_corr = {ecs_str}")

    if args.format in ['csv', 'all']:
        output_lines.append("\n# CSV DATA\n```csv")
        output_lines.append(generate_csv(results))
        output_lines.append("```")

    if args.format in ['json', 'all']:
        json_data = generate_json_summary(results)
        if args.output:
            json_path = Path(args.output)
            if json_path.suffix != '.json':
                json_path = json_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"JSON summary saved to {json_path}")
        elif args.format == 'json':
            print(json.dumps(json_data, indent=2, ensure_ascii=False))

    if args.format != 'json':
        text = "\n".join(output_lines)
        if args.output:
            out_path = Path(args.output)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Results saved to {out_path}")
        else:
            print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
