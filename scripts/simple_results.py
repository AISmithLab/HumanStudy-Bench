#!/usr/bin/env python3
"""
HumanStudy-Bench Simple Results Table

User-friendly script that outputs:
- Benchmark-level: final PAS_raw, final ECS (CCC), total tokens, total cost
- Per-study: PAS_raw, tokens, cost, ECS (if available)
- Per-finding: finding score (PAS_raw) with stable sequential index 0..N-1

Data sources: evaluation_results.json (scores), full_benchmark.json (tokens/cost).
"""

import json
import argparse
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

STUDY_GROUPS = {
    "Cognition": ["study_001", "study_002", "study_003", "study_004"],
    "Strategic": ["study_009", "study_010", "study_011", "study_012"],
    "Social": ["study_005", "study_006", "study_007", "study_008"],
}


def _sort_key_finding(fr: dict) -> Tuple[str, int, str]:
    """Deterministic sort key for finding_results: sub_study_id, numeric part of finding_id, original finding_id."""
    sub = fr.get("sub_study_id", "")
    fid = fr.get("finding_id", "")
    m = re.match(r"F(\d+)", str(fid), re.I)
    num = int(m.group(1)) if m else 999
    return (sub, num, fid)


def calculate_usage(benchmark_data: dict, raw_responses_path: Optional[Path] = None) -> dict:
    """Get usage stats from full_benchmark, or compute from individual_data/raw_responses."""
    usage = benchmark_data.get("usage_stats", {})
    if usage.get("total_tokens", 0) > 0:
        return usage

    # Fallback: individual_data in full_benchmark
    individual_data = benchmark_data.get("individual_data", [])
    if individual_data:
        is_flat = "responses" not in individual_data[0] if individual_data else True
        total_tok = total_cost = 0
        seen = set()
        for item in individual_data:
            if is_flat:
                pid = item.get("participant_id")
                if pid not in seen:
                    seen.add(pid)
                u = item.get("usage", {})
            else:
                u = item.get("usage", {})
                if not u:
                    for r in item.get("responses", []):
                        u = r.get("usage", {})
                        if u:
                            break
            if u and u.get("total_tokens", 0):
                total_tok += u.get("total_tokens", 0)
                total_cost += u.get("cost", 0.0) or 0.0
        if total_tok > 0:
            n = len(seen) if is_flat else len(individual_data)
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": total_tok,
                "total_cost": float(total_cost),
                "avg_tokens_per_participant": total_tok / n if n else 0,
                "avg_cost_per_participant": total_cost / n if n else 0,
            }

    # Fallback: raw_responses.json
    if raw_responses_path and raw_responses_path.exists():
        try:
            with open(raw_responses_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            total_tok = total_cost = 0
            for run in raw.get("all_runs_raw_responses", []):
                for p in run.get("participants", []):
                    for r in p.get("raw_responses", []):
                        u = r.get("usage", {})
                        if u:
                            total_tok += u.get("total_tokens", 0)
                            total_cost += u.get("cost", 0.0) or 0.0
            if total_tok > 0:
                return {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": total_tok,
                    "total_cost": float(total_cost),
                }
        except Exception:
            pass

    return usage


def load_all_results(
    results_dir: Path,
    study_filter: Optional[str] = None,
    config_filter: Optional[str] = None,
) -> Dict[str, Dict[str, dict]]:
    """Load results: {config_name: {study_id: study_result}}."""
    if "benchmark" in str(results_dir):
        search_base = results_dir
    elif (results_dir / "benchmark").exists():
        search_base = results_dir / "benchmark"
    else:
        search_base = results_dir

    all_results = defaultdict(dict)
    for study_dir in sorted(search_base.iterdir()):
        if not study_dir.is_dir() or not study_dir.name.startswith("study_"):
            continue
        study_id = study_dir.name
        if study_filter and study_id != study_filter:
            continue

        for config_dir in sorted(study_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name
            if config_filter and config_name != config_filter:
                continue

            benchmark_file = config_dir / "full_benchmark.json"
            eval_file = config_dir / "evaluation_results.json"
            raw_file = config_dir / "raw_responses.json"

            if not benchmark_file.exists():
                continue

            try:
                with open(benchmark_file, "r", encoding="utf-8") as fp:
                    benchmark_data = json.load(fp)
                eval_data = {}
                if eval_file.exists():
                    try:
                        with open(eval_file, "r", encoding="utf-8") as fp:
                            eval_data = json.load(fp)
                    except Exception:
                        pass

                usage = calculate_usage(benchmark_data, raw_file)
                study_result = {
                    "study_id": study_id,
                    "title": benchmark_data.get("title", "N/A"),
                    "usage_stats": usage,
                    "pas_result": eval_data,
                    "_mtime": benchmark_file.stat().st_mtime,
                }
                if study_id not in all_results[config_name] or study_result["_mtime"] > all_results[config_name][study_id].get("_mtime", 0):
                    all_results[config_name][study_id] = study_result
            except Exception as e:
                print(f"Warning: Could not load {benchmark_file}: {e}", file=sys.stderr)

    return dict(all_results)


def compute_benchmark_ecs(results: Dict[str, dict]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Compute ECS (CCC) at benchmark level and per-study. Uses composite finding_id to avoid collisions."""
    from src.evaluation.stats_lib import compute_ecs_corr

    all_tests = []
    for sid, s in sorted(results.items()):
        pr = s.get("pas_result", {})
        for t in pr.get("test_results", []):
            t_copy = dict(t)
            t_copy["study_id"] = sid
            sub = t_copy.get("sub_study_id", "unknown")
            fid = t_copy.get("finding_id", "unknown")
            t_copy["finding_id"] = f"{sub}:{fid}"
            all_tests.append(t_copy)

    if not all_tests:
        return None, {}

    ecs_result = compute_ecs_corr(all_tests, study_groups=STUDY_GROUPS)
    overall = ecs_result.get("ecs_overall")
    per_study = ecs_result.get("ecs_per_study", {})
    return overall, per_study


def get_ecs_study(study_result: dict, ecs_per_study: Dict[str, Optional[float]]) -> Optional[float]:
    """Get ECS for a single study from ecs_per_study or pas_result fallback."""
    sid = study_result.get("study_id")
    if sid and sid in ecs_per_study and ecs_per_study[sid] is not None:
        return float(ecs_per_study[sid])
    pr = study_result.get("pas_result", {})
    ecs = pr.get("ecs_corr") or pr.get("ecs_corr_study")
    if ecs is not None:
        if isinstance(ecs, dict):
            ecs = ecs.get("ecs_overall")
        if ecs is not None and isinstance(ecs, (int, float)):
            return float(ecs)
    details = pr.get("ecs_corr_details", {})
    ps = details.get("ecs_per_study", {})
    if sid and sid in ps and ps[sid] is not None:
        return float(ps[sid])
    return None


def ordered_findings_with_idx(pas_result: dict) -> List[dict]:
    """Return finding_results sorted deterministically, each with finding_idx 0..N-1."""
    frs = pas_result.get("finding_results", [])
    sorted_frs = sorted(frs, key=_sort_key_finding)
    out = []
    for idx, fr in enumerate(sorted_frs):
        row = dict(fr)
        row["finding_idx"] = idx
        out.append(row)
    return out


def build_simple_data(results: Dict[str, Dict[str, dict]]) -> Dict[str, Any]:
    """Build in-memory simple summary for one config."""
    pas_scores = []
    total_tokens = 0
    total_cost = 0.0
    studies_data = []
    findings_data = []

    ecs_overall, ecs_per_study = compute_benchmark_ecs(results)

    for sid in sorted(results.keys()):
        s = results[sid]
        pr = s.get("pas_result", {})
        usage = s.get("usage_stats", {}) or pr.get("usage_stats", {})
        tok = usage.get("total_tokens", 0)
        cost = usage.get("total_cost", 0.0)
        total_tokens += tok
        total_cost += cost

        pas = pr.get("score")
        if pas is not None:
            pas_scores.append(float(pas))

        ecs_s = get_ecs_study(s, ecs_per_study)

        studies_data.append({
            "study_id": sid,
            "title": s.get("title", "N/A"),
            "pas_raw": float(pas) if pas is not None else None,
            "ecs": float(ecs_s) if ecs_s is not None else None,
            "total_tokens": tok,
            "total_cost": float(cost),
        })

        for fr in ordered_findings_with_idx(pr):
            findings_data.append({
                "study_id": sid,
                "finding_idx": fr["finding_idx"],
                "sub_study_id": fr.get("sub_study_id", ""),
                "finding_id": fr.get("finding_id", ""),
                "finding_score": float(fr.get("finding_score", 0)),
                "n_tests": fr.get("n_tests", 0),
            })

    pas_benchmark = float(np.mean(pas_scores)) if pas_scores else None

    return {
        "pas_raw": pas_benchmark,
        "ecs": ecs_overall,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "studies": studies_data,
        "findings": findings_data,
    }


def generate_md_single(config_name: str, data: dict) -> str:
    """Generate markdown for one config."""
    lines = [
        f"## {config_name}",
        "",
        "### Benchmark Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| PAS_raw | {data['pas_raw']:.4f} |" if data["pas_raw"] is not None else "| PAS_raw | N/A |",
        f"| ECS (CCC) | {data['ecs']:.4f} |" if data["ecs"] is not None else "| ECS (CCC) | N/A |",
        f"| Total Tokens | {data['total_tokens']:,} |",
        f"| Total Cost | ${data['total_cost']:.4f} |",
        "",
        "### Per-Study",
        "",
        "| Study | Title | PAS_raw | ECS | Tokens | Cost |",
        "|-------|-------|---------|-----|--------|------|",
    ]
    for row in data["studies"]:
        pas_s = f"{row['pas_raw']:.4f}" if row["pas_raw"] is not None else "N/A"
        ecs_s = f"{row['ecs']:.4f}" if row["ecs"] is not None else "N/A"
        title = (row["title"][:40] or "N/A").replace("|", " ")
        lines.append(f"| {row['study_id']} | {title} | {pas_s} | {ecs_s} | {row['total_tokens']:,} | ${row['total_cost']:.4f} |")

    lines.extend(["", "### Per-Finding (index 0..N-1)", ""])
    current_study = None
    for row in data["findings"]:
        if row["study_id"] != current_study:
            current_study = row["study_id"]
            lines.append(f"\n**{current_study}**\n")
            lines.append("| idx | sub_study | finding_id | PAS_raw | n_tests |")
            lines.append("|-----|-----------|------------|---------|---------|")
        lines.append(f"| {row['finding_idx']} | {row['sub_study_id']} | {row['finding_id']} | {row['finding_score']:.4f} | {row['n_tests']} |")

    return "\n".join(lines)


def generate_md(all_configs_data: Dict[str, dict]) -> str:
    """Generate combined markdown for all configs."""
    lines = [
        "# HS-Bench Simple Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for config in sorted(all_configs_data.keys()):
        lines.append(generate_md_single(config, all_configs_data[config]))
        lines.append("")
    return "\n".join(lines)


def generate_studies_csv(all_configs_data: Dict[str, dict]) -> str:
    """CSV: config, study_id, title, pas_raw, ecs, total_tokens, total_cost."""
    lines = ["config,study_id,title,pas_raw,ecs,total_tokens,total_cost"]
    for config, data in sorted(all_configs_data.items()):
        for row in data["studies"]:
            pas = f"{row['pas_raw']:.4f}" if row["pas_raw"] is not None else ""
            ecs = f"{row['ecs']:.4f}" if row["ecs"] is not None else ""
            title = (row["title"] or "").replace(",", ";")
            lines.append(f"{config},{row['study_id']},{title},{pas},{ecs},{row['total_tokens']},{row['total_cost']:.4f}")
    return "\n".join(lines)


def generate_findings_csv(all_configs_data: Dict[str, dict]) -> str:
    """CSV: config, study_id, finding_idx, sub_study_id, finding_id, finding_score, n_tests."""
    lines = ["config,study_id,finding_idx,sub_study_id,finding_id,finding_score,n_tests"]
    for config, data in sorted(all_configs_data.items()):
        for row in data["findings"]:
            lines.append(f"{config},{row['study_id']},{row['finding_idx']},{row['sub_study_id']},{row['finding_id']},{row['finding_score']:.6f},{row['n_tests']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate simple HS-Bench results table")
    parser.add_argument("--results-dir", type=str, default="results/benchmark", help="Results directory (default: results/benchmark)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for files (default: same as results-dir)")
    parser.add_argument("--format", choices=["md", "csv", "json", "all"], default="all", help="Output format")
    parser.add_argument("--study", type=str, help="Filter to single study_id")
    parser.add_argument("--config", type=str, help="Filter to single config folder name")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found", file=sys.stderr)
        return 1

    raw_results = load_all_results(results_dir, study_filter=args.study, config_filter=args.config)
    if not raw_results:
        print("No results found.", file=sys.stderr)
        return 1

    all_configs_data = {}
    for config_name, studies in raw_results.items():
        all_configs_data[config_name] = build_simple_data(studies)

    # Default: write files into results-dir
    out_dir = results_dir
    out_prefix = "simple_results"
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix and out_path.suffix != ".json":
            out_dir = out_path.parent
            out_prefix = out_path.stem
        else:
            out_dir = out_path if (out_path.is_dir() or not out_path.exists()) else out_path.parent
        if out_dir and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

    def write_md():
        text = generate_md(all_configs_data)
        if out_dir:
            p = out_dir / "simple_summary.md"
            p.write_text(text, encoding="utf-8")
            print(f"Wrote {p}")
        else:
            print(text)

    def write_csv():
        studies_csv = generate_studies_csv(all_configs_data)
        findings_csv = generate_findings_csv(all_configs_data)
        if out_dir:
            (out_dir / "simple_studies.csv").write_text(studies_csv, encoding="utf-8")
            (out_dir / "simple_findings.csv").write_text(findings_csv, encoding="utf-8")
            print(f"Wrote {out_dir / 'simple_studies.csv'}")
            print(f"Wrote {out_dir / 'simple_findings.csv'}")
        else:
            print("--- simple_studies.csv ---")
            print(studies_csv)
            print("\n--- simple_findings.csv ---")
            print(findings_csv)

    def write_json():
        out = {
            "metadata": {"generated_at": datetime.now().isoformat(), "format": "simple_results"},
            "configs": all_configs_data,
        }
        txt = json.dumps(out, indent=2, ensure_ascii=False)
        if out_dir:
            p = out_dir / f"{out_prefix}.json"
            p.write_text(txt, encoding="utf-8")
            print(f"Wrote {p}")
        else:
            print(txt)

    if args.format == "md":
        write_md()
    elif args.format == "csv":
        write_csv()
    elif args.format == "json":
        write_json()
    else:
        write_md()
        write_csv()
        write_json()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
