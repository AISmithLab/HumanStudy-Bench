#!/usr/bin/env python3
"""
LEGACY: HumanStudy-Bench Results Table Generator (PAS + ECS)

This script is kept for backward compatibility. It generates tables that include
PAS (Probability Alignment Score) and ECS (Effect Consistency Score) metrics.

For the current pipeline, use generate_results_table.py instead, which outputs
ECS_corr (CCC) only. See README "Legacy metrics" for details.
"""

import json
import argparse
import sys
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_usage_from_individual_data(benchmark_data: dict) -> dict:
    """
    Calculate usage stats from individual_data if usage_stats is missing or zero.
    This handles cases where usage is stored in individual responses.
    
    Handles both flat and nested structures, and also handles cases where
    usage might be stored at the participant level (study_009) or response level.
    """
    usage_stats = benchmark_data.get('usage_stats', {})
    
    # If usage_stats already has valid data, use it
    if usage_stats.get('total_tokens', 0) > 0:
        return usage_stats
    
    # Otherwise, calculate from individual_data
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    participant_count = 0
    
    individual_data = benchmark_data.get('individual_data', [])
    if not individual_data:
        return usage_stats
    
    # Check if data is flat or nested
    is_flat = len(individual_data) > 0 and 'responses' not in individual_data[0]
    
    if is_flat:
        # Flat structure: each item is a response
        seen_participants = set()
        for resp in individual_data:
            pid = resp.get('participant_id')
            if pid not in seen_participants:
                seen_participants.add(pid)
                participant_count += 1
            
            # Try to get usage from response
            usage = resp.get('usage', {})
            if not usage or usage.get('total_tokens', 0) == 0:
                # For study_009, usage might be stored differently
                # Check if there's a usage field at the participant level
                continue
            
            total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
            total_completion_tokens += usage.get('completion_tokens', 0) or 0
            total_tokens += usage.get('total_tokens', 0) or 0
            total_cost += usage.get('cost', 0.0) or 0.0
    else:
        # Nested structure: participant has responses list
        participant_count = len(individual_data)
        for participant in individual_data:
            # Check if usage is stored at participant level (study_009 style)
            participant_usage = participant.get('usage', {})
            if participant_usage and participant_usage.get('total_tokens', 0) > 0:
                total_prompt_tokens += participant_usage.get('prompt_tokens', 0) or 0
                total_completion_tokens += participant_usage.get('completion_tokens', 0) or 0
                total_tokens += participant_usage.get('total_tokens', 0) or 0
                total_cost += participant_usage.get('cost', 0.0) or 0.0
            else:
                # Check each response for usage (study_012 style)
                for resp in participant.get('responses', []):
                    usage = resp.get('usage', {})
                    if usage:
                        total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                        total_completion_tokens += usage.get('completion_tokens', 0) or 0
                        total_tokens += usage.get('total_tokens', 0) or 0
                        total_cost += usage.get('cost', 0.0) or 0.0
    
    if participant_count > 0 and total_tokens > 0:
        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": float(total_cost),
            "avg_tokens_per_participant": float(total_tokens / participant_count),
            "avg_cost_per_participant": float(total_cost / participant_count)
        }
    
    # If no usage found, return original (might be 0 for cached/simulated runs)
    return usage_stats


def load_all_results(results_dir: Path) -> dict:
    """
    Load all benchmark results and organize by model.
    
    Supports two directory structures:
    1. New structure: results/{benchmark_folder}/{study_id}/{config}/full_benchmark.json
    2. Legacy structure: results/full_benchmark_*.json
    
    Returns: {model_config: {study_id: study_result}}
    where model_config is "{model}_{prompt_preset}"
    """
    all_results = defaultdict(dict)
    
    # Method 1: Scan new directory structure (results/benchmark/{study_id}/{config}/)
    benchmark_folder = results_dir.name if results_dir.name != "results" else None
    if benchmark_folder and (results_dir / benchmark_folder).exists():
        search_base = results_dir / benchmark_folder
    elif "benchmark" in str(results_dir):
        search_base = results_dir
    else:
        # Look for benchmark folder inside results_dir
        benchmark_subdirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in ["benchmark", "benchmark_baseline", "runs"]]
        if benchmark_subdirs:
            search_base = benchmark_subdirs[0]  # Use first found
        else:
            search_base = results_dir
    
    # Scan study directories: {search_base}/{study_id}/{config_folder}/
    for study_dir in search_base.iterdir():
        if not study_dir.is_dir():
            continue
        
        study_id = study_dir.name
        if not study_id.startswith("study_"):
            continue
        
        # Look for config folders (e.g., "mistralai_mistral_nemo_v2-human")
        for config_dir in study_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            config_name = config_dir.name
            
            # Load full_benchmark.json
            benchmark_file = config_dir / "full_benchmark.json"
            eval_file = config_dir / "evaluation_results.json"
            
            if not benchmark_file.exists():
                continue
            
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as fp:
                    benchmark_data = json.load(fp)
                
                # Extract model and prompt info
                model = benchmark_data.get('model', 'unknown')
                prompt_preset = benchmark_data.get('system_prompt_preset', 'unknown')
                # Use config_name as unique identifier (includes both model and prompt)
                model_config = config_name
                
                # Load evaluation results if available
                eval_data = {}
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r', encoding='utf-8') as fp:
                            eval_data = json.load(fp)
                    except Exception as e:
                        print(f"Warning: Could not load {eval_file}: {e}")
                
                # Calculate usage stats (from usage_stats or individual_data)
                usage_stats = calculate_usage_from_individual_data(benchmark_data)
                
                # Combine data
                study_result = {
                    'study_id': study_id,
                    'title': benchmark_data.get('title', 'N/A'),
                    'model': model,
                    'system_prompt_preset': prompt_preset,
                    'n_participants': len(benchmark_data.get('individual_data', [])),
                    'usage_stats': usage_stats,
                    'pas_result': eval_data,  # evaluation_results.json becomes pas_result
                    'normalized_score': eval_data.get('normalized_score'),
                    'score': eval_data.get('score'),
                    '_mtime': benchmark_file.stat().st_mtime,
                    '_config': config_name,
                    '_elapsed_time': benchmark_data.get('elapsed_time', 0),  # Store for missing usage detection
                    '_benchmark_data': benchmark_data  # Store for access in table generation
                }
                
                # Keep latest if duplicate (same study_id + model_config)
                if study_id not in all_results[model_config] or study_result['_mtime'] > all_results[model_config][study_id].get('_mtime', 0):
                    all_results[model_config][study_id] = study_result
                    
            except Exception as e:
                print(f"Warning: Could not load {benchmark_file}: {e}")
    
    # Method 2: Legacy support - scan for full_benchmark_*.json in root
    for f in results_dir.glob("full_benchmark_*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                model = data.get('model', 'unknown')
                prompt_preset = data.get('system_prompt_preset', 'unknown')
                model_config = f"{model}_{prompt_preset}"
                mtime = f.stat().st_mtime
                
                for study in data.get('studies', []):
                    sid = study['study_id']
                    key = f"{model_config}_legacy"
                    if sid not in all_results[key] or mtime > all_results[key][sid].get('_mtime', 0):
                        all_results[key][sid] = {**study, '_mtime': mtime}
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return all_results


def generate_summary_table(results: dict, output_format: str = 'markdown', se_method: str = 'auto', verbose: bool = False, method_stats: dict = None,
                           compute_bootstrap: bool = False, bootstrap_iterations: int = 200, results_dir: Path = None, n_jobs: int = -1, allow_write: bool = False,
                           force_recompute_bootstrap: bool = False) -> str:
    """Generate summary table with all studies per model."""
    
    lines = []
    if method_stats is None:
        method_stats = {'total_studies': 0, 'has_repeat': 0, 'has_bootstrap': 0, 'missing_requested_method': []}
    
    for model, studies in sorted(results.items()):
        lines.append(f"\n{'='*100}")
        lines.append(f"Model: {model}")
        lines.append(f"{'='*100}\n")
        
        # Header
        if output_format == 'markdown':
            lines.append("| Study | Title | PAS (Norm) | ECS (r) | Raw Fail % | Final Fail % | Total Input Tokens | Total Output Tokens | Total Tokens | Total Cost | Status |")
            lines.append("|-------|-------|------------|---------|------------|--------------|-------------------|---------------------|--------------|------------|--------|")
        else:
            lines.append(f"{'Study':<12} | {'Title':<30} | {'PAS (Norm)':<12} | {'ECS (r)':<10} | {'Raw Fail%':<10} | {'Final Fail%':<12} | {'Input Tok':<15} | {'Output Tok':<15} | {'Total Tok':<12} | {'Total Cost':<12} | {'Status':<8}")
            lines.append("-" * 200)
        
        total_score = 0
        total_total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_all_tokens = 0
        count = 0
        positive_count = 0
        
        for sid in sorted(studies.keys()):
            s = studies[sid]
            
            # Get PAS score from various possible locations
            pas = s.get('normalized_bas') or s.get('normalized_score')
            pas_result = s.get('pas_result', {})
            if pas is None:
                pas = pas_result.get('normalized_score')
                if pas is None:
                    # Try to compute from score if available
                    raw_score = pas_result.get('score', 0)
                    if raw_score is not None:
                        # normalized_score = 2 * score - 1 (if score is in [0,1])
                        # But score might already be normalized, check range
                        if 0 <= raw_score <= 1:
                            pas = 2 * raw_score - 1
                        else:
                            pas = raw_score  # Already normalized
            
            if pas is None:
                pas = 0.0
            
            # Calculate average replication consistency from test results
            test_results = pas_result.get('test_results', [])
            replication_consistency = None
            if test_results:
                consistency_scores = [t.get('replication_consistency') for t in test_results if t.get('replication_consistency') is not None]
                if consistency_scores:
                    replication_consistency = np.mean(consistency_scores)
            
            # Get SE values for display (if requested)
            method_stats['total_studies'] += 1
            
            # Track method availability
            if pas_result.get('score_repeat_se') is not None:
                method_stats['has_repeat'] += 1
            if pas_result.get('bootstrap_iterations', 0) > 0:
                method_stats['has_bootstrap'] += 1
            
            # Build config_path if needed for bootstrap computation
            config_path = None
            if compute_bootstrap and results_dir:
                config_path = results_dir / "benchmark" / sid / s.get('_config', 'unknown')
            
            score_se, normalized_score_se, method_used = get_se_values(
                pas_result, se_method, verbose=verbose,
                compute_bootstrap=compute_bootstrap, bootstrap_iterations=bootstrap_iterations,
                study_id=sid, config_path=config_path, n_jobs=n_jobs,
                allow_write=allow_write, force_recompute_bootstrap=force_recompute_bootstrap
            )
            
            # Track missing requested method
            if (se_method == 'bootstrap' and method_used == 'none') or \
               (se_method == 'repeat' and method_used == 'none'):
                method_stats['missing_requested_method'].append(sid)
            
            # Get failure rates (from pas_result if available)
            failure_rates = pas_result.get('failure_rates', {})
            if not failure_rates:
                failure_rates = s.get('failure_rates', {})
            raw_fail_rate = failure_rates.get('raw_failure_rate', None)
            final_fail_rate = failure_rates.get('final_failure_rate', None)
                
            # Get usage stats
            usage = s.get('usage_stats', {})
            if not usage:
                usage = pas_result.get('usage_stats', {})
            
            tot_input_tokens = usage.get('total_prompt_tokens', 0)
            tot_output_tokens = usage.get('total_completion_tokens', 0)
            tot_all_tokens = usage.get('total_tokens', 0)
            tot_cost = usage.get('total_cost', 0.0)
            
            # Check if this is likely missing data (elapsed_time > 0 but tokens = 0)
            # This indicates the experiment ran but usage wasn't saved
            elapsed_time = s.get('_elapsed_time', 0)
            is_missing_usage = (elapsed_time > 10 and tot_all_tokens == 0)  # Likely missing if ran > 10s but no tokens
            
            # Accumulate totals (only if we have valid data)
            if not is_missing_usage:
                total_total_cost += tot_cost
                total_input_tokens += tot_input_tokens
                total_output_tokens += tot_output_tokens
                total_all_tokens += tot_all_tokens

            status = "✅" if pas > 0 else "❌"
            title = s.get('title', 'N/A')[:30]
            
            # Format failure rates
            raw_fail_display = f"{raw_fail_rate:.2f}%" if raw_fail_rate is not None else "N/A"
            final_fail_display = f"{final_fail_rate:.2f}%" if final_fail_rate is not None else "N/A"
            
            # Show warning for missing usage data
            if is_missing_usage:
                cost_display = "N/A*"
                tokens_display = "N/A*"
            else:
                cost_display = f"${tot_cost:.4f}"
                tokens_display = f"{tot_all_tokens:,}"
            
            # Format replication consistency
            rep_cons_display = f"{replication_consistency:.4f}" if replication_consistency is not None else "N/A"
            
            if output_format == 'markdown':
                lines.append(f"| {sid} | {title} | {pas:+.4f} | {rep_cons_display} | {raw_fail_display} | {final_fail_display} | {tot_input_tokens:,} | {tot_output_tokens:,} | {tokens_display} | {cost_display} | {status} |")
            else:
                lines.append(f"{sid:<12} | {title:<30} | {pas:>+10.4f}   | {rep_cons_display:>11}   | {raw_fail_display:>9}   | {final_fail_display:>11}   | {tot_input_tokens:>13,}   | {tot_output_tokens:>13,}   | {tokens_display:>10}   | {cost_display:>10} | {status:<8}")
            
            total_score += pas
            count += 1
            if pas > 0:
                positive_count += 1
        
        # Summary row
        avg_score = total_score / count if count > 0 else 0
        lines.append("")
        if output_format == 'markdown':
            lines.append("| **TOTAL** | **All Studies** | **" + f"{avg_score:+.4f}" + "** | **-** | **-** | **" + f"{total_input_tokens:,}" + "** | **" + f"{total_output_tokens:,}" + "** | **" + f"{total_all_tokens:,}" + "** | **$" + f"{total_total_cost:.4f}" + "** | **" + f"{positive_count}/{count}" + "** |")
        else:
            lines.append(f"{'TOTAL':<12} | {'All Studies':<30} | {avg_score:>+10.4f}   | {'-':>9}   | {'-':>11}   | {total_input_tokens:>13,}   | {total_output_tokens:>13,}   | {total_all_tokens:>10,}   | ${total_total_cost:>10.4f} | {positive_count}/{count}")
        lines.append("")
        lines.append(f"**Summary**: {count} studies | {positive_count} positive | Average PAS: {avg_score:+.4f} | Total Cost: ${total_total_cost:.4f} | Total Tokens: {total_all_tokens:,} ({total_input_tokens:,} input, {total_output_tokens:,} output)")
        
    return "\n".join(lines)


def generate_detailed_table(results: dict, se_method: str = 'auto', verbose: bool = False, method_stats: dict = None,
                            compute_bootstrap: bool = False, bootstrap_iterations: int = 200, results_dir: Path = None, n_jobs: int = -1, allow_write: bool = False,
                            force_recompute_bootstrap: bool = False) -> str:
    """Generate detailed table with per-test breakdown."""
    
    lines = []
    if method_stats is None:
        method_stats = {'total_studies': 0, 'has_repeat': 0, 'has_bootstrap': 0, 'missing_requested_method': []}
    
    for model, studies in sorted(results.items()):
        lines.append(f"\n{'#'*100}")
        lines.append(f"# DETAILED RESULTS: {model}")
        lines.append(f"{'#'*100}\n")
        
        for sid in sorted(studies.keys()):
            s = studies[sid]
            title = s.get('title', 'N/A')
            
            # Get PAS
            pas = s.get('normalized_bas') or s.get('normalized_score')
            if pas is None:
                pas_result = s.get('pas_result', {})
                pas = pas_result.get('normalized_score', 0)
                if pas == 0:
                    # Try to compute from raw score
                    raw_score = pas_result.get('score', 0)
                    if 0 <= raw_score <= 1:
                        pas = 2 * raw_score - 1
                    else:
                        pas = raw_score
            
            # Get SE values for display
            pas_result = s.get('pas_result', {})
            method_stats['total_studies'] += 1
            
            # Track method availability
            if pas_result.get('score_repeat_se') is not None:
                method_stats['has_repeat'] += 1
            if pas_result.get('bootstrap_iterations', 0) > 0:
                method_stats['has_bootstrap'] += 1
            
            if verbose:
                print(f"\nProcessing {sid} ({s.get('title', 'N/A')[:50]}):")
            
            # Build config_path if needed for bootstrap computation
            config_path = None
            if compute_bootstrap and results_dir:
                config_path = results_dir / "benchmark" / sid / s.get('_config', 'unknown')
            
            score_se, normalized_score_se, method_used = get_se_values(
                pas_result, se_method, verbose=verbose,
                compute_bootstrap=compute_bootstrap, bootstrap_iterations=bootstrap_iterations,
                study_id=sid, config_path=config_path, n_jobs=n_jobs,
                allow_write=allow_write, force_recompute_bootstrap=force_recompute_bootstrap
            )
            
            # Track missing requested method
            if (se_method == 'bootstrap' and method_used == 'none') or \
               (se_method == 'repeat' and method_used == 'none'):
                method_stats['missing_requested_method'].append(sid)
            
            lines.append(f"\n## {sid}: {title}")
            if normalized_score_se > 0:
                lines.append(f"**Overall PAS (Normalized)**: {pas:+.4f} ± {normalized_score_se:.4f} (SE method: {method_used})")
            else:
                lines.append(f"**Overall PAS (Normalized)**: {pas:+.4f}")
            lines.append(f"**Participants**: {s.get('n_participants', 'N/A')}")
            
            # Get usage stats
            usage = s.get('usage_stats', {})
            if not usage:
                usage = s.get('pas_result', {}).get('usage_stats', {})
            
            if usage:
                lines.append(f"**Total Tokens**: {usage.get('total_tokens', 0)} ({usage.get('total_prompt_tokens', 0)} prompt, {usage.get('total_completion_tokens', 0)} completion)")
                lines.append(f"**Total Cost**: ${usage.get('total_cost', 0.0):.4f}")
                lines.append(f"**Avg per Participant**: {usage.get('avg_tokens_per_participant', 0):.1f} tokens, ${usage.get('avg_cost_per_participant', 0.0):.6f}")
            
            # Get finding results and test results (pas_result already retrieved above)
            finding_results = pas_result.get('finding_results', [])
            test_results = pas_result.get('test_results', [])
            
            # Show finding-level PAS scores
            if finding_results:
                lines.append("\n### Finding-Level PAS Scores:")
                lines.append(f"| Finding ID | Finding Score | Finding Weight | N Tests |")
                lines.append("|------------|---------------|----------------|---------|")
                
                for finding in finding_results:
                    fid = finding.get('finding_id', 'N/A')
                    fscore = finding.get('finding_score', 0.5)
                    fweight = finding.get('finding_weight', 1.0)
                    n_tests = finding.get('n_tests', 0)
                    lines.append(f"| {fid} | {fscore:.4f} | {fweight:.2f} | {n_tests} |")
            
            # Show test-level PAS scores
            if test_results:
                lines.append("\n### Test-Level PAS Scores:")
                lines.append(f"| Test Name | Finding | Pi Human | Pi Agent | PAS | Test Weight |")
                lines.append("|-----------|--------|----------|----------|-----|-------------|")
                
                for test in test_results:
                    name = test.get('test_name', 'Unknown')[:40]
                    finding_id = test.get('finding_id', 'N/A')
                    pi_h = test.get('pi_human', 0.5)
                    pi_a = test.get('pi_agent', 0.5)
                    score = test.get('pas') or test.get('score', 0.5)
                    test_weight = test.get('test_weight', 1.0)
                    lines.append(f"| {name} | {finding_id} | {pi_h:.4f} | {pi_a:.4f} | {score:.4f} | {test_weight:.2f} |")
            else:
                details = pas_result.get('details', 'No details available')
                lines.append(f"\n*Details*: {details}")
            
            lines.append("")
    
    return "\n".join(lines)


def generate_csv(results: dict) -> str:
    """Generate CSV format for easy import to spreadsheets."""
    
    lines = ["model,study_id,title,bas_normalized,rep_cons,status,n_participants,test_count,finding_count,total_input_tokens,total_output_tokens,total_tokens,total_cost,avg_tokens,avg_cost"]
    
    for model, studies in sorted(results.items()):
        # Pre-compute study-level ECS_Strict using RMS-Stouffer method
        from src.evaluation.stats_lib import aggregate_finding_ecs, aggregate_study_ecs_strict
        from collections import defaultdict
        
        study_level_ecs_strict = {}
        for sid in sorted(studies.keys()):
            s = studies[sid]
            pas_result = s.get('pas_result', {})
            test_results = pas_result.get('test_results', [])
            
            if not test_results:
                study_level_ecs_strict[sid] = None
                continue
            
            # Group tests by finding_id
            finding_tests = defaultdict(list)
            for t in test_results:
                finding_id = t.get('finding_id', 'default')
                finding_tests[finding_id].append(t)
            
            # Calculate finding-level p-values using RMS (Chi-squared)
            finding_p_values = []
            for finding_id, tests in finding_tests.items():
                finding_p = aggregate_finding_ecs(tests)
                if finding_p is not None and not (np.isnan(finding_p) or np.isinf(finding_p)):
                    finding_p_values.append(finding_p)
            
            # Aggregate to study-level using Stouffer method
            if finding_p_values:
                study_p = aggregate_study_ecs_strict(finding_p_values)
                study_level_ecs_strict[sid] = study_p
            else:
                study_level_ecs_strict[sid] = None
        
        for sid in sorted(studies.keys()):
            s = studies[sid]
            
            pas = s.get('normalized_bas') or s.get('normalized_score')
            if pas is None:
                pas_result = s.get('pas_result', {})
                pas = pas_result.get('normalized_score', 0)
                if pas == 0:
                    raw_score = pas_result.get('score', 0)
                    if raw_score and 0 <= raw_score <= 1:
                        pas = 2 * raw_score - 1
                    else:
                        pas = raw_score or 0
            
            # Get ECS_corr (main metric: correlation-based)
            # First try to get study-level ECS_corr
            pas_result = s.get('pas_result', {})
            ecs_corr = pas_result.get('ecs_corr_study')
            
            # Fallback to overall ECS_corr if available
            if ecs_corr is None:
                ecs_corr_data = pas_result.get('ecs_corr', {})
                if ecs_corr_data:
                    ecs_corr = ecs_corr_data.get('ecs_corr_overall')
            
            # Legacy fallback: use strict metric (appendix)
            if ecs_corr is None:
                rep_cons = s.get('replication_consistency')
                if rep_cons is None:
                    rep_cons = pas_result.get('average_consistency_score') or pas_result.get('ecs_strict_study')
                    if rep_cons is None:
                        test_results = pas_result.get('test_results', [])
                        consistency_scores = [t.get('consistency_score') for t in test_results if t.get('consistency_score') is not None]
                        if consistency_scores:
                            rep_cons = np.mean(consistency_scores)
                ecs_corr = rep_cons if rep_cons is not None else 0.0
            
            if ecs_corr is None:
                ecs_corr = 0.0
            
            # Get usage stats
            usage = s.get('usage_stats', {})
            if not usage:
                usage = s.get('pas_result', {}).get('usage_stats', {})
            
            total_input_tokens = usage.get('total_prompt_tokens', 0)
            total_output_tokens = usage.get('total_completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            total_cost = usage.get('total_cost', 0.0)
            avg_tokens = usage.get('avg_tokens_per_participant', 0)
            avg_cost = usage.get('avg_cost_per_participant', 0.0)

            title = s.get('title', 'N/A').replace(',', ';')[:50]
            status = "pass" if pas > 0 else "fail"
            n_participants = s.get('n_participants', 0)
            pas_result = s.get('pas_result', {})
            test_count = len(pas_result.get('test_results', []))
            finding_count = len(pas_result.get('finding_results', []))
            
            lines.append(f"{model},{sid},{title},{pas:.4f},{ecs_corr:.4f},{status},{n_participants},{test_count},{finding_count},{total_input_tokens},{total_output_tokens},{total_tokens},{total_cost:.4f},{avg_tokens:.1f},{avg_cost:.6f}")
    
    return "\n".join(lines)


def generate_json_summary(results: dict, se_method: str = 'auto', verbose: bool = False,
                          compute_bootstrap: bool = False, bootstrap_iterations: int = 200, results_dir: Path = None, n_jobs: int = -1, allow_write: bool = False,
                          force_recompute_bootstrap: bool = False) -> dict:
    """
    Generate standardized JSON format for results storage and analysis.
    
    Returns a structured JSON object with:
    - Metadata (generation time, version)
    - Summary by model (with totals)
    - Detailed study-level data
    - Finding-level and test-level scores
    
    Args:
        results: Dictionary of {model_config: {study_id: study_result}}
        
    Returns:
        dict: Standardized JSON structure
    """
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "format": "humanstudy_bench_summary",
            "se_method": se_method
        },
        "models": {}
    }
    
    for model, studies in sorted(results.items()):
        # ECS_Strict: RMS-Stouffer over the entire benchmark (all findings, all studies)
        from src.evaluation.stats_lib import aggregate_finding_ecs, aggregate_study_ecs_strict, aggregate_study_pas_mean_only
        from collections import defaultdict
        
        # Study-level ECS_Strict (for per-study display)
        study_level_ecs_strict = {}
        # Pool all tests across studies, group by (study_id, finding_id) for global RMS-Stouffer
        all_finding_tests = defaultdict(list)
        for sid in sorted(studies.keys()):
            s = studies[sid]
            pas_result = s.get('pas_result', {})
            test_results = pas_result.get('test_results', [])
            if not test_results:
                study_level_ecs_strict[sid] = None
                continue
            finding_tests = defaultdict(list)
            for t in test_results:
                fid = t.get('finding_id', 'default')
                finding_tests[fid].append(t)
                all_finding_tests[(sid, fid)].append(t)
            finding_p_values = []
            for fid, tests in finding_tests.items():
                finding_p = aggregate_finding_ecs(tests)
                if finding_p is not None and not (np.isnan(finding_p) or np.isinf(finding_p)):
                    finding_p_values.append(finding_p)
            if finding_p_values:
                study_level_ecs_strict[sid] = aggregate_study_ecs_strict(finding_p_values)
            else:
                study_level_ecs_strict[sid] = None
        
        # Global ECS_Strict = one RMS-Stouffer over entire benchmark (all findings)
        global_finding_p_values = []
        for (_sid, _fid), tests in all_finding_tests.items():
            p = aggregate_finding_ecs(tests)
            if p is not None and not (np.isnan(p) or np.isinf(p)):
                global_finding_p_values.append(p)
        ecs_strict_overall_value = (
            float(aggregate_study_ecs_strict(global_finding_p_values))
            if global_finding_p_values else 0.0
        )
        
        # Calculate totals for this model
        total_input_tokens = 0
        total_output_tokens = 0
        total_all_tokens = 0
        total_cost = 0.0
        total_score = 0.0
        count = 0
        positive_count = 0
        
        model_studies = []
        pas_mean_study_list = []  # for PAS (mean chain): mean at finding & study, only test special
        
        total_studies = len(studies)
        study_idx = 0
        
        for sid in sorted(studies.keys()):
            study_idx += 1
            s = studies[sid]
            
            if verbose:
                print(f"\n[{study_idx}/{total_studies}] Processing {sid}...")
            
            # Get PAS score (same logic as generate_summary_table)
            pas = s.get('normalized_bas') or s.get('normalized_score')
            if pas is None:
                pas_result = s.get('pas_result', {})
                pas = pas_result.get('normalized_score')
                if pas is None:
                    # Try to compute from score if available
                    raw_score = pas_result.get('score', 0)
                    if raw_score is not None:
                        # normalized_score = 2 * score - 1 (if score is in [0,1])
                        if 0 <= raw_score <= 1:
                            pas = 2 * raw_score - 1
                        else:
                            pas = raw_score  # Already normalized
            
            if pas is None:
                pas = 0.0
            
            # Get usage stats
            usage = s.get('usage_stats', {})
            if not usage:
                usage = s.get('pas_result', {}).get('usage_stats', {})
            
            tot_input_tokens = usage.get('total_prompt_tokens', 0)
            tot_output_tokens = usage.get('total_completion_tokens', 0)
            tot_all_tokens = usage.get('total_tokens', 0)
            tot_cost = usage.get('total_cost', 0.0)
            
            # Check if missing usage data
            elapsed_time = s.get('_elapsed_time', 0)
            is_missing_usage = (elapsed_time > 10 and tot_all_tokens == 0)
            
            # Accumulate totals (only if valid data)
            if not is_missing_usage:
                total_input_tokens += tot_input_tokens
                total_output_tokens += tot_output_tokens
                total_all_tokens += tot_all_tokens
                total_cost += tot_cost
            
            # Get finding and test results
            pas_result = s.get('pas_result', {})
            finding_results = pas_result.get('finding_results', [])
            test_results = pas_result.get('test_results', [])
            # PAS (mean chain): finding = mean(tests), study = mean(findings); only test level special
            if test_results:
                pas_mean_study_list.append(aggregate_study_pas_mean_only(test_results))
            
            # Get raw score for pas.raw
            raw_score = pas_result.get('score', 0.5)
            if raw_score is None:
                raw_score = 0.5
            
            # Calculate Statistical Reproducibility Score (SRS) metric
            # UPDATED: Two-level hierarchical aggregation (matching PAS structure)
            # This respects the hierarchical organization of findings in psychological studies
            # and improves discriminability by properly weighting findings.
            #
            # SRS calculation (per finding):
            # - Skip tests where human p > alpha (non-significant human findings)
            # - For tests where human p <= alpha (significant):
            #   * SRS = 1 if agent also significant (p <= alpha) AND direction matches
            #   * SRS = 0 otherwise
            # - Finding SRS = (passed_tests_weight) / (total_significant_human_tests_weight)
            #
            # Study SRS = weighted average of finding SRS scores (by finding_weight)
            #
            # This metric is similar to Average Pass Rate (APR) but uses weighted tests
            # and focuses on statistical replication of published findings.
            # This uses actual p-values from statistical tests, not PAS estimates
            # Two-level hierarchical aggregation (matching PAS structure)
            # Level 1: Aggregate tests into findings (weighted by test weights)
            # Level 2: Aggregate findings into study score (weighted by finding weights)
            sig_level_default = 0.05
            
            # Group tests by finding_id
            finding_test_groups = {}
            for t in test_results:
                finding_id = t.get('finding_id', 'unknown')
                if finding_id not in finding_test_groups:
                    finding_test_groups[finding_id] = []
                finding_test_groups[finding_id].append(t)
            
            # Level 1: Calculate SRS for each finding
            finding_srs_scores = {}
            for finding_id, tests in finding_test_groups.items():
                srs_pass_count = 0.0
                srs_total_weight = 0.0
                
                for t in tests:
                    weight = t.get('test_weight', 1.0)
                    
                    # Get p-values and significance flags
                    p_val_human = t.get('p_value_human')
                    p_val_agent = t.get('p_value_agent')
                    is_sig_human = t.get('is_significant_human', False)
                    is_sig_agent = t.get('is_significant_agent', False)
                    direction_match = t.get('direction_match', False)
                    sig_level = t.get('significance_level', sig_level_default)
                    
                    # Determine human significance:
                    # - If p_value_human is available, use it directly
                    # - If p_value_human is None (missing), default to significant (p < 0.05)
                    # - Only skip if we explicitly know p_value_human > sig_level
                    if p_val_human is not None:
                        # We have explicit p-value: use it to determine significance
                        is_sig_human = p_val_human < sig_level
                        # If p > alpha, skip this test
                        # if not is_sig_human:
                        #     continue  # Skip: human p > alpha (not significant)
                    else:
                        # Missing human p-value: default to significant (assume p < 0.05)
                        # Include the test in APR calculation
                        is_sig_human = True
                    
                    if p_val_agent is not None:
                        is_sig_agent = p_val_agent < sig_level
                    
                    # Human is significant - include this test in SRS calculation
                    srs_total_weight += weight
                    
                    # Calculate SRS: pass if agent also significant AND direction matches
                    if is_sig_agent and direction_match:
                        srs_pass_count += weight
                
                finding_srs = (srs_pass_count / srs_total_weight 
                              if srs_total_weight > 0 else 0.0)
                finding_srs_scores[finding_id] = finding_srs
            
            # Level 2: Aggregate findings into study score (weighted by finding weights)
            # Get finding weights from finding_results
            finding_weights_map = {}
            for fr in finding_results:
                finding_id = fr.get('finding_id', 'unknown')
                finding_weight = fr.get('finding_weight', 1.0)
                finding_weights_map[finding_id] = finding_weight
            
            # Calculate weighted average of finding SRS scores
            total_weighted_srs = 0.0
            total_finding_weight = 0.0
            
            for finding_id, finding_srs in finding_srs_scores.items():
                finding_weight = finding_weights_map.get(finding_id, 1.0)
                total_weighted_srs += finding_srs * finding_weight
                total_finding_weight += finding_weight
            
            srs_score = (total_weighted_srs / total_finding_weight 
                        if total_finding_weight > 0 else 0.0)
            srs_pass_rate = srs_score  # Same as score for binary metric
            
            # Get SE values based on selected method
            # Build config_path if needed for bootstrap computation
            config_path = None
            if compute_bootstrap and results_dir:
                config_path = results_dir / "benchmark" / sid / s.get('_config', 'unknown')
            
            score_se, normalized_score_se, method_used = get_se_values(
                pas_result, se_method, verbose=verbose,
                compute_bootstrap=compute_bootstrap, bootstrap_iterations=bootstrap_iterations,
                study_id=sid, config_path=config_path, n_jobs=n_jobs,
                allow_write=allow_write, force_recompute_bootstrap=force_recompute_bootstrap
            )
            
            study_data = {
                "study_id": sid,
                "title": s.get('title', 'N/A'),
                "pas": {
                    "normalized": float(pas),
                    "normalized_se": normalized_score_se,
                    "raw": float(raw_score),
                    "raw_se": score_se,
                    "status": "pass" if pas > 0 else "fail",
                    "se_method": method_used
                },
                "statistical_replication": {
                    "score": float(srs_score),
                    "pass_rate": float(srs_pass_rate),
                    "average_ecs": float(pas_result.get('ecs_corr_study') or pas_result.get('average_ecs') or pas_result.get('average_consistency_score', 0.0)),
                    "ecs_corr_study": pas_result.get('ecs_corr_study'),
                    "ecs_strict_study": study_level_ecs_strict.get(sid),  # RMS-Stouffer method
                    "replication_consistency": float(study_level_ecs_strict.get(sid) or pas_result.get('average_consistency_score', 0.0)),  # Legacy alias
                    "passed_tests": float(srs_pass_count),
                    "total_tests": float(srs_total_weight),
                    "status": "pass" if srs_pass_rate >= 0.5 else "fail"
                },
                "participants": {
                    "count": s.get('n_participants', 0)
                },
                "usage": {
                    "input_tokens": int(tot_input_tokens),
                    "output_tokens": int(tot_output_tokens),
                    "total_tokens": int(tot_all_tokens),
                    "total_cost": float(tot_cost),
                    "avg_tokens_per_participant": float(usage.get('avg_tokens_per_participant', 0.0)),
                    "avg_cost_per_participant": float(usage.get('avg_cost_per_participant', 0.0)),
                    "missing_data": is_missing_usage
                },
                "findings": [
                    {
                        "finding_id": f.get('finding_id', 'N/A'),
                        "finding_score": float(f.get('finding_score', 0.5)),
                        "finding_weight": float(f.get('finding_weight', 1.0)),
                        "n_tests": f.get('n_tests', 0)
                    }
                    for f in finding_results
                ],
                "tests": [
                    {
                        "test_name": t.get('test_name', 'Unknown'),
                        "finding_id": t.get('finding_id', 'N/A'),
                        "pi_human": float(t.get('pi_human', 0.5)),
                        "pi_agent": float(t.get('pi_agent', 0.5)),
                        "pas": float(t.get('pas') or t.get('score', 0.5)),
                        "test_weight": float(t.get('test_weight', 1.0)),
                        # Include p-value fields if available
                        "p_value_human": t.get('p_value_human'),
                        "p_value_agent": t.get('p_value_agent'),
                        "is_significant_human": bool(t.get('is_significant_human')) if t.get('is_significant_human') is not None else None,
                        "is_significant_agent": bool(t.get('is_significant_agent')) if t.get('is_significant_agent') is not None else None,
                        "direction_match": bool(t.get('direction_match')) if t.get('direction_match') is not None else None,
                        "replication_consistency": t.get('replication_consistency'),
                        "significance_level": t.get('significance_level')
                    }
                    for t in test_results
                ]
            }
            
            # Preserve _statistics field if present (for mixed models with aggregated results)
            if '_statistics' in pas_result:
                stats = pas_result['_statistics']
                study_data['_statistics'] = {
                    'mean_score': float(stats.get('mean_score', 0.0)),
                    'std_score': float(stats.get('std_score', 0.0)),
                    'n_iterations': int(stats.get('n_iterations', 0)),
                    'aggregated_from_iterations': bool(stats.get('aggregated_from_iterations', False))
                }
            
            model_studies.append(study_data)
            
            total_score += pas
            count += 1
            if pas > 0:
                positive_count += 1
        
        # Calculate averages
        avg_score = total_score / count if count > 0 else 0.0
        
        # PAS (mean chain): study=mean(findings), finding=mean(tests); only test level special
        pas_mean_chain = float(np.mean(pas_mean_study_list)) if pas_mean_study_list else None
        # PAS_agg: inverse-variance (Z-score) weighted aggregate of per-study PAS
        from src.evaluation.stats_lib import aggregate_pas_inverse_variance
        pas_raw_list = []
        pas_se_list = []
        for s in model_studies:
            pas = s.get("pas", {})
            r = pas.get("raw")
            if r is not None:
                pas_raw_list.append(r)
                pas_se_list.append(pas.get("raw_se", 0.0))
        pas_agg, pas_agg_se = (aggregate_pas_inverse_variance(pas_raw_list, pas_se_list)
                               if pas_raw_list else (float(avg_score), 0.0))
        
        # Calculate statistical replication statistics
        total_srs_score = sum(s.get("statistical_replication", {}).get("score", 0.0) 
                              for s in model_studies)
        total_srs_pass = sum(1 for s in model_studies 
                            if s.get("statistical_replication", {}).get("status") == "pass")
        avg_srs_score = total_srs_score / count if count > 0 else 0.0
        srs_pass_rate = float(total_srs_pass / count) if count > 0 else 0.0
        
        # ECS_Strict overall = global RMS-Stouffer over entire benchmark (all findings)
        # Already computed as ecs_strict_overall_value above.
        avg_consistency_score = ecs_strict_overall_value
        # Fallback if no findings (e.g. no test results at all)
        if not global_finding_p_values and count > 0:
            total_consistency_score = sum(s.get("statistical_replication", {}).get("replication_consistency", 0.0) 
                                         for s in model_studies)
            avg_consistency_score = total_consistency_score / count
        
        # Compute overall ECS_corr from all tests across all studies
        # Collect test results from original studies dict (which has effect_d fields)
        all_test_results = []
        for sid in sorted(studies.keys()):
            s = studies[sid]
            pas_result = s.get('pas_result', {})
            test_results = pas_result.get('test_results', [])
            for t in test_results:
                # Add study_id to test for grouping
                test_copy = t.copy()
                test_copy['study_id'] = sid
                all_test_results.append(test_copy)
        
        # Compute overall ECS_corr if we have test results with effect_d fields
        # Also compute ECS missing rate (per model-method overall)
        overall_ecs_corr = None
        ecs_missing_rate_overall = None
        n_tests_total = len(all_test_results)
        n_tests_valid_ecs = 0
        
        if all_test_results:
            from src.evaluation.stats_lib import compute_ecs_corr
            import math
            STUDY_GROUPS = {
                "Cognition": ["study_001", "study_002", "study_003", "study_004"],
                "Strategic": ["study_009", "study_010", "study_011", "study_012"],
                "Social": ["study_005", "study_006", "study_007", "study_008"]
            }
            
            # Count valid ECS tests (both human_effect_d and agent_effect_d present and finite)
            for t in all_test_results:
                h_d = t.get('human_effect_d')
                a_d = t.get('agent_effect_d')
                if (h_d is not None and a_d is not None and 
                    not math.isnan(h_d) and not math.isnan(a_d) and
                    not math.isinf(h_d) and not math.isinf(a_d)):
                    n_tests_valid_ecs += 1
            
            # Compute missing rate
            if n_tests_total > 0:
                ecs_missing_rate_overall = 1.0 - (n_tests_valid_ecs / n_tests_total)
            else:
                ecs_missing_rate_overall = None
            
            # compute_ecs_corr expects test_result dicts with 'agent_effect_d' and 'human_effect_d'
            ecs_result = compute_ecs_corr(all_test_results, study_groups=STUDY_GROUPS)
            
            # Main ECS (CCC-based)
            overall_ecs = ecs_result.get('ecs_overall')  # CCC overall
            ecs_per_study = ecs_result.get('ecs_per_study', {})  # CCC per study
            
            # Retained Pearson correlation (for figures/appendix)
            overall_ecs_corr = ecs_result.get('ecs_corr_overall')  # Pearson r overall
            ecs_corr_per_study = ecs_result.get('ecs_corr_per_study', {})  # Pearson r per study
            
            # Caricature regression (Pearson-based)
            caricature_overall = ecs_result.get('caricature_overall', {'a': None, 'b': None})
        else:
            overall_ecs = None
            ecs_per_study = {}
            overall_ecs_corr = None
            ecs_corr_per_study = {}
            caricature_overall = {'a': None, 'b': None}
        
        output["models"][model] = {
            "summary": {
                "total_studies": count,
                "positive_studies": positive_count,
                "pass_rate": float(positive_count / count) if count > 0 else 0.0,
                "average_pas_raw": float(avg_score),
                "average_pas": float(avg_score),  # Legacy alias
                "pas_agg": float(pas_agg),
                "pas_agg_se": float(pas_agg_se),
                "pas_mean_chain": float(pas_mean_chain) if pas_mean_chain is not None else None,
                
                # Main ECS (CCC-based)
                "ecs_overall": overall_ecs,  # CCC overall (main ECS metric)
                "ecs_per_study": ecs_per_study,  # CCC per study (main ECS metric)
                "average_ecs": float(overall_ecs) if overall_ecs is not None else float(avg_consistency_score),  # Main ECS for backward compat
                
                # Retained Pearson correlation (for figures/appendix)
                "ecs_corr_overall": overall_ecs_corr,  # Pearson r overall
                "ecs_corr_per_study": ecs_corr_per_study,  # Pearson r per study
                
                # Missing rate based on CCC availability (same logic as before since both require >= 3 points)
                "ecs_missing_rate_overall": ecs_missing_rate_overall,  # Missing rate (1 - valid/total)
                "ecs_n_tests_total": n_tests_total,  # Total number of tests
                "ecs_n_tests_valid": n_tests_valid_ecs,  # Number of tests with valid effect_d fields
                "ecs_strict_overall": float(avg_consistency_score),  # Appendix metric
                "caricature_overall": caricature_overall,  # Regression parameters: a (slope), b (intercept)
                "average_consistency_score": float(avg_consistency_score),  # Legacy alias
                "statistical_replication": {
                    "average_score": float(avg_srs_score),
                    "pass_rate": float(srs_pass_rate),
                    "passed_studies": int(total_srs_pass),
                    "total_studies": count
                },
                "total_usage": {
                    "input_tokens": int(total_input_tokens),
                    "output_tokens": int(total_output_tokens),
                    "total_tokens": int(total_all_tokens),
                    "total_cost": float(total_cost)
                }
            },
            "studies": model_studies
        }
    
    return output


# Module-level worker function for multiprocessing (must be at top level)
def _compute_single_bootstrap_worker(args):
    """
    Worker function for single bootstrap iteration.
    Must be a top-level function for multiprocessing to work.
    
    We resample AGENT RESPONSES (not participants) to measure agent uncertainty.
    """
    seed, idx, study_id_str, all_responses_list, is_flat_structure = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Import inside worker to avoid pickling issues
    import importlib.util
    from pathlib import Path as PathClass
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from collections import defaultdict
    
    try:
        # Load evaluator module (each worker loads it)
        evaluator_path = PathClass(f"src/studies/{study_id_str}_evaluator.py")
        spec = importlib.util.spec_from_file_location(f"{study_id_str}_evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)
        
        # Import normalization function
        from src.evaluation.stats_lib import aggregate_study_pas
        
        # Resample AGENT RESPONSES with replacement (not participants)
        # This measures uncertainty in agent's responses across trials
        n_responses = len(all_responses_list)
        resampled_indices = np.random.choice(n_responses, size=n_responses, replace=True)
        resampled_responses = [all_responses_list[i] for i in resampled_indices]
        
        # Convert back to the format expected by evaluate_study.
        # Many evaluators (e.g. study_001) expect nested: list of {participant_id, "responses": [...]}.
        # The pipeline converts flat→nested before calling evaluate_study; we must do the same here.
        if is_flat_structure:
            # Flat: each item is one response. Group by participant_id into nested form.
            by_participant = defaultdict(list)
            for r in resampled_responses:
                by_participant[r.get("participant_id", 0)].append(r)
            resampled_data = [{"participant_id": pid, "responses": rs} for pid, rs in by_participant.items()]
        else:
            # Nested structure: need to reorganize responses back into participants
            # Group resampled responses by participant_id
            participant_map = defaultdict(lambda: {"participant_id": None, "profile": {}, "responses": []})
            
            for response in resampled_responses:
                participant_id = response.get("participant_id", 0)
                if participant_map[participant_id]["participant_id"] is None:
                    # First response for this participant, store profile
                    participant_map[participant_id]["participant_id"] = participant_id
                    participant_map[participant_id]["profile"] = response.get("profile", {})
                
                # Extract just the response part (without participant-level fields)
                response_only = {k: v for k, v in response.items() 
                               if k not in ["participant_id", "profile"]}
                participant_map[participant_id]["responses"].append(response_only)
            
            resampled_data = list(participant_map.values())
        
        # Suppress errors
        error_capture = io.StringIO()
        output_capture = io.StringIO()
        
        with redirect_stderr(error_capture), redirect_stdout(output_capture):
            # Compute raw score
            raw_results = {"individual_data": resampled_data}
            pas_result = evaluator_module.evaluate_study(raw_results)
            
            if "error" in pas_result:
                return None
            
            raw_score = pas_result.get('score', 0.5)
            
            # Compute normalized score
            test_results = pas_result.get('test_results', [])
            if not test_results:
                return None
            
            from src.evaluation.stats_lib import aggregate_study_pas
            _, norm_score, _ = aggregate_study_pas(test_results)
            
            return (raw_score, norm_score)
            
    except Exception as e:
        # Suppress expected errors (JZS calculation errors have fallback)
        error_msg = str(e).lower()
        if "division by zero" not in error_msg and "jzs" not in error_msg.lower():
            # Only unexpected errors would reach here
            pass
        return None


def compute_bootstrap_se(
    study_id: str,
    config_path: Path,
    bootstrap_iterations: int = 200,
    verbose: bool = False,
    n_jobs: int = -1
) -> tuple:
    """
    Compute bootstrap standard error by resampling AGENT RESPONSES (not participants).
    This measures uncertainty in the agent's performance across different trials/responses.
    Uses multiprocessing for parallel computation.
    
    Args:
        study_id: Study ID (e.g., "study_001")
        config_path: Path to config directory containing full_benchmark.json
        bootstrap_iterations: Number of bootstrap iterations (default: 200)
        verbose: Whether to show progress
        n_jobs: Number of parallel workers (-1 for all CPUs, default: -1)
        
    Returns:
        Tuple of (score_se, normalized_score_se, success)
    """
    try:
        # Load evaluator module
        evaluator_path = Path(f"src/studies/{study_id}_evaluator.py")
        if not evaluator_path.exists():
            if verbose:
                print(f"  ⚠️  Evaluator not found: {evaluator_path}")
            return 0.0, 0.0, False
        
        # Import evaluator module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator_module)
        
        # Load full_benchmark.json with participant data
        full_benchmark_path = config_path / "full_benchmark.json"
        if not full_benchmark_path.exists():
            if verbose:
                print(f"  ⚠️  full_benchmark.json not found: {full_benchmark_path}")
            return 0.0, 0.0, False
        
        with open(full_benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        # Extract agent responses (not participants - we resample agent responses to measure agent uncertainty)
        individual_data = benchmark_data.get('individual_data', [])
        if not individual_data:
            if verbose:
                print(f"  ⚠️  No individual_data found in full_benchmark.json")
            return 0.0, 0.0, False
        
        # Check if data is flat structure or nested structure
        is_flat_structure = len(individual_data) > 0 and 'responses' not in individual_data[0]
        
        # Extract all agent responses (flatten if nested)
        all_responses = []
        if is_flat_structure:
            # Flat structure: each item is already a response
            all_responses = individual_data
        else:
            # Nested structure: extract all responses from all participants
            for participant in individual_data:
                participant_id = participant.get('participant_id', 0)
                profile = participant.get('profile', {})
                for response in participant.get('responses', []):
                    # Combine participant info with response info
                    full_response = {
                        "participant_id": participant_id,
                        "profile": profile,
                        **response  # Include all response fields
                    }
                    all_responses.append(full_response)
        
        n_responses = len(all_responses)
        if n_responses == 0:
            if verbose:
                print(f"  ⚠️  No agent responses found in individual_data")
            return 0.0, 0.0, False
        
        # Determine number of workers
        import multiprocessing as mp
        import time
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, bootstrap_iterations, mp.cpu_count())
        
        print(f"  → Running bootstrap with {bootstrap_iterations} iterations on {n_responses} agent responses...")
        print(f"    (Resampling agent responses to measure agent uncertainty, not participant uncertainty)")
        print(f"    Using {n_jobs} parallel workers for acceleration")
        
        # Generate all seeds and indices (for reproducibility)
        base_seed = 42
        seeds = [base_seed + i for i in range(bootstrap_iterations)]
        indices = list(range(bootstrap_iterations))
        
        # Prepare arguments for workers (pass all_responses and structure type)
        worker_args = [(seed, idx, study_id, all_responses, is_flat_structure) 
                      for seed, idx in zip(seeds, indices)]
        
        # Parallel computation
        start_time = time.time()
        with mp.Pool(n_jobs) as pool:
            # Use imap for better progress tracking
            results = []
            completed = 0
            
            # Show progress more frequently (every 5 iterations or every worker, whichever is smaller)
            progress_interval = min(5, max(1, bootstrap_iterations // 20))  # Show at least 20 updates
            
            for result in pool.imap(_compute_single_bootstrap_worker, worker_args):
                if result is not None:
                    results.append(result)
                completed += 1
                
                # Show progress more frequently
                if completed % progress_interval == 0 or completed == 1 or completed == bootstrap_iterations:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (bootstrap_iterations - completed) / rate if rate > 0 else 0
                    progress_pct = completed / bootstrap_iterations * 100
                    # Use carriage return to update same line
                    bar_length = 40
                    filled = int(bar_length * completed / bootstrap_iterations)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r    [{completed:4d}/{bootstrap_iterations}] {bar} {progress_pct:5.1f}% | "
                          f"Elapsed: {elapsed:6.1f}s | ETA: {remaining:6.1f}s | {rate:5.1f} iter/s", 
                          end='', flush=True)
            
            # Print newline after progress bar
            print()  # New line after progress
        
        elapsed_total = time.time() - start_time
        
        # Separate raw and normalized scores
        if results:
            bootstrap_scores = [r[0] for r in results]
            bootstrap_norm_scores = [r[1] for r in results]
        else:
            bootstrap_scores = []
            bootstrap_norm_scores = []
        
        if not bootstrap_scores:
            print(f"  ⚠️  All bootstrap iterations failed")
            return 0.0, 0.0, False
        
        # Calculate standard error (standard deviation of bootstrap distribution)
        score_se = float(np.std(bootstrap_scores, ddof=1)) if bootstrap_scores else 0.0
        normalized_score_se = float(np.std(bootstrap_norm_scores, ddof=1)) if bootstrap_norm_scores else 0.0
        
        print(f"  ✓ Bootstrap SE computed: {score_se:.4f} (raw), {normalized_score_se:.4f} (normalized)")
        print(f"    Total time: {elapsed_total:.1f}s, Successful iterations: {len(results)}/{bootstrap_iterations}")
        if n_jobs > 1:
            estimated_single_thread_time = elapsed_total * n_jobs
            speedup = estimated_single_thread_time / elapsed_total if elapsed_total > 0 else 1.0
            print(f"    Speedup: ~{speedup:.1f}x (estimated single-thread time: {estimated_single_thread_time:.1f}s)")
        
        return score_se, normalized_score_se, True
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Bootstrap computation failed: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return 0.0, 0.0, False


def get_se_values(pas_result: dict, se_method: str = 'auto', verbose: bool = False,
                  compute_bootstrap: bool = False, bootstrap_iterations: int = 200,
                  study_id: str = None, config_path: Path = None, n_jobs: int = -1,
                  allow_write: bool = False, force_recompute_bootstrap: bool = False) -> tuple:
    """
    Get standard error values based on the selected method.
    
    Args:
        pas_result: Dictionary containing evaluation results
        se_method: Method to use ('repeat', 'bootstrap', or 'auto')
        verbose: Whether to print progress information
        compute_bootstrap: Whether to compute bootstrap SE if not available
        bootstrap_iterations: Number of bootstrap iterations (default: 200)
        study_id: Study ID for bootstrap computation
        config_path: Config path for bootstrap computation
        n_jobs: Number of parallel workers for bootstrap (-1 for all CPUs)
        
    Returns:
        Tuple of (score_se, normalized_score_se, method_used)
    """
    variance_method = pas_result.get('variance_method', 'none')
    score_se = pas_result.get('score_se', 0.0)
    normalized_score_se = pas_result.get('normalized_score_se', 0.0)
    
    # Check available methods
    has_repeat = pas_result.get('score_repeat_se') is not None
    existing_bootstrap_iterations = pas_result.get('bootstrap_iterations', 0)
    has_bootstrap = existing_bootstrap_iterations > 0
    n_runs = pas_result.get('n_runs', 1)
    
    if verbose:
        print(f"  Checking SE methods:")
        print(f"    - Repeat-based: {'✓ Available' if has_repeat else '✗ Not available'} (n_runs={n_runs})")
        print(f"    - Bootstrap: {'✓ Available' if has_bootstrap else '✗ Not available'} (iterations={existing_bootstrap_iterations})")
        print(f"    - Current variance_method: {variance_method}")
    
    # Auto: prefer repeat-based if available, else bootstrap, else whatever is in score_se
    if se_method == 'auto':
        if has_repeat:
            method_used = 'repeat'
            score_se = pas_result.get('score_repeat_se', score_se)
            normalized_score_se = pas_result.get('normalized_score_repeat_se', normalized_score_se)
            if verbose:
                print(f"  → Using repeat-based SE: {score_se:.4f} (from {n_runs} runs)")
        elif has_bootstrap:
            method_used = 'bootstrap'
            if verbose:
                print(f"  → Using bootstrap SE: {score_se:.4f} (from {existing_bootstrap_iterations} iterations)")
        else:
            method_used = variance_method if variance_method != 'none' else 'none'
            if verbose:
                print(f"  → Using default SE: {score_se:.4f} (method: {method_used})")
    elif se_method == 'repeat':
        if has_repeat:
            method_used = 'repeat'
            score_se = pas_result.get('score_repeat_se', score_se)
            normalized_score_se = pas_result.get('normalized_score_repeat_se', normalized_score_se)
            if verbose:
                print(f"  → Using repeat-based SE: {score_se:.4f} (from {n_runs} runs)")
        else:
            method_used = 'none'
            score_se = 0.0
            normalized_score_se = 0.0
            if verbose:
                print(f"  ⚠️  Repeat-based SE requested but not available (n_runs={n_runs}, need ≥2)")
    elif se_method == 'bootstrap':
        if has_bootstrap and not force_recompute_bootstrap:
            method_used = 'bootstrap'
            if verbose:
                print(f"  ✓ Bootstrap SE available: {score_se:.4f} ± SE (from {existing_bootstrap_iterations} bootstrap iterations)")
                print(f"    Bootstrap process: Resampled {existing_bootstrap_iterations} times from participant pool")
        else:
            # Try to compute bootstrap if requested (or force recompute to fix flat/nested bug)
            if (compute_bootstrap or force_recompute_bootstrap) and study_id and config_path:
                if verbose:
                    print(f"  → Computing bootstrap SE for {study_id} (this may take a while)...")
                # Always show progress for bootstrap computation (verbose=True)
                new_score_se, new_norm_score_se, success = compute_bootstrap_se(
                    study_id, config_path, bootstrap_iterations, verbose=True, n_jobs=n_jobs
                )
                if success:
                    method_used = 'bootstrap'
                    score_se = new_score_se
                    normalized_score_se = new_norm_score_se
                    # Update pas_result for future use
                    pas_result['score_se'] = score_se
                    pas_result['normalized_score_se'] = normalized_score_se
                    pas_result['bootstrap_iterations'] = bootstrap_iterations
                    pas_result['variance_method'] = 'bootstrap'
                    
                    # Save to evaluation_results.json file (only if write is allowed)
                    if allow_write:
                        eval_results_path = config_path / "evaluation_results.json"
                        if eval_results_path.exists():
                            try:
                                with open(eval_results_path, 'r', encoding='utf-8') as f:
                                    eval_data = json.load(f)
                                
                                # Update with bootstrap results
                                eval_data['score_se'] = float(score_se)
                                eval_data['normalized_score_se'] = float(normalized_score_se)
                                eval_data['bootstrap_iterations'] = bootstrap_iterations
                                eval_data['variance_method'] = 'bootstrap'
                                eval_data['variance_method_note'] = f'Bootstrap SE computed with {bootstrap_iterations} iterations'
                                
                                # Save back to file
                                with open(eval_results_path, 'w', encoding='utf-8') as f:
                                    json.dump(eval_data, f, indent=2, ensure_ascii=False)
                                
                                if verbose:
                                    print(f"  ✓ Bootstrap SE saved to {eval_results_path}")
                            except Exception as e:
                                if verbose:
                                    print(f"  ⚠️  Failed to save bootstrap SE to file: {e}")
                        else:
                            if verbose:
                                print(f"  ⚠️  evaluation_results.json not found, cannot save bootstrap SE")
                    else:
                        if verbose:
                            print(f"  ⚠️  Write disabled (default): Bootstrap SE computed but not saved to file. Use --allow-write to save.")
                        # Don't save - files are protected by default
                else:
                    method_used = 'none'
                    score_se = 0.0
                    normalized_score_se = 0.0
            else:
                method_used = 'none'
                score_se = 0.0
                normalized_score_se = 0.0
                if verbose:
                    print(f"  ⚠️  Bootstrap SE not available (will use SE=0)")
                    if not compute_bootstrap:
                        print(f"    Use --compute-bootstrap to calculate bootstrap SE on the fly")
    else:
        method_used = variance_method if variance_method != 'none' else 'none'
    
    return float(score_se), float(normalized_score_se), method_used


def main():
    parser = argparse.ArgumentParser(description="Generate HumanStudy-Bench results tables")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--format", choices=['summary', 'detailed', 'csv', 'json', 'all'], default='all', help="Output format")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    parser.add_argument(
        "--se-method",
        choices=['auto', 'repeat', 'bootstrap'],
        default='auto',
        help="Method for standard error calculation: 'auto' (prefer repeat if available, else bootstrap), 'repeat' (use repeat-based SE only), or 'bootstrap' (use bootstrap SE only). Default: auto"
    )
    parser.add_argument(
        "--allow-backfill",
        action='store_true',
        help="Allow backfilling effect sizes and p-values in evaluation_results.json files. By default, backfill is DISABLED to protect existing files."
    )
    parser.add_argument(
        "--allow-write",
        action='store_true',
        help="Allow writing to evaluation_results.json files (e.g., for bootstrap SE). By default, files are read-only to prevent accidental modifications."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information, especially for bootstrap SE processing"
    )
    parser.add_argument(
        "--compute-bootstrap",
        action="store_true",
        help="Actually compute bootstrap SE for studies that don't have it (may take time)"
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
        help="Number of bootstrap iterations when --compute-bootstrap is used (default: 200)"
    )
    parser.add_argument(
        "--bootstrap-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for bootstrap computation (-1 for all CPUs, default: -1)"
    )
    parser.add_argument(
        "--force-recompute-bootstrap",
        action="store_true",
        help="Recompute bootstrap SE for all studies even when already present (e.g. after fixing flat/nested bug)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return
    
    # Load results first
    results = load_all_results(results_dir)
    
    # Run in-memory backfill on loaded data to populate missing effect_d fields for ECS calculation
    # This modifies data in memory only, without writing to files (optional: requires backfill_effect_sizes)
    try:
        from scripts.backfill_effect_sizes import backfill_effect_d_fields, backfill_p_value_fields, build_gt_test_lookup
    except ImportError:
        print("Warning: scripts.backfill_effect_sizes not available; skipping in-memory backfill (ECS may use existing effect_d only).\n")
        backfill_effect_d_fields = backfill_p_value_fields = build_gt_test_lookup = None
    if backfill_effect_d_fields is not None:
        print("Running in-memory backfill for ECS calculation (files not modified)...")
        try:
            total_backfilled = 0
            for model_config, studies in results.items():
                for study_id, study_data in studies.items():
                    pas_result = study_data.get('pas_result', {})
                    test_results = pas_result.get('test_results', [])
                    if not test_results:
                        continue
                    
                    gt_path = Path(__file__).parent.parent / "data" / "studies" / study_id / "ground_truth.json"
                    gt_lookup = None
                    if gt_path.exists():
                        try:
                            with open(gt_path, 'r', encoding='utf-8') as f:
                                ground_truth = json.load(f)
                            gt_lookup = build_gt_test_lookup(ground_truth)
                        except Exception:
                            pass
                    
                    for test in test_results:
                        effect_d_updated = backfill_effect_d_fields(test)
                        p_value_updated = backfill_p_value_fields(test, study_id, gt_lookup)
                        if effect_d_updated or p_value_updated:
                            total_backfilled += 1
            
            print(f"In-memory backfill: {total_backfilled} test results backfilled (used for ECS calculation only)")
            print("(Files not modified - backfilled values exist only in memory)\n")
            
            if args.allow_backfill:
                print("Writing backfilled values to files (--allow-backfill enabled)...")
                from scripts.backfill_effect_sizes import backfill_results_directory
                benchmark_dir = results_dir / "benchmark" if (results_dir / "benchmark").exists() else results_dir
                files_processed, total_updated, total_skipped = backfill_results_directory(benchmark_dir, dry_run=False)
                print(f"Backfill complete: {files_processed} files processed, {total_updated} test results updated, {total_skipped} skipped\n")
        except Exception as e:
            print(f"Warning: In-memory backfill failed: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without backfill...\n")
    
    if not results:
        print("No results found!")
        return
    
    # Show processing info if verbose or bootstrap method selected
    verbose_mode = args.verbose or (args.se_method == 'bootstrap')
    
    # Track method availability across all studies for summary
    method_stats = {
        'total_studies': 0,
        'has_repeat': 0,
        'has_bootstrap': 0,
        'missing_requested_method': []
    }
    
    if verbose_mode:
        print(f"\n{'='*80}")
        print(f"Processing results with SE method: {args.se_method}")
        print(f"{'='*80}\n")
    
    output_lines = []
    output_lines.append(f"# HumanStudy-Bench Results Report")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Standard Error Method: {args.se_method}\n")
    
    if args.format in ['summary', 'all']:
        output_lines.append("\n# SUMMARY TABLE\n")
        output_lines.append(generate_summary_table(
            results, se_method=args.se_method, verbose=verbose_mode, method_stats=method_stats,
            compute_bootstrap=args.compute_bootstrap, bootstrap_iterations=args.bootstrap_iterations,
            results_dir=results_dir, n_jobs=args.bootstrap_jobs, allow_write=args.allow_write,
            force_recompute_bootstrap=args.force_recompute_bootstrap
        ))
    
    if args.format in ['detailed', 'all']:
        output_lines.append(generate_detailed_table(
            results, se_method=args.se_method, verbose=verbose_mode, method_stats=method_stats,
            compute_bootstrap=args.compute_bootstrap, bootstrap_iterations=args.bootstrap_iterations,
            results_dir=results_dir, n_jobs=args.bootstrap_jobs, allow_write=args.allow_write,
            force_recompute_bootstrap=args.force_recompute_bootstrap
        ))
    
    if args.format in ['csv', 'all']:
        output_lines.append("\n# CSV DATA\n```csv")
        output_lines.append(generate_csv(results))
        output_lines.append("```")
    
    # Generate JSON format if requested
    if args.format in ['json', 'all']:
        json_data = generate_json_summary(
            results, se_method=args.se_method, verbose=verbose_mode,
            compute_bootstrap=args.compute_bootstrap, bootstrap_iterations=args.bootstrap_iterations,
            results_dir=results_dir, n_jobs=args.bootstrap_jobs, allow_write=args.allow_write,
            force_recompute_bootstrap=args.force_recompute_bootstrap
        )
        if args.output:
            json_output_file = Path(args.output).with_suffix('.json')
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"JSON summary saved to {json_output_file}")
        else:
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
    
    # Show summary of method availability if verbose
    if verbose_mode and method_stats['total_studies'] > 0:
        print(f"\n{'='*80}")
        print(f"SE Method Availability Summary")
        print(f"{'='*80}")
        print(f"Total studies processed: {method_stats['total_studies']}")
        print(f"  - Repeat-based SE available: {method_stats['has_repeat']}/{method_stats['total_studies']}")
        print(f"  - Bootstrap SE available: {method_stats['has_bootstrap']}/{method_stats['total_studies']}")
        
        if method_stats['missing_requested_method']:
            print(f"\n⚠️  {len(method_stats['missing_requested_method'])} studies missing requested SE method ({args.se_method}):")
            for sid in method_stats['missing_requested_method'][:10]:  # Show first 10
                print(f"    - {sid}")
            if len(method_stats['missing_requested_method']) > 10:
                print(f"    ... and {len(method_stats['missing_requested_method']) - 10} more")
            if args.se_method == 'bootstrap':
                print(f"\n  To enable bootstrap for these studies, run:")
                print(f"    python generation_pipeline/run.py --stage 6 --study-id <study_id> --enable-bootstrap")
        print(f"{'='*80}\n")
    
    # Only output markdown/text formats if not JSON-only
    if args.format != 'json':
        output = "\n".join(output_lines)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()



