#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import importlib.util
import sys
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_all_participants_by_method(results_dir: Path, study_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all participant data for a given study, grouped by experimental method (v1-v4).
    
    Groups participants by their ACTUAL method (from system_prompt_preset or variant field),
    not just by directory name, to ensure proper separation of A1-A4 variants.
    """
    participants_by_method = defaultdict(list)
    study_dir = results_dir / "benchmark" / study_id
    if not study_dir.exists():
        print(f"Warning: Study directory {study_dir} not found.")
        return {}
    
    # Map directory suffixes to methods (for fallback)
    method_map = {
        "v1-empty": "v1",
        "v2-human": "v2",
        "v3-human-plus-demo": "v3",
        "v4-background": "v4"
    }
    
    # Map system_prompt_preset values to methods
    preset_to_method = {
        "v1-empty": "v1",
        "v1": "v1",
        "v2-human": "v2",
        "v2": "v2",
        "v3-human-plus-demo": "v3",
        "v3": "v3",
        "v4-background": "v4",
        "v4": "v4"
    }
    
    for model_config_dir in study_dir.iterdir():
        if not model_config_dir.is_dir():
            continue
        
        # Skip mixed_models directories (they're outputs, not sources)
        if model_config_dir.name.startswith("mixed_models"):
            continue
        
        # Skip example/test directories
        if "example" in model_config_dir.name.lower():
            continue
        
        # Skip temperature ablation models (experimental variations, not different base models)
        # Patterns: _temp0.1, _temp0.3, _temp0.5, _temp0.7, temperature, etc.
        import re
        if re.search(r'_temp0?\.?\d+|temperature', model_config_dir.name, re.IGNORECASE):
            continue
        
        # Skip x_ai_grok_4.1_fast_none (experimental variation)
        if "x_ai_grok_4.1_fast_none" in model_config_dir.name:
            continue
        
        # Try full_benchmark.json first
        benchmark_file = model_config_dir / "full_benchmark.json"
        participants = []
        metadata_system_preset = None
        
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    participants = data.get('individual_data', [])
                    # Get the actual system_prompt_preset from metadata
                    metadata_system_preset = data.get('system_prompt_preset')
            except Exception as e:
                print(f"Warning: Could not load {benchmark_file}: {e}")
        
        # Fallback to raw_responses.jsonl if full_benchmark.json fails or is missing
        if not participants:
            raw_responses_file = model_config_dir / "raw_responses.jsonl"
            if raw_responses_file.exists():
                try:
                    with open(raw_responses_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                resp = json.loads(line)
                                if 'responses' in resp:
                                    p = resp
                                else:
                                    p = {
                                        "participant_id": resp.get("participant_id"),
                                        "profile": resp.get("participant_profile") or resp.get("profile", {}),
                                        "responses": [resp]
                                    }
                                participants.append(p)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Warning: Could not load {raw_responses_file}: {e}")
        
        if not participants:
            continue
        
        # Determine method: prioritize metadata > directory name > variant field
        # (variant field is least reliable as it may be hardcoded in study config)
        method = None
        
        # 1. Try metadata system_prompt_preset (most reliable - reflects what was actually used)
        if metadata_system_preset:
            method = preset_to_method.get(metadata_system_preset)
        
        # 2. Fallback to directory name (reliable - directory name reflects system prompt used)
        if not method:
            for suffix, m_key in method_map.items():
                if model_config_dir.name.endswith(suffix):
                    method = m_key
                    break
        
        # 3. Last resort: try to infer from participant variant fields
        # (This is unreliable since variant may be hardcoded, but better than nothing)
        if not method:
            sample_size = min(10, len(participants))
            variant_counts = defaultdict(int)
            for p in participants[:sample_size]:
                for resp in p.get('responses', []):
                    trial_info = resp.get('trial_info', {})
                    variant = trial_info.get('variant')
                    if variant:
                        variant_counts[variant] += 1
            
            if variant_counts:
                most_common_variant = max(variant_counts.items(), key=lambda x: x[1])[0]
                method = preset_to_method.get(most_common_variant)
        
        if not method:
            print(f"Warning: Could not determine method for {model_config_dir.name}, skipping.")
            continue
        
        # Add to the correct method pool
        # NOTE: We trust the directory name (which reflects actual system_prompt_preset used)
        # over the variant field in trial_info (which may be hardcoded in study config)
        for p in participants:
            p['_source_model'] = model_config_dir.name
            participants_by_method[method].append(p)
                
    return participants_by_method

import time
from datetime import datetime

def get_target_sample_size(results_dir: Path, study_id: str) -> int:
    """Find the target sample size from existing benchmark results for this study."""
    study_dir = results_dir / "benchmark" / study_id
    if not study_dir.exists():
        return 100  # Default fallback
    
    for config_dir in study_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        benchmark_file = config_dir / "full_benchmark.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    participants = data.get('individual_data', [])
                    if participants:
                        return len(participants)
            except:
                continue
                
    return 100  # Default fallback

def aggregate_evaluation_results(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate multiple evaluation results by averaging across iterations.
    
    Returns a single evaluation result with averaged scores.
    """
    if not all_results:
        return {}
    
    # Aggregate overall scores
    overall_scores = [r["score"] for r in all_results if "score" in r]
    avg_score = np.mean(overall_scores) if overall_scores else 0.0
    
    # Aggregate finding_results
    finding_dict = defaultdict(list)
    for r in all_results:
        for finding in r.get("finding_results", []):
            fid = finding.get("finding_id")
            if fid:
                finding_dict[fid].append(finding.get("finding_score", 0.0))
    
    aggregated_findings = []
    for fid, scores in finding_dict.items():
        # Use first finding's structure as template
        template = next((f for r in all_results for f in r.get("finding_results", []) if f.get("finding_id") == fid), {})
        aggregated_findings.append({
            "finding_id": fid,
            "sub_study_id": template.get("sub_study_id", "unknown"),
            "finding_score": float(np.mean(scores)),
            "finding_weight": template.get("finding_weight", 1.0),
            "n_tests": template.get("n_tests", 1),
            "_std": float(np.std(scores)),  # Include std for reference
            "_n_iterations": len(scores)
        })
    
    # Aggregate test_results (use first iteration's structure, but average pi_agent and pas)
    test_dict = defaultdict(list)
    for r in all_results:
        for test in r.get("test_results", []):
            test_key = (test.get("finding_id"), test.get("test_name"), test.get("sub_study_id"))
            test_dict[test_key].append(test)
    
    aggregated_tests = []
    for test_key, test_list in test_dict.items():
        # Use first test as template
        template = test_list[0]
        
        # Average numeric fields
        pi_agent_vals = [t.get("pi_agent", 0.5) for t in test_list if "pi_agent" in t]
        pas_vals = [t.get("pas", 0.5) for t in test_list if "pas" in t]
        
        aggregated_test = dict(template)  # Copy all fields
        aggregated_test["pi_agent"] = float(np.mean(pi_agent_vals)) if pi_agent_vals else 0.5
        aggregated_test["pas"] = float(np.mean(pas_vals)) if pas_vals else 0.5
        aggregated_test["_pi_agent_std"] = float(np.std(pi_agent_vals)) if len(pi_agent_vals) > 1 else 0.0
        aggregated_test["_bas_std"] = float(np.std(pas_vals)) if len(pas_vals) > 1 else 0.0
        aggregated_test["_n_iterations"] = len(test_list)
        
        aggregated_tests.append(aggregated_test)
    
    # Build final aggregated result
    return {
        "score": float(avg_score),
        "finding_results": aggregated_findings,
        "test_results": aggregated_tests,
        "_aggregated": True,
        "_n_iterations": len(all_results)
    }


def save_as_model_result(output_dir: Path, study_id: str, method: str, sample: List[Dict[str, Any]], score_result: Dict[str, Any], results_dir: Path, stats: Dict[str, Any] = None):
    """Save results in the same format as other models (full_benchmark.json and evaluation_results.json)."""
    # Create the target directory: results/benchmark/{study_id}/mixed_models_{method}
    target_dir = results_dir / "benchmark" / study_id / f"mixed_models_{method}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create full_benchmark.json
    full_benchmark = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "study_id": study_id,
        "title": score_result.get("title", f"Mixed Models Benchmark - {study_id}"),
        "model": "mixed_models",
        "system_prompt_preset": method,
        "random_seed": 42,
        "individual_data": sample,
        "status": "COMPLETED"
    }
    
    with open(target_dir / "full_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(full_benchmark, f, indent=2, ensure_ascii=False)
        
    # 2. Create evaluation_results.json
    evaluation_results = {
        "study_id": study_id,
        "model": "mixed_models",
        "method": method,
        "score": score_result.get("score", 0.0),
        "normalized_score": 2 * score_result.get("score", 0.0) - 1,
        "finding_results": score_result.get("finding_results", []),
        "test_results": score_result.get("test_results", []),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add any other fields from the evaluator result (but preserve _statistics if we add it)
    for k, v in score_result.items():
        if k not in evaluation_results and k != "individual_data":
            evaluation_results[k] = v
    
    # Add statistics if provided (do this AFTER copying score_result fields to ensure it's not overwritten)
    if stats:
        evaluation_results["_statistics"] = {
            "mean_score": stats.get("mean_score"),
            "std_score": stats.get("std_score"),
            "n_iterations": stats.get("iterations"),
            "aggregated_from_iterations": True
        }
        print(f"    DEBUG: Added _statistics to evaluation_results: mean={stats.get('mean_score'):.4f}, std={stats.get('std_score'):.4f}, n={stats.get('iterations')}")
    else:
        print(f"    WARNING: No stats provided to save_as_model_result for {study_id} {method} - stats will be missing!")
            
    from src.utils.io import atomic_write_json
    atomic_write_json(target_dir / "evaluation_results.json", evaluation_results, indent=2, ensure_ascii=False, encoding='utf-8')
        
    return target_dir

def evaluate_worker(args):
    """Worker function for parallel random sampling and evaluation."""
    study_id, pool, sample_size, seed, evaluator_path = args
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample with replacement
    if not pool:
        return None
    sample = random.choices(pool, k=sample_size)
    
    # Import evaluator
    spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", evaluator_path)
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    
    # Prepare results for evaluator
    results = {"individual_data": sample}
    
    try:
        score_result = evaluator_module.evaluate_study(results)
        return {
            "score": score_result.get("score", 0.0),
            "finding_results": score_result.get("finding_results", []),
            "test_results": score_result.get("test_results", []),
            "seed": seed,
            "sample": sample # Include sample if needed for saving
        }
    except Exception as e:
        # print(f"Error evaluating sample for {study_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate random alignment scores by sampling from all model results")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output-dir", type=str, default="results/random_alignment", help="Output directory")
    parser.add_argument("--iterations", type=int, default=100, help="Number of bootstrap iterations")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of participants to sample per iteration")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--studies", type=str, help="Comma-separated list of studies to process (default: all)")
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.jobs <= 0:
        args.jobs = mp.cpu_count()
    
    # Determine studies to process
    if args.studies:
        study_ids = args.studies.split(",")
    else:
        study_ids = sorted([d.name for d in (results_dir / "benchmark").iterdir() if d.is_dir() and d.name.startswith("study_")])
    
    summary_records = []
    
    for study_id in study_ids:
        print(f"\nProcessing {study_id}...")
        
        # 1. Load participants grouped by method
        method_pools = load_all_participants_by_method(results_dir, study_id)
        if not method_pools:
            print(f"  Skipping {study_id}: No participant data found.")
            continue
        
        # Determine target sample size from human studies (match existing runs)
        target_n = get_target_sample_size(results_dir, study_id)
        print(f"  Target sample size (matching human/benchmark): {target_n}")
        
        # 2. Setup evaluator
        evaluator_path = Path(f"src/studies/{study_id}_evaluator.py")
        if not evaluator_path.exists():
            print(f"  Skipping {study_id}: Evaluator not found at {evaluator_path}.")
            continue
            
        for method, pool in sorted(method_pools.items()):
            print(f"\n  Method {method}: Found {len(pool)} total participants in the pool.")
            
            # 3. Run bootstrapping in parallel for this method
            # We'll use one specific iteration to save as the "mixed_model" result
            # but still run iterations for statistics
            worker_args = [(study_id, pool, target_n, random.randint(0, 1000000), str(evaluator_path)) 
                           for _ in range(args.iterations)]
            
            all_scores = []
            finding_scores = defaultdict(list)
            representative_result = None
            
            with mp.Pool(args.jobs) as p:
                results = list(tqdm(p.imap(evaluate_worker, worker_args), total=args.iterations, desc=f"    Bootstrapping {study_id} ({method})"))
                
            for r in results:
                if r is not None:
                    all_scores.append(r["score"])
                    for f in r["finding_results"]:
                        finding_scores[f["finding_id"]].append(f["finding_score"])
            
            # Filter valid results for aggregation
            valid_results = [r for r in results if r is not None]
            
            if not all_scores or not valid_results:
                print(f"    Warning: No successful evaluations for {study_id} ({method}).")
                continue
            
            # 4. Aggregate all iterations into a single averaged result
            aggregated_result = aggregate_evaluation_results(valid_results)
            
            # Use first result's sample as representative sample for full_benchmark.json
            representative_sample = valid_results[0]["sample"]
            
            # 5. Calculate stats
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            q05 = np.percentile(all_scores, 5)
            q95 = np.percentile(all_scores, 95)
            mean_norm = 2 * mean_score - 1
            
            stats_dict = {
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "iterations": args.iterations,
                "mean_normalized_score": float(mean_norm),
                "percentiles": {
                    "p2.5": float(np.percentile(all_scores, 2.5)),
                    "p5": float(q05),
                    "p50": float(np.median(all_scores)),
                    "p95": float(q95),
                    "p97.5": float(np.percentile(all_scores, 97.5))
                }
            }
            
            # 6. Save as a model-like result in mixed_models folder (with aggregated results)
            # Verify stats_dict is not None before saving
            if not stats_dict or not stats_dict.get("mean_score") is not None:
                print(f"    ERROR: stats_dict is invalid for {study_id} {method}: {stats_dict}")
            save_path = save_as_model_result(output_dir, study_id, method, 
                                            representative_sample, 
                                            aggregated_result, 
                                            results_dir,
                                            stats=stats_dict)
            print(f"    ✓ Saved aggregated results (n={args.iterations} iterations) to {save_path}")
            
            # Verify _statistics was saved
            eval_file = save_path / "evaluation_results.json"
            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        saved_data = json.load(f)
                        if "_statistics" in saved_data:
                            print(f"    ✓ Verified _statistics saved: mean={saved_data['_statistics']['mean_score']:.4f}, std={saved_data['_statistics']['std_score']:.4f}")
                        else:
                            print(f"    ⚠️  WARNING: _statistics NOT found in saved file!")
                except Exception as e:
                    print(f"    ⚠️  Could not verify saved file: {e}")
            
            print(f"    Random Alignment Score (PAS): {mean_score:.4f} ± {std_score:.4f}")
            print(f"    Normalized PAS: {mean_norm:.4f} ± {2*std_score:.4f}")
            
            # 7. Save detailed results for this study-method (original random_alignment format)
            study_method_output = {
                "study_id": study_id,
                "method": method,
                "total_pool_size": len(pool),
                "iterations": args.iterations,
                "sample_size": target_n,
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "mean_normalized_score": float(mean_norm),
                "percentiles": stats_dict["percentiles"],
                "finding_stats": {
                    fid: {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores))
                    } for fid, scores in finding_scores.items()
                },
                "raw_scores": [float(s) for s in all_scores]
            }
            
            method_dir = output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            with open(method_dir / f"{study_id}_random_alignment.json", 'w') as f:
                json.dump(study_method_output, f, indent=2)
                
            summary_records.append({
                "Study": study_id,
                "Method": method,
                "Pool_Size": len(pool),
                "Sample_Size": target_n,
                "Mean_BAS": mean_score,
                "Std_BAS": std_score,
                "Norm_BAS": mean_norm,
                "Norm_Std": 2*std_score,
                "P95_Upper": q95
            })
    
    # 6. Save summary
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(output_dir / "summary.csv", index=False)
        
        # Save as JSON too
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary_records, f, indent=2)
            
        print(f"\n✓ Random alignment calculations complete! Results saved to {output_dir}")
        print("\nSummary Table:")
        print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()

