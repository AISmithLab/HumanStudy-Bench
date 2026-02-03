#!/usr/bin/env python3
"""
Run pipeline stages 3-6 for multiple studies in parallel.

Usage:
    python scripts/run_studies_parallel.py --studies study_003 study_004 ...
    python scripts/run_studies_parallel.py --all  # Run study_003 to study_012
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_stage(study_id: str, stage: int, **kwargs) -> tuple:
    """
    Run a single stage for a study.
    
    Returns:
        (study_id, stage, success: bool, output: str, error: str)
    """
    cmd = ["python", "generation_pipeline/run.py", f"--stage", str(stage), f"--study-id", study_id]
    
    # Add stage-specific arguments
    if stage == 5:
        if kwargs.get("model"):
            cmd.append("--model")
            cmd.append(kwargs["model"])
        if kwargs.get("real_llm"):
            cmd.append("--real-llm")
        if kwargs.get("num_workers"):
            cmd.append("--num-workers")
            cmd.append(str(kwargs["num_workers"]))
        if kwargs.get("n_participants"):
            cmd.append("--n-participants")
            cmd.append(str(kwargs["n_participants"]))
        if kwargs.get("run_name"):
            cmd.append("--run-name")
            cmd.append(kwargs["run_name"])
        if kwargs.get("reasoning"):
            cmd.append("--reasoning")
            cmd.append(kwargs["reasoning"])
        if kwargs.get("temperature") is not None:
            cmd.append("--temperature")
            cmd.append(str(kwargs["temperature"]))
    elif stage == 6:
        if kwargs.get("run_name"):
            cmd.append("--run-name")
            cmd.append(kwargs["run_name"])
    
    try:
        print(f"[{study_id}] Starting Stage {stage}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per stage
        )
        
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        
        if success:
            print(f"[{study_id}] ✅ Stage {stage} completed")
        else:
            print(f"[{study_id}] ❌ Stage {stage} failed")
            print(f"Error: {error[:500]}")
        
        return (study_id, stage, success, output, error)
    
    except subprocess.TimeoutExpired:
        error_msg = f"Stage {stage} timed out after 1 hour"
        print(f"[{study_id}] ⏱️  {error_msg}")
        return (study_id, stage, False, "", error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"[{study_id}] ❌ Stage {stage} exception: {error_msg}")
        return (study_id, stage, False, "", error_msg)


def run_study_pipeline(study_id: str, **kwargs) -> dict:
    """
    Run stages 3-6 sequentially for a single study.
    
    Returns:
        dict with results for each stage
    """
    results = {
        "study_id": study_id,
        "stages": {},
        "success": True,
        "start_time": datetime.now()
    }
    
    stages = [3, 4, 5, 6]
    
    for stage in stages:
        stage_start = time.time()
        study_id_result, stage_num, success, output, error = run_stage(study_id, stage, **kwargs)
        stage_elapsed = time.time() - stage_start
        
        results["stages"][stage] = {
            "success": success,
            "elapsed": stage_elapsed,
            "output": output[-1000:] if output else "",  # Keep last 1000 chars
            "error": error[-1000:] if error else ""  # Keep last 1000 chars
        }
        
        if not success:
            results["success"] = False
            print(f"[{study_id}] ⚠️  Stopping pipeline after Stage {stage} failure")
            break
    
    results["end_time"] = datetime.now()
    results["total_elapsed"] = (results["end_time"] - results["start_time"]).total_seconds()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run pipeline stages 3-6 for multiple studies")
    parser.add_argument(
        "--studies",
        nargs="+",
        help="Study IDs to run (e.g., study_003 study_004)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all studies from study_003 to study_012"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of studies to run in parallel (default: 3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/mistral-nemo",
        help="Model to use for Stage 5 (default: mistralai/mistral-nemo)"
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM API for Stage 5"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=5,
        help="Number of parallel workers for Stage 5 (default: 5)"
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        help="Number of participants for Stage 5 (default: use specification)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Name for the run directory. All studies will be saved to the same run. (default: 'default')"
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="default",
        help="Reasoning effort level for OpenRouter models (default: 'default')"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the LLM (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Determine which studies to run
    if args.all:
        study_ids = [f"study_{i:03d}" for i in range(3, 13)]  # study_003 to study_012
    elif args.studies:
        study_ids = args.studies
    else:
        print("Error: Must specify --studies or --all")
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Running Pipeline for {len(study_ids)} Studies")
    print(f"{'='*80}")
    print(f"Studies: {', '.join(study_ids)}")
    print(f"Max parallel studies: {args.max_workers}")
    print(f"Stage 5 options: model={args.model}, real_llm={args.real_llm}, num_workers={args.num_workers}, temperature={args.temperature}")
    if args.n_participants:
        print(f"  n_participants: {args.n_participants}")
    print(f"{'='*80}\n")
    
    # Prepare kwargs for stage 5
    kwargs = {
        "model": args.model,
        "real_llm": args.real_llm,
        "num_workers": args.num_workers,
        "temperature": args.temperature,
    }
    if args.n_participants:
        kwargs["n_participants"] = args.n_participants
    if args.run_name:
        kwargs["run_name"] = args.run_name
    if args.reasoning:
        kwargs["reasoning"] = args.reasoning
    
    # Run studies in parallel
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all studies
        future_to_study = {
            executor.submit(run_study_pipeline, study_id, **kwargs): study_id
            for study_id in study_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_study):
            study_id = future_to_study[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"[{study_id}] ❌ Exception: {e}")
                all_results.append({
                    "study_id": study_id,
                    "success": False,
                    "error": str(e)
                })
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Pipeline Summary")
    print(f"{'='*80}")
    
    successful = [r for r in all_results if r.get("success", False)]
    failed = [r for r in all_results if not r.get("success", False)]
    
    print(f"\n✅ Successful: {len(successful)}/{len(all_results)}")
    for r in successful:
        print(f"   {r['study_id']}: {r.get('total_elapsed', 0):.1f}s")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(all_results)}")
        for r in failed:
            print(f"   {r['study_id']}")
            # Show which stage failed
            if "stages" in r:
                for stage_num, stage_result in r["stages"].items():
                    if not stage_result.get("success", False):
                        print(f"      Stage {stage_num} failed")
                        if stage_result.get("error"):
                            print(f"      Error: {stage_result['error'][:200]}")
    
    print(f"\n⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*80}\n")
    
    # Save results to file
    output_file = Path("results") / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_elapsed": total_elapsed,
            "successful": len(successful),
            "failed": len(failed),
            "results": all_results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

