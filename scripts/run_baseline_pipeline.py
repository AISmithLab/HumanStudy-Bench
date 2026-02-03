#!/usr/bin/env python3
"""
Single end-to-end pipeline: PDF -> study files -> config -> simulation -> evaluation -> finding explanations -> summary & production tables.

Default: run Stage 5 (simulation) + Stage 6 (evaluation) for all studies/presets. Optional: Stage 1-4 from PDF, validation, final explain, summary/production.

Usage:
    # Run simulation + evaluation (default)
    python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model mistralai/mistral-nemo --presets v1_empty v2_human v3_human_plus_demo

    # From PDF: Stage 1-4 then 5-6 (PDF must be in data/studies/{study_id}/)
    python scripts/run_baseline_pipeline.py --study-id study_001 --from-pdf --real-llm

    # Optional: validation after Stage 3/4, skip summary/production
    python scripts/run_baseline_pipeline.py --study-id study_001 --from-pdf --with-validation --skip-summary --skip-production

    # Named run (results under results/runs/{run_name}/)
    python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --run-name myrun
"""

import subprocess
import argparse
import json
import sys
from pathlib import Path
import os
from datetime import datetime

# Project root setup
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Environment setup
venv_python = repo_root / ".venv" / "bin" / "python"
PYTHON_EXE = str(venv_python) if venv_python.exists() else sys.executable

def get_active_studies(data_dir):
    # Try both possible registry locations
    registry_path = Path(data_dir) / "registry.json"
    if not registry_path.exists():
        registry_path = Path(data_dir) / "studies" / "registry.json"
    if not registry_path.exists():
        return []
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return [s["id"] for s in registry.get("studies", []) if s.get("status") == "active"]

def get_config_folder_name(model, preset, reasoning="default", temperature=1.0):
    model_slug = model.replace("/", "_").replace("-", "_")
    prompt_slug = preset.replace("_", "-")
    
    # Include temperature in folder name if it's not the default
    temp_suffix = f"_temp{temperature}" if temperature != 1.0 else ""
    
    # Don't add reasoning postfix for "default", "low", or "minimal" (treat as default)
    if reasoning and reasoning != "default" and reasoning != "low" and reasoning != "minimal":
        return f"{model_slug}_{reasoning}{temp_suffix}_{prompt_slug}"
    return f"{model_slug}{temp_suffix}_{prompt_slug}"

def run_stage(cmd, label):
    print(f"\n   >>> {label}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid UTF-8 characters instead of crashing
            bufsize=1,
            env=os.environ.copy(),
            cwd=repo_root
        )
        
        # Capture all output for error reporting
        all_output = []
        for line in process.stdout:
            all_output.append(line)
            line_stripped = line.strip()
            # Condensed progress bar output
            if any(kw in line for kw in ["Trials", "Progress", "trials"]) and ("|" in line or "%" in line or "/" in line):
                print(f"\r   {line_stripped}", end='', flush=True)
            elif ">>> Run" in line or "Evaluating:" in line or "Results saved:" in line or "Processing" in line or "config folder" in line:
                print(f"\n   {line_stripped}")
            elif any(kw in line_stripped for kw in ["Stage 5", "Stage 6", "Score:", "Computing", "Loading", "Running sanity", "Saving", "📊 Progress"]):
                print(f"   {line_stripped}")
            # Show progress indicators
            elif any(kw in line_stripped for kw in ["✓", "✅", "→", "[", "]", "/"]):
                # Show progress markers
                if line_stripped.startswith(("  -", "    →", "    ✓", "  ✓", "  >>>", "  [", "    [")):
                    print(f"   {line_stripped}")
            # Show error-related lines
            elif any(kw in line_stripped.lower() for kw in ["error", "exception", "traceback", "failed", "❌", "⚠️"]):
                print(f"   {line_stripped}")
                
        process.wait()
        success = process.returncode == 0
        
        # If failed, show the last 20 lines of output to help debug
        if not success:
            print(f"\n   ❌ Process failed with exit code {process.returncode}")
            print(f"   Last 20 lines of output:")
            for line in all_output[-20:]:
                print(f"   {line.rstrip()}")
        
        return success
    except Exception as e:
        print(f"   ❌ Error running {label}: {e}")
        import traceback
        traceback.print_exc()
        return False

def _results_base_dir(args):
    """Default: results/benchmark; with --run-name: results/runs/{run_name}."""
    if getattr(args, "run_name", None):
        return Path("results/runs") / args.run_name
    return Path("results/benchmark")


def main():
    parser = argparse.ArgumentParser(description="Single end-to-end pipeline: PDF -> study -> simulation -> evaluation -> explain -> summary/production")
    parser.add_argument("--real-llm", action="store_true", help="Make actual LLM API calls")
    parser.add_argument("--model", default="mistralai/mistral-nemo", help="LLM model to use")
    parser.add_argument("--num-workers", type=int, default=16, help="Parallel workers for Stage 5")
    parser.add_argument("--repeats", type=int, default=1, help="Number of simulation repeats")
    parser.add_argument("--use-cache", action="store_true", help="Use LLM response cache")
    parser.add_argument("--continue", dest="continue_mode", action="store_true", help="Only run missing configs")
    parser.add_argument("--presets", nargs="+", default=["v1_empty", "v2_human", "v3_human_plus_demo"],
                        help="Presets from SystemPromptRegistry to run")
    parser.add_argument("--study-id", help="Run only a specific study")
    parser.add_argument("--run-name", help="Write results under results/runs/{run_name}/ instead of results/benchmark/")
    parser.add_argument("--from-pdf", action="store_true", help="Run Stage 1-4 from PDF (PDF in data/studies/{study_id}/) before 5-6")
    parser.add_argument("--generation-provider", type=str, default="gemini", choices=["gemini", "openai", "anthropic", "xai", "openrouter"],
                        help="LLM provider for Stage 1-4 when using --from-pdf (default: gemini)")
    parser.add_argument("--generation-model", type=str, default=None,
                        help="Model for Stage 1-4 when using --from-pdf (default: provider default, e.g. models/gemini-3-flash-preview)")
    parser.add_argument("--until", type=int, choices=[1, 2, 3, 4, 5, 6], default=None,
                        help="Stop after this stage (e.g. --until 4 = only run 1-4)")
    parser.add_argument("--with-validation", action="store_true", help="Run validation pipeline after Stage 3/4 (when using --from-pdf)")
    parser.add_argument("--skip-summary", action="store_true", help="Do not run generate_results_table after evaluation")
    parser.add_argument("--skip-production", action="store_true", help="Do not run generate_production_results after summary")
    parser.add_argument("--reasoning", type=str, default="default",
                        help="Reasoning effort level for OpenRouter models (default: default)")
    parser.add_argument("--enable-reasoning", action="store_true",
                        help="Force enable reasoning for OpenRouter models")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for the LLM (default: 1.0)")
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Skip Stage 5 (simulation) and only run Stage 6 (evaluation) for all configs")
    parser.add_argument("--config-filter", type=str, default=None,
                        help="Filter config folders by name pattern (e.g., 'qwen.*thinking' to evaluate only qwen thinking models)")
    
    args = parser.parse_args()

    # Get studies - try data/registry.json first, then data/studies/registry.json
    if args.study_id:
        studies = [args.study_id]
    else:
        studies = get_active_studies("data")
        if not studies:
            studies = get_active_studies("data/studies")
    
    if not studies:
        print("No active studies found. Check data/registry.json or specify --study-id")
        return

    if args.evaluation_only:
        print(f"Running evaluation only for {len(studies)} studies (all configs will be evaluated)...")
    else:
        print(f"Starting pipeline for {len(studies)} studies and {len(args.presets)} presets...")
    results = {}

    run_stages_5_6 = not (args.until and args.until <= 4)

    for study_id in studies:
        print(f"\n{'='*80}\n[STUDY: {study_id}]\n{'='*80}")
        results[study_id] = {}

        # Stage 1-4: from PDF (optional)
        if args.from_pdf or (args.until and args.until <= 4):
            for stage in [1, 2, 3, 4]:
                if args.until and stage > args.until:
                    break
                cmd = [PYTHON_EXE, "generation_pipeline/run.py", "--stage", str(stage), "--study-id", study_id,
                       "--provider", args.generation_provider]
                if getattr(args, "generation_model", None):
                    cmd += ["--model", args.generation_model]
                if not run_stage(cmd, f"Stage {stage}"):
                    print(f"   Stage {stage} failed for {study_id}")
                    results[study_id]["stage_1_4"] = False
                    break
            else:
                results[study_id]["stage_1_4"] = True
            if args.with_validation and (not args.until or args.until >= 3):
                run_stage([PYTHON_EXE, "legacy/validation_pipeline/run_validation.py", study_id], "Validation")
            if args.until and args.until <= 4:
                continue

        # Stage 5: Simulations (skip if evaluation-only mode)
        if not args.evaluation_only and run_stages_5_6:
            for preset in args.presets:
                config_name = get_config_folder_name(args.model, preset, args.reasoning, args.temperature)
                config_dir = _results_base_dir(args) / study_id / config_name
                
                # Check if we should skip in continue mode and calculate actual repeats needed
                stage5_complete = False
                actual_repeats = args.repeats
                needs_merge = False
                
                if args.continue_mode and config_dir.exists():
                    try:
                        with open(config_dir / "full_benchmark.json", 'r') as f:
                            data = json.load(f)
                            
                            # A run is truly complete only if status is "complete" OR we have enough repeats
                            is_complete = data.get('status') == 'complete'
                            current_repeats = len(data.get('all_runs_raw_results', []))
                            
                            if is_complete or current_repeats >= args.repeats:
                                print(f"   Preset {preset:15s} ... ✅ Stage 5 complete ({current_repeats} repeats)")
                                results[study_id][preset] = {"stage5": True}
                                stage5_complete = True
                            else:
                                needed = args.repeats - current_repeats
                                if data.get('status') in ['in_progress', 'starting']:
                                    print(f"   Preset {preset:15s} ... 🔄 Resuming partial run (at repeat {current_repeats+1}/{args.repeats})")
                                else:
                                    print(f"   Preset {preset:15s} ... ➕ Adding {needed} more repeats (will have {args.repeats} total)")
                                actual_repeats = needed
                                needs_merge = True # This will trigger resume logic in pipeline.py
                    except Exception as e:
                        # If we can't read the file, run normally (folder exists but no valid data)
                        print(f"   Preset {preset:15s} ... ⚠️  Found folder but couldn't read data, will run fresh")
                elif args.continue_mode and not config_dir.exists():
                    # Folder doesn't exist, so we need to run it
                    print(f"   Preset {preset:15s} ... 🆕 No existing data, will run {args.repeats} repeat(s)")

                if not stage5_complete:
                    # Show what will be run
                    if needs_merge:
                        print(f"   Preset {preset:15s} ... Running {actual_repeats} additional repeat(s) (merging with existing)")
                    else:
                        print(f"   Preset {preset:15s} ... Running {actual_repeats} repeat(s)")
                    
                    cmd = [PYTHON_EXE, "generation_pipeline/run.py", "--stage", "5", "--study-id", study_id,
                           "--model", args.model, "--system-prompt-preset", preset, "--repeats", str(actual_repeats),
                           "--reasoning", args.reasoning, "--temperature", str(args.temperature)]
                    if args.run_name: cmd.extend(["--run-name", args.run_name])
                    if args.enable_reasoning: cmd.append("--enable-reasoning")
                    if args.real_llm: cmd.append("--real-llm")
                    if args.num_workers: cmd.extend(["--num-workers", str(args.num_workers)])
                    if args.use_cache: cmd.append("--use-cache")
                    if needs_merge:  # Only use --merge-repeats if we're adding to existing runs
                        cmd.append("--merge-repeats")
                    
                    success = run_stage(cmd, f"Simulation (Preset: {preset})")
                    if success:
                        print(f"   Preset {preset:15s} ... ✅ Completed")
                    else:
                        print(f"   Preset {preset:15s} ... ❌ Failed")
                    results[study_id][preset] = {"stage5": success}
                else:
                    # Stage 5 is complete, but we still need to check Stage 6
                    results[study_id][preset] = {"stage5": True}

        # Stage 6: Evaluation (skip when --until 1..4)
        if not (run_stages_5_6 or args.evaluation_only):
            continue
        study_dir = _results_base_dir(args) / study_id
        needs_stage6 = False
        missing_evaluations = []  # Track configs missing Stage 6
        missing_stage5 = []  # Track configs missing Stage 5 entirely
        complete_configs = []  # Track configs with both Stage 5 and Stage 6
        qwen_thinking_missing = []  # Track qwen thinking configs that are missing
        qwen_thinking_complete = []  # Track qwen thinking configs that are complete
        qwen_thinking_needs_eval = []  # Track qwen thinking configs that need evaluation
        
        if study_dir.exists():
            import re
            config_filter_pattern = None
            if args.config_filter:
                config_filter_pattern = re.compile(args.config_filter)
            
            for config_dir in study_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                
                # Apply config filter if provided
                if config_filter_pattern and not config_filter_pattern.search(config_dir.name):
                    continue
                
                config_name = config_dir.name
                is_qwen_thinking = "qwen" in config_name.lower() and "thinking" in config_name.lower()
                
                has_stage5 = (config_dir / "full_benchmark.json").exists()
                has_stage6 = (
                    (config_dir / "evaluation_results.json").exists() and
                    (config_dir / "detailed_stats.csv").exists()
                )
                
                if has_stage5 and has_stage6:
                    complete_configs.append(config_name)
                    if is_qwen_thinking:
                        qwen_thinking_complete.append(config_name)
                elif has_stage5 and not has_stage6:
                    missing_evaluations.append(config_name)
                    needs_stage6 = True
                    if is_qwen_thinking:
                        qwen_thinking_needs_eval.append(config_name)
                elif not has_stage5:
                    missing_stage5.append(config_name)
                    if is_qwen_thinking:
                        qwen_thinking_missing.append(config_name)
        
        # In evaluation-only mode, always run evaluation (unless continue mode and all are done)
        if args.evaluation_only:
            # Print status report
            print(f"\n   📊 Config Status for {study_id}:")
            
            # Qwen thinking specific status
            if qwen_thinking_needs_eval or qwen_thinking_missing or qwen_thinking_complete:
                print(f"      🤔 Qwen Thinking Models:")
                if qwen_thinking_needs_eval:
                    print(f"         ⚠️  Missing evaluation ({len(qwen_thinking_needs_eval)}): {', '.join(qwen_thinking_needs_eval)}")
                if qwen_thinking_missing:
                    print(f"         ❌ Missing Stage 5 data ({len(qwen_thinking_missing)}): {', '.join(qwen_thinking_missing)}")
                if qwen_thinking_complete:
                    print(f"         ✅ Complete ({len(qwen_thinking_complete)}): {', '.join(qwen_thinking_complete)}")
                if not qwen_thinking_needs_eval and not qwen_thinking_missing and not qwen_thinking_complete:
                    print(f"         ℹ️  No qwen thinking configs found in this study")
            
            # General status
            if missing_evaluations:
                non_qwen_missing = [c for c in missing_evaluations if c not in qwen_thinking_needs_eval]
                if non_qwen_missing:
                    print(f"      ⚠️  Missing evaluations ({len(non_qwen_missing)}): {', '.join(non_qwen_missing[:5])}")
                    if len(non_qwen_missing) > 5:
                        print(f"         ... and {len(non_qwen_missing) - 5} more")
            if missing_stage5:
                non_qwen_stage5 = [c for c in missing_stage5 if c not in qwen_thinking_missing]
                if non_qwen_stage5:
                    print(f"      ⚠️  Missing Stage 5 data ({len(non_qwen_stage5)}): {', '.join(non_qwen_stage5[:5])}")
                    if len(non_qwen_stage5) > 5:
                        print(f"         ... and {len(non_qwen_stage5) - 5} more")
            if complete_configs:
                non_qwen_complete = [c for c in complete_configs if c not in qwen_thinking_complete]
                if non_qwen_complete:
                    print(f"      ✅ Complete ({len(non_qwen_complete)} configs)")
            
            if needs_stage6 or not args.continue_mode:
                print(f"\n   --- Evaluation ---")
                if missing_evaluations:
                    print(f"   Evaluating {len(missing_evaluations)} config(s) missing evaluation...")
                eval_cmd = [PYTHON_EXE, "generation_pipeline/run.py", "--stage", "6", "--study-id", study_id, "--skip-generation"]
                if args.run_name: eval_cmd.extend(["--run-name", args.run_name])
                success = run_stage(eval_cmd, "Evaluation")
                # In evaluation-only mode, we don't track by preset since we evaluate all configs
                results[study_id]["evaluation"] = success
                results[study_id]["missing_evaluations"] = missing_evaluations
                results[study_id]["missing_stage5"] = missing_stage5
                results[study_id]["complete_configs"] = complete_configs
                results[study_id]["qwen_thinking"] = {
                    "needs_eval": qwen_thinking_needs_eval,
                    "missing_stage5": qwen_thinking_missing,
                    "complete": qwen_thinking_complete
                }
            else:
                print(f"\n   --- Evaluation ---")
                print(f"   ✅ All configs already evaluated")
                results[study_id]["evaluation"] = True
                results[study_id]["missing_evaluations"] = []
                results[study_id]["missing_stage5"] = missing_stage5
                results[study_id]["complete_configs"] = complete_configs
                results[study_id]["qwen_thinking"] = {
                    "needs_eval": [],
                    "missing_stage5": qwen_thinking_missing,
                    "complete": qwen_thinking_complete
                }
        else:
            # Normal mode: check if evaluation is needed
            if needs_stage6 or not args.continue_mode:
                print(f"\n   --- Evaluation ---")
                eval_cmd = [PYTHON_EXE, "generation_pipeline/run.py", "--stage", "6", "--study-id", study_id, "--skip-generation"]
                if args.run_name: eval_cmd.extend(["--run-name", args.run_name])
                success = run_stage(eval_cmd, "Evaluation")
                
                # Check each preset's evaluation status individually
                # (Stage 6 may have partially completed, so check actual files)
                for preset in args.presets:
                    if preset not in results[study_id]:
                        results[study_id][preset] = {}
                    # Check if this preset's config folder has evaluation results
                    config_name = get_config_folder_name(args.model, preset, args.reasoning, args.temperature)
                    config_dir = _results_base_dir(args) / study_id / config_name
                    has_stage6 = (
                        (config_dir / "evaluation_results.json").exists() and
                        (config_dir / "detailed_stats.csv").exists()
                    )
                    # Use actual file check if Stage 6 succeeded, or if files exist despite failure
                    results[study_id][preset]["stage6"] = has_stage6
            else:
                print(f"\n   --- Evaluation ---")
                print(f"   ✅ All configs already evaluated")
                for preset in args.presets:
                    if preset not in results[study_id]:
                        results[study_id][preset] = {}
                    results[study_id][preset]["stage6"] = True

        # Stage final: finding explanations for this study
        final_cmd = [PYTHON_EXE, "generation_pipeline/run.py", "--stage", "final", "--study-id", study_id]
        if args.run_name: final_cmd.extend(["--run-name", args.run_name])
        run_stage(final_cmd, "Finding explanations")

    # Summary + production (unless skipped or we only ran 1-4)
    results_base = _results_base_dir(args)
    if run_stages_5_6 and not getattr(args, "skip_summary", False):
        print(f"\n   >>> Summary (benchmark_summary)")
        summary_cmd = [PYTHON_EXE, "scripts/generate_results_table.py", "--format", "all", "--results-dir", str(results_base), "--output", str(results_base / "benchmark_summary")]
        run_stage(summary_cmd, "Summary")
    if run_stages_5_6 and not getattr(args, "skip_production", False):
        summary_json = results_base / "benchmark_summary.json"
        if not summary_json.exists():
            summary_json = Path("results/benchmark_summary.json")
        print(f"\n   >>> Production tables")
        prod_cmd = [PYTHON_EXE, "scripts/generate_production_results.py", "--summary-json", str(summary_json), "--results-dir", str(results_base), "--output-latex", str(results_base / "production_tables.tex")]
        run_stage(prod_cmd, "Production tables")

    # Summary
    print(f"\n{'='*80}\nFINAL SUMMARY\n{'='*80}")
    if args.evaluation_only:
        # Evaluation-only mode: detailed summary with missing data info
        print(f"\n{'Study ID':<15} | {'Evaluation':<12} | {'Missing Eval':<15} | {'Missing Stage5':<15}")
        print("-" * 80)
        for study_id in sorted(studies):
            eval_status = "✅" if results[study_id].get("evaluation") else "❌"
            missing_eval = results[study_id].get("missing_evaluations", [])
            missing_s5 = results[study_id].get("missing_stage5", [])
            missing_eval_str = f"{len(missing_eval)} config(s)" if missing_eval else "None"
            missing_s5_str = f"{len(missing_s5)} config(s)" if missing_s5 else "None"
            print(f"{study_id:<15} | {eval_status:<12} | {missing_eval_str:<15} | {missing_s5_str:<15}")
        
        # Detailed breakdown of missing evaluations
        all_missing_eval = []
        all_missing_s5 = []
        for study_id in sorted(studies):
            missing_eval = results[study_id].get("missing_evaluations", [])
            missing_s5 = results[study_id].get("missing_stage5", [])
            if missing_eval:
                all_missing_eval.extend([f"{study_id}/{config}" for config in missing_eval])
            if missing_s5:
                all_missing_s5.extend([f"{study_id}/{config}" for config in missing_s5])
        
        if all_missing_eval:
            print(f"\n⚠️  Configs Missing Evaluation ({len(all_missing_eval)} total):")
            for config_path in sorted(all_missing_eval)[:20]:  # Show first 20
                print(f"   - {config_path}")
            if len(all_missing_eval) > 20:
                print(f"   ... and {len(all_missing_eval) - 20} more")
        
        if all_missing_s5:
            print(f"\n⚠️  Configs Missing Stage 5 Data ({len(all_missing_s5)} total):")
            for config_path in sorted(all_missing_s5)[:20]:  # Show first 20
                print(f"   - {config_path}")
            if len(all_missing_s5) > 20:
                print(f"   ... and {len(all_missing_s5) - 20} more")
        
        # Qwen thinking summary
        all_qwen_thinking_needs_eval = []
        all_qwen_thinking_missing = []
        all_qwen_thinking_complete = []
        for study_id in sorted(studies):
            qwen_info = results[study_id].get("qwen_thinking", {})
            needs_eval = qwen_info.get("needs_eval", [])
            missing = qwen_info.get("missing_stage5", [])
            complete = qwen_info.get("complete", [])
            if needs_eval:
                all_qwen_thinking_needs_eval.extend([f"{study_id}/{config}" for config in needs_eval])
            if missing:
                all_qwen_thinking_missing.extend([f"{study_id}/{config}" for config in missing])
            if complete:
                all_qwen_thinking_complete.extend([f"{study_id}/{config}" for config in complete])
        
        if all_qwen_thinking_needs_eval or all_qwen_thinking_missing:
            print(f"\n🤔 Qwen Thinking Models Summary:")
            if all_qwen_thinking_needs_eval:
                print(f"   ⚠️  Missing Evaluation ({len(all_qwen_thinking_needs_eval)}):")
                for config_path in sorted(all_qwen_thinking_needs_eval):
                    print(f"      - {config_path}")
            if all_qwen_thinking_missing:
                print(f"   ❌ Missing Stage 5 Data ({len(all_qwen_thinking_missing)}):")
                for config_path in sorted(all_qwen_thinking_missing):
                    print(f"      - {config_path}")
            if all_qwen_thinking_complete:
                print(f"   ✅ Complete ({len(all_qwen_thinking_complete)} configs)")
            # Check which studies have no qwen thinking at all
            studies_with_qwen = set()
            for config_path in all_qwen_thinking_needs_eval + all_qwen_thinking_missing + all_qwen_thinking_complete:
                study_id = config_path.split("/")[0]
                studies_with_qwen.add(study_id)
            studies_without_qwen = set(studies) - studies_with_qwen
            if studies_without_qwen:
                print(f"   ℹ️  Studies with no qwen thinking configs: {', '.join(sorted(studies_without_qwen))}")
    else:
        # Normal mode: show both Stage 5 and Stage 6 for each preset
        header = f"{'Study ID':<15} | " + " | ".join([f"{p[:6]:<6}" for p in args.presets])
        print(header)
        print("-" * len(header))
        for study_id in sorted(studies):
            row = f"{study_id:<15} | "
            icons = []
            for p in args.presets:
                s5 = "✅" if results[study_id].get(p, {}).get("stage5") else "❌"
                s6 = "✅" if results[study_id].get(p, {}).get("stage6") else "❌"
                icons.append(f"{s5}{s6}")
            print(row + " | ".join(icons))

if __name__ == "__main__":
    main()
