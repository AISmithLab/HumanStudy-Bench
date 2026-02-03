"""
CLI for Generation Pipeline

Usage:
    python generation_pipeline/run.py --stage 1
    python generation_pipeline/run.py --stage 2
    python generation_pipeline/run.py --stage 1 --refine
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_pdf_in_current_dir(study_id: str = None) -> Path:
    """Find PDF file in current directory or study directory"""
    # If study_id is provided, try to find PDF in study directory first
    if study_id:
        study_dir = Path("data/studies") / study_id
        if study_dir.exists():
            pdf_files = list(study_dir.glob("*.pdf"))
            if pdf_files:
                return pdf_files[0]
    
    # Fall back to current directory
    current_dir = Path.cwd()
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if len(pdf_files) == 0:
        if study_id:
            raise FileNotFoundError(f"No PDF file found in current directory or data/studies/{study_id}/")
        else:
            raise FileNotFoundError("No PDF file found in current directory")
    elif len(pdf_files) > 1:
        raise ValueError(f"Multiple PDF files found: {[f.name for f in pdf_files]}")
    
    return pdf_files[0]


def find_latest_stage_file(stage: int, output_dir: Path, paper_id: str = None) -> Path:
    """Find latest stage file, optionally filtered by paper_id"""
    if paper_id:
        pattern = f"{paper_id}_stage{stage}_*.json"
    else:
        pattern = f"*_stage{stage}_*.json"
    
    json_files = list(output_dir.glob(pattern))
    
    if not json_files:
        if paper_id:
            raise FileNotFoundError(f"No stage {stage} JSON file found for paper '{paper_id}' in {output_dir}")
        else:
            raise FileNotFoundError(f"No stage {stage} JSON file found in {output_dir}")
    
    # Return most recent
    return max(json_files, key=lambda p: p.stat().st_mtime)


def infer_paper_id_from_study(study_id: str) -> str:
    """Try to infer paper_id from the PDF in the study directory"""
    study_dir = Path("data/studies") / study_id
    if study_dir.exists():
        pdf_files = list(study_dir.glob("*.pdf"))
        if pdf_files:
            return pdf_files[0].stem.replace(' ', '_').replace('-', '_').lower()
    return None


def main():
    parser = argparse.ArgumentParser(description="Semi-Manual Study Generation Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["1", "2", "3", "4", "5", "6", "final"],
        help="Stage to run (1=Filter, 2=Extraction, 3=JSON/Materials, 4=Config, 5=Simulation, 6=Evaluator, final=Finding explanations)"
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine current stage (re-run with existing review)"
    )
    parser.add_argument(
        "--study-id",
        type=str,
        help="Study ID (e.g., study_001). Used to find PDF in data/studies/{study_id}/ for stage 1&2, and for generating files in stage 3"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generation_pipeline/outputs"),
        help="Output directory for review files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,  # Will be set per-stage
        help="Model to use (stages 1-4: default gemini; stage 5: default mistralai/mistral-nemo)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "anthropic", "xai", "openrouter"],
        help="LLM provider for stages 1-4 (default: gemini)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (else from env: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (e.g. for OpenRouter: https://openrouter.ai/api/v1)"
    )
    parser.add_argument(
        "--file",
        type=str,
        choices=["metadata", "specification", "ground_truth", "materials"],
        help="Specific file to generate in stage 3 (optional)"
    )
    parser.add_argument(
        "--regeneration-instructions",
        type=str,
        help="Path to stage2_regeneration_instructions.json file from validation pipeline (for Stage 2)"
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM API (for Stage 5)"
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        help="Number of participants (for Stage 5)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs (for Stage 5). If results exist, new repeats will be added to existing ones."
    )
    parser.add_argument(
        "--merge-repeats",
        action="store_true",
        help="Merge new repeats with existing results instead of overwriting (for Stage 5)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for the run directory (default: timestamp). Use same name to append multiple studies to one run."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers (for Stage 5)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable result caching (for Stage 5)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="results/cache",
        help="Cache directory (for Stage 5)"
    )
    parser.add_argument(
        "--profiles-json",
        type=str,
        help="Path to participant profiles JSON file (for Stage 5)"
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to system prompt override file (for Stage 5)"
    )
    parser.add_argument(
        "--system-prompt-preset",
        type=str,
        default="v3_human_plus_demo",
        help="System prompt preset (for Stage 5). Available: v1_empty, v2_human, v3_human_plus_demo, or any custom method in src/agents/custom_methods/"
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="default",
        # Removed choices to allow numeric string for max_tokens
        help="Reasoning effort level for OpenRouter models (for Stage 5, default: default)"
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Force enable reasoning for OpenRouter models (for Stage 5)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (for Stage 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the LLM (for Stage 5, default: 1.0)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip evaluator generation and only run evaluation (for Stage 6)"
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        help="Specific config folder to evaluate (for Stage 6, e.g., 'mistralai_mistral_small_creative_temp0.7_v2-human')"
    )
    
    args = parser.parse_args()
    stage = int(args.stage) if args.stage.isdigit() else args.stage

    # Stage "final" does not need the full pipeline (no Gemini/PDF deps)
    if stage == "final":
        try:
            if not args.study_id:
                raise ValueError("--study-id is required for stage final")
            from src.evaluation.finding_explainer import run_finding_explanations
            import json
            if args.run_name:
                results_dir = Path("results/runs") / args.run_name
            else:
                results_dir = Path("results/benchmark")
            study_data_dir = Path("data/studies")
            out = run_finding_explanations(
                args.study_id,
                results_dir=results_dir,
                study_data_dir=study_data_dir,
                config_folder=args.config_folder,
            )
            if out.get("error"):
                print(f"Warning: {out['error']}", file=sys.stderr)
            out_dir = results_dir / args.study_id
            if args.config_folder:
                out_dir = out_dir / args.config_folder
            else:
                study_path = results_dir / args.study_id
                if study_path.exists():
                    configs = [d for d in study_path.iterdir() if d.is_dir()]
                    if configs:
                        out_dir = configs[0]
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "finding_explanations.json"
            md_path = out_dir / "finding_explanations.md"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            lines = [f"# Finding explanations: {args.study_id}", ""]
            for f in out.get("findings", []):
                lines.append(f"## {f.get('finding_id', '')}")
                lines.append("")
                lines.append(f.get("explanation", ""))
                lines.append("")
            md_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"✓ Stage final complete!")
            print(f"  {json_path}")
            print(f"  {md_path}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Initialize pipeline for stages 1-6
    from generation_pipeline.pipeline import GenerationPipeline
    stage_1_4_model = args.model or "models/gemini-3-flash-preview"
    pipeline = GenerationPipeline(
        provider=args.provider,
        model=stage_1_4_model,
        api_key=args.api_key,
        api_base=args.api_base,
        output_dir=args.output_dir
    )

    try:
        # Determine paper_id if study_id is provided
        paper_id = None
        if args.study_id:
            paper_id = infer_paper_id_from_study(args.study_id)

        if stage == 1:
            if args.refine:
                # Find latest stage1 review file
                review_file = find_latest_stage_file(1, args.output_dir, paper_id=paper_id)
                review_file = review_file.with_suffix('.md')
                
                if not review_file.exists():
                    raise FileNotFoundError(f"Review file not found: {review_file}")
                
                print(f"Refining Stage 1 based on {review_file.name}")
                review_status = pipeline.check_stage1_review(review_file)
                print(f"Review status: {review_status['action']}")
                
                # Re-run stage1
                pdf_path = find_pdf_in_current_dir(args.study_id)
                pipeline.run_stage1(pdf_path)
            else:
                # Run stage1
                pdf_path = find_pdf_in_current_dir(args.study_id)
                pipeline.run_stage1(pdf_path)
        
        elif stage == 2:
            # Load regeneration instructions if provided
            regeneration_instructions_path = None
            if args.regeneration_instructions:
                regeneration_instructions_path = Path(args.regeneration_instructions)
                if not regeneration_instructions_path.exists():
                    raise FileNotFoundError(f"Regeneration instructions file not found: {regeneration_instructions_path}")
            
            if args.refine:
                # Find latest stage2 review file
                review_file = find_latest_stage_file(2, args.output_dir, paper_id=paper_id)
                review_file = review_file.with_suffix('.md')
                
                if not review_file.exists():
                    raise FileNotFoundError(f"Review file not found: {review_file}")
                
                print(f"Refining Stage 2 based on {review_file.name}")
                review_status = pipeline.check_stage2_review(review_file)
                print(f"Review status: {review_status['action']}")
                
                # Re-run stage2
                stage1_json = find_latest_stage_file(1, args.output_dir, paper_id=paper_id)
                pdf_path = find_pdf_in_current_dir(args.study_id)
                pipeline.run_stage2(stage1_json, pdf_path, regeneration_instructions_path=regeneration_instructions_path)
            else:
                # Run stage2
                stage1_json = find_latest_stage_file(1, args.output_dir, paper_id=paper_id)
                pdf_path = find_pdf_in_current_dir(args.study_id)
                pipeline.run_stage2(stage1_json, pdf_path, regeneration_instructions_path=regeneration_instructions_path)
        
        elif stage == 3:
            # Generate study JSON and materials
            stage2_json = find_latest_stage_file(2, args.output_dir, paper_id=paper_id)
            
            # Determine study_id
            if args.study_id:
                study_id = args.study_id
            else:
                # Infer from stage2 JSON
                import json
                stage2_result = json.loads(stage2_json.read_text(encoding='utf-8'))
                paper_id_in_json = stage2_result.get('paper_id', 'unknown')
                study_id = f"study_{paper_id_in_json.split('_')[-1]}" if '_' in paper_id_in_json else f"study_{paper_id_in_json}"
                if not any(c.isdigit() for c in study_id):
                    study_id = "study_001"
            
            print(f"Generating study JSON and materials for {study_id} using {stage2_json.name}")
            result = pipeline.generate_study(stage2_json, study_id, file_type=args.file)
            print(f"\n✓ Stage 3 complete!")
            print(f"  Files saved to: {result['study_dir']}")
            
        elif stage == 4:
            # Generate StudyConfig Agent
            stage2_json = find_latest_stage_file(2, args.output_dir, paper_id=paper_id)
            
            # Determine study_id
            if args.study_id:
                study_id = args.study_id
            else:
                # Infer from stage2 JSON
                import json
                stage2_result = json.loads(stage2_json.read_text(encoding='utf-8'))
                paper_id_in_json = stage2_result.get('paper_id', 'unknown')
                study_id = f"study_{paper_id_in_json.split('_')[-1]}" if '_' in paper_id_in_json else f"study_{paper_id_in_json}"
                if not any(c.isdigit() for c in study_id):
                    study_id = "study_001"
            
            print(f"Generating StudyConfig Agent for {study_id} using {stage2_json.name}")
            config_path = pipeline.run_stage4(stage2_json, study_id)
            print(f"\n✓ Stage 4 complete!")
            print(f"  Agent code saved to: {config_path}")
        
        elif stage == 5:
            # Run Simulation
            if not args.study_id:
                raise ValueError("--study-id is required for Stage 5")
            
            print(f"Running Simulation for {args.study_id}")
            # Use mistral-nemo as default for Stage 5 (agent testing)
            llm_model = args.model or "mistralai/mistral-nemo"
            result_path = pipeline.run_stage5(
                args.study_id,
                use_real_llm=args.real_llm,
                model=llm_model,
                n_participants=args.n_participants,
                random_seed=args.random_seed,
                num_workers=args.num_workers,
                use_cache=args.use_cache,
                cache_dir=args.cache_dir,
                profiles_json=args.profiles_json,
                system_prompt_file=args.system_prompt_file,
                system_prompt_preset=args.system_prompt_preset,
                repeats=args.repeats,
                run_name=args.run_name,
                merge_existing_repeats=args.merge_repeats,
                reasoning=args.reasoning,
                enable_reasoning=args.enable_reasoning,
                temperature=args.temperature
            )
            print(f"\n✓ Stage 5 complete!")
            print(f"  Results saved to: {result_path}")
        
        elif stage == 6:
            # Generate Evaluator and Compute Scores
            if not args.study_id:
                raise ValueError("--study-id is required for Stage 6")
            
            if args.skip_generation:
                print(f"Re-running Evaluation for {args.study_id} (skipping generation)")
            else:
                print(f"Generating Evaluator and Computing Scores for {args.study_id}")
            evaluator_path = pipeline.run_stage6(
                args.study_id, 
                skip_generation=args.skip_generation, 
                run_name=args.run_name,
                config_folder=args.config_folder
            )
            print(f"\n✓ Stage 6 complete!")
            print(f"  Evaluator code saved to: {evaluator_path}")

        # stage "final" handled above (early return)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

