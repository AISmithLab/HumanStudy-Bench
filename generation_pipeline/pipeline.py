"""
Generation Pipeline Orchestrator

Coordinates filters, extractors, and generators to create study configurations.
"""

import json
import sys
import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation_pipeline.filters.replicability_filter import ReplicabilityFilter
from generation_pipeline.extractors.study_data_extractor import StudyDataExtractor
from generation_pipeline.generators.config_generator import ConfigGenerator
from generation_pipeline.utils.json_generator import JSONGenerator
from generation_pipeline.utils.review_parser import ReviewParser
from src.llm.factory import get_client
from generation_pipeline.utils.output_formatter import OutputFormatter
from src.generators.evaluator_generator import EvaluatorGenerator
from src.core.benchmark import HumanStudyBench
from src.agents.llm_participant_agent import ParticipantPool
from src.agents.prompt_builder import get_prompt_builder
from src.core.study_config import get_study_config
# Import all study configurations to register them
import src.studies
import time
import json


class GenerationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "models/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline.

        Args:
            provider: One of gemini, openai, anthropic, xai, openrouter
            model: Model name for the provider
            api_key: Optional API key (else from env)
            api_base: Optional API base URL (for openrouter/openai-compatible)
            output_dir: Directory for output files
        """
        self.provider = (provider or "gemini").lower()
        self.model = model
        self.client = get_client(provider=self.provider, model=model, api_key=api_key, api_base=api_base)
        self.output_dir = Path(output_dir) if output_dir else Path("generation_pipeline/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (all use unified client + PDF text)
        self.filter = ReplicabilityFilter(self.client)
        self.extractor = StudyDataExtractor(self.client)
        self.config_generator = ConfigGenerator(provider=self.provider, model=model, api_key=api_key, api_base=api_base)
        self.json_generator = JSONGenerator(provider=self.provider, model=model, api_key=api_key, api_base=api_base)
    
    def run_stage1(self, pdf_path: Path) -> Tuple[Path, Path, Dict[str, Any]]:
        """
        Run stage 1 (filter).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (markdown_path, json_path, result_dict)
        """
        print(f"Running Stage 1: Replicability Filter for {pdf_path.name}")
        
        # Run filter
        result = self.filter.process(pdf_path)
        
        # Generate paper_id from filename
        paper_id = pdf_path.stem.replace(' ', '_').replace('-', '_').lower()
        result['paper_id'] = paper_id
        
        # Format as markdown
        md_content = OutputFormatter.format_stage1_review(result)
        
        # Save files
        md_path = self.output_dir / f"{paper_id}_stage1_filter.md"
        json_path = self.output_dir / f"{paper_id}_stage1_filter.json"
        
        md_path.write_text(md_content, encoding='utf-8')
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        
        print(f"Stage 1 complete. Review file: {md_path}")
        print(f"JSON file: {json_path}")
        
        return md_path, json_path, result
    
    def check_stage1_review(self, review_file: Path) -> Dict[str, Any]:
        """
        Check stage1 review status.
        
        Args:
            review_file: Path to review markdown file
            
        Returns:
            Dictionary with review status and action
        """
        parsed = ReviewParser.parse(review_file)
        action = ReviewParser.get_action(parsed['review_status'])
        
        return {
            "action": action,
            "review_status": parsed['review_status'],
            "comments": parsed['comments'],
            "checklists": parsed['checklists']
        }
    
    def run_stage2(self, stage1_json_path: Path, pdf_path: Path, regeneration_instructions_path: Optional[Path] = None) -> Tuple[Path, Path, Dict[str, Any]]:
        """
        Run stage 2 (extraction).
        
        Args:
            stage1_json_path: Path to stage1 JSON result
            pdf_path: Path to PDF file
            regeneration_instructions_path: Optional path to validation feedback JSON file
            
        Returns:
            Tuple of (markdown_path, json_path, result_dict)
        """
        print(f"Running Stage 2: Study & Data Extraction")
        
        # Load stage1 results
        if not stage1_json_path.exists():
            raise FileNotFoundError(f"Stage1 JSON file not found: {stage1_json_path}")
        
        stage1_result = json.loads(stage1_json_path.read_text(encoding='utf-8'))
        
        if not isinstance(stage1_result, dict):
            raise ValueError(f"Stage1 result is not a dictionary: {type(stage1_result)}")
        
        # Load regeneration instructions if provided
        regeneration_instructions = None
        if regeneration_instructions_path:
            if not regeneration_instructions_path.exists():
                print(f"Warning: Regeneration instructions file not found: {regeneration_instructions_path}")
            else:
                regeneration_instructions = json.loads(regeneration_instructions_path.read_text(encoding='utf-8'))
                print(f"Using validation feedback from: {regeneration_instructions_path.name}")
        
        # Run extractor
        result = self.extractor.process(stage1_result, pdf_path, regeneration_instructions=regeneration_instructions)
        
        # Format as markdown
        md_content = OutputFormatter.format_stage2_review(result)
        
        # Save files
        paper_id = stage1_result.get('paper_id', 'unknown')
        md_path = self.output_dir / f"{paper_id}_stage2_extraction.md"
        json_path = self.output_dir / f"{paper_id}_stage2_extraction.json"
        
        md_path.write_text(md_content, encoding='utf-8')
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        
        print(f"Stage 2 complete. Review file: {md_path}")
        print(f"JSON file: {json_path}")
        
        return md_path, json_path, result
    
    def check_stage2_review(self, review_file: Path) -> Dict[str, Any]:
        """Check stage2 review status"""
        parsed = ReviewParser.parse(review_file)
        action = ReviewParser.get_action(parsed['review_status'])
        
        return {
            "action": action,
            "review_status": parsed['review_status'],
            "comments": parsed['comments'],
            "checklists": parsed['checklists']
        }
    
    def generate_study(
        self,
        stage2_json_path: Path,
        study_id: str,
        study_dir: Optional[Path] = None,
        file_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run stage 3 (JSON files + materials).
        
        Args:
            stage2_json_path: Path to stage2 JSON result
            study_id: Study ID (e.g., "study_005")
            study_dir: Directory to save study files
            file_type: Optional file type to generate (metadata|specification|ground_truth|materials)
            
        Returns:
            Dictionary with paths to generated files
        """
        print(f"Running Stage 3: Generating {'all' if not file_type else file_type} JSON and materials for {study_id}")
        
        # Load stage2 results
        extraction_result = json.loads(stage2_json_path.read_text(encoding='utf-8'))
        
        # Determine study directory
        if study_dir is None:
            study_dir = Path("data/studies") / study_id
        study_dir = Path(study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Find PDF for context
        pdf_path = None
        pdf_files = list(study_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = pdf_files[0]
        
        results = {"study_dir": study_dir}
        
        # Generate JSON files
        if file_type in [None, "metadata"]:
            metadata = self.json_generator.generate_metadata(extraction_result, study_id, pdf_path=pdf_path)
            metadata_path = study_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
            results["metadata"] = metadata_path
            print(f"  - Generated: {metadata_path}")
            
        if file_type in [None, "specification"]:
            specification = self.json_generator.generate_specification(extraction_result, study_id)
            spec_path = study_dir / "specification.json"
            spec_path.write_text(json.dumps(specification, indent=2, ensure_ascii=False), encoding='utf-8')
            results["specification"] = spec_path
            print(f"  - Generated: {spec_path}")
            
        if file_type in [None, "ground_truth"]:
            ground_truth = self.json_generator.generate_ground_truth(extraction_result, study_id)
            gt_path = study_dir / "ground_truth.json"
            gt_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding='utf-8')
            results["ground_truth"] = gt_path
            print(f"  - Generated: {gt_path}")
        
        # Generate materials files using LLM-based dynamic generation
        if file_type in [None, "materials"]:
            print(f"\nGenerating materials files using LLM...")
            material_files = self.json_generator.generate_materials(
                extraction_result,
                study_dir,
                pdf_path=pdf_path
            )
            results["materials"] = material_files
            print(f"  - Generated {len(material_files)} material files")
            
            # Inject gt_keys into materials (requires ground_truth.json)
            # Only inject if ground_truth was generated or already exists
            gt_path = study_dir / "ground_truth.json"
            if gt_path.exists() or file_type == "materials":
                # If ground_truth wasn't generated this run, try to load existing
                if not gt_path.exists() and file_type == "materials":
                    print(f"  - Warning: ground_truth.json not found. gt_key injection will be skipped.")
                    print(f"    Run Stage 3 with '--file ground_truth' first, or run without --file to generate all files.")
                else:
                    print(f"\nInjecting gt_keys into materials...")
                    try:
                        coverage_stats = self.json_generator.inject_gt_keys_into_materials(
                            study_dir,
                            material_files
                        )
                        results["gt_key_coverage"] = coverage_stats
                        print(f"  ✓ gt_key injection complete")
                    except ValueError as e:
                        # In strict mode, this raises an error
                        print(f"  ❌ {e}")
                        raise
                    except Exception as e:
                        print(f"  ⚠️  Error during gt_key injection: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"\nStage 3 complete.")
        return results

    def run_stage4(
        self,
        stage2_json_path: Path,
        study_id: str,
        study_dir: Optional[Path] = None
    ) -> Path:
        """
        Run stage 4 (Config/Agent generation).
        
        Args:
            stage2_json_path: Path to stage2 JSON result
            study_id: Study ID
            study_dir: Study directory
            
        Returns:
            Path to generated config file
        """
        print(f"Running Stage 4: Generating StudyConfig Agent for {study_id}")
        
        # Load stage2 results
        extraction_result = json.loads(stage2_json_path.read_text(encoding='utf-8'))
        
        # Determine study directory
        if study_dir is None:
            study_dir = Path("data/studies") / study_id
        
        # Find PDF for context
        pdf_path = None
        pdf_files = list(study_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = pdf_files[0]
            
        # Generate study config class
        config_path = Path("src/studies") / f"{study_id}_config.py"
        self.config_generator.generate(
            extraction_result,
            study_id,
            config_path,
            pdf_path=pdf_path,
            study_dir=study_dir
        )
        
        # NEW: Automatically dump prompts for verification
        try:
            print(f"  - Dumping sample prompts for verification...")
            from src.core.study_config import get_study_config
            import importlib
            import sys
            
            # Ensure src is in path
            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
            
            # Force reload of the newly generated module
            module_name = f"src.studies.{study_id}_config"
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
                
            spec = extraction_result.get('specification', {}) or {}
            if not spec:
                # Load from file if not in extraction_result
                spec_file = study_dir / "specification.json"
                if spec_file.exists():
                    spec = json.loads(spec_file.read_text(encoding='utf-8'))
            
            config = get_study_config(study_id, study_dir, spec)
            
            prompt_dump_dir = self.output_dir / "prompts" / study_id
            prompt_dump_dir.mkdir(parents=True, exist_ok=True)
            
            if hasattr(config, 'dump_prompts'):
                config.dump_prompts(prompt_dump_dir)
                print(f"  - Sample prompts dumped to: {prompt_dump_dir}")
            else:
                print(f"  - Warning: Generated config for {study_id} missing dump_prompts method")
        except Exception as e:
            print(f"  - Warning: Failed to dump sample prompts: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"\nStage 4 complete:")
        print(f"  - Generated: {config_path}")
        print(f"\nNext steps:")
        print(f"  1. Review the generated config")
        print(f"  2. Run Stage 5 (Simulation): python generation_pipeline/run.py --stage 5 --study-id {study_id}")
        print(f"  3. Run Stage 6 (Evaluation): python generation_pipeline/run.py --stage 6 --study-id {study_id}")
        
        return config_path
    
    def run_stage5(
        self,
        study_id: str,
        use_real_llm: bool = False,
        model: str = "mistralai/mistral-nemo",
        n_participants: Optional[int] = None,
        random_seed: int = 42,
        num_workers: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = "results/cache",
        profiles_json: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        system_prompt_preset: str = "v3_human_plus_demo",
        repeats: int = 1,
        run_name: Optional[str] = None,
        merge_existing_repeats: bool = False,
        reasoning: str = "default",
        enable_reasoning: bool = False,
        temperature: float = 1.0
    ) -> Path:
        """
        Run stage 5 (Simulation - run agents and collect raw responses).
        
        Args:
            study_id: Study ID (e.g., "study_001")
            use_real_llm: Whether to use real LLM API
            model: Model name
            n_participants: Number of participants (None = use specification)
            random_seed: Random seed
            num_workers: Number of parallel workers
            use_cache: Whether to use cache
            cache_dir: Cache directory
            profiles_json: Optional path to profiles JSON file
            system_prompt_file: Optional path to system prompt override file
            system_prompt_preset: System prompt preset
            repeats: Number of repeated runs
            run_name: Name for the run directory
            merge_existing_repeats: Whether to merge new repeats with existing results
            reasoning: Reasoning effort level
            enable_reasoning: Force enable reasoning for OpenRouter models
            
        Returns:
            Path to saved benchmark results
        """
        print(f"Running Stage 5: Simulation for {study_id}")
        
        # Load benchmark and study
        benchmark = HumanStudyBench("data")
        study = benchmark.load_study(study_id)
        print(f"Study: {study.metadata['title']}")
        
        # Get study config
        study_path = study.materials_path.parent
        study_config = get_study_config(study_id, study_path, study.specification)
        
        # Create prompt builder
        if hasattr(study_config, 'get_prompt_builder'):
            builder = study_config.get_prompt_builder()
        else:
            builder = get_prompt_builder(study_id)
        instructions = builder.get_instructions()
        
        # Determine n_participants and sub-study distribution
        by_sub_study_spec = study.specification.get("participants", {}).get("by_sub_study", {})
        
        # Create trials based on distribution
        if n_participants is not None:
            # When n_participants is specified, create that many trials total (1 participant per trial)
            # This ensures paper-faithful: each trial gets exactly 1 participant
            print(f"Creating {n_participants} trials (1 participant per trial)")
            trials = study_config.create_trials(n_trials=n_participants)
        elif by_sub_study_spec:
            # Paper-faithful: use distribution from specification
            trials = study_config.create_trials(n_trials=None)
        else:
            # Fallback: use default N
            n_def = study.specification.get('participants', {}).get('n') or 30
            print(f"Using default N={n_def}")
            trials = study_config.create_trials(n_trials=n_def)
        
        print(f"Trials: {len(trials)}", flush=True)
        print(f"⏳ Setting up participant pool and starting API calls...", flush=True)
        
        # Load profiles if provided
        loaded_profiles = None
        if profiles_json:
            with open(profiles_json, "r", encoding="utf-8") as f:
                loaded_profiles = json.load(f)
            print(f"Loaded {len(loaded_profiles)} participant profiles from {profiles_json}")
        
        # Load system prompt override if provided
        system_prompt_override = None
        if system_prompt_file:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                system_prompt_override = f.read()
        
        # Override distribution if n_participants is manually specified (use Case 2 logic)
        if n_participants is not None:
            print(f"Using 1 participant per trial (total: {len(trials)} trials)")
            by_sub_study_spec = None  # Force Case 2 logic 
        
        # Cache setup
        def _slugify(text: str) -> str:
            """Slugify a string for use in folder names, preserving dots but replacing slashes/hyphens."""
            return text.replace("/", "_").replace("-", "_")
        
        model_slug = _slugify(model)
        n_tag = f"n{n_participants}" if n_participants else "auto"
        cache_path_base = Path(cache_dir) / f"{study_id}__{model_slug}__{n_tag}__{system_prompt_preset}__seed{random_seed}"
        
        # Run simulation
        start_time = time.time()
        all_runs_raw_results = []
        
        # Set up output directory structure early for incremental saving
        # Import os at function level to avoid scoping issues in closures
        import os as _os_module
        if run_name:
            output_dir = Path("results/runs")
            run_dir = output_dir / run_name
        else:
            benchmark_folder = _os_module.getenv("BENCHMARK_FOLDER", "benchmark")
            output_dir = Path("results")
            run_dir = output_dir / benchmark_folder
        
        # Create config subfolder
        model_slug = model.replace("/", "_").replace("-", "_")
        prompt_slug = system_prompt_preset.replace("_", "-")
        
        # Include temperature in folder name if it's not the default
        temp_suffix = f"_temp{temperature}" if temperature != 1.0 else ""
        
        if reasoning and reasoning != "default" and reasoning != "low" and reasoning != "minimal":
            config_folder = f"{model_slug}_{reasoning}{temp_suffix}_{prompt_slug}"
        else:
            config_folder = f"{model_slug}{temp_suffix}_{prompt_slug}"
        
        study_dir = run_dir / study_id
        config_dir = study_dir / config_folder
        config_dir.mkdir(parents=True, exist_ok=True)
        incremental_output_file = config_dir / "full_benchmark.json"
        raw_responses_json = config_dir / "raw_responses.json"
        log_file_jsonl = config_dir / "raw_responses.jsonl"
        
        # RESUME LOGIC: Check if we can resume from a partial run
        existing_progress_data = None
        all_runs_raw_results = []
        
        # 1. Try to load from raw_responses.json first (the most reliable full format)
        if raw_responses_json.exists():
            try:
                with open(raw_responses_json, 'r', encoding='utf-8') as f:
                    old_raw_data = json.load(f)
                
                # Check if this file already has all the data we need
                total_collected = 0
                if old_raw_data.get('all_runs_raw_responses'):
                    for run in old_raw_data['all_runs_raw_responses']:
                        total_collected += len(run.get('participants', []))
                
                # If we have data, we'll use it to resume
                if total_collected > 0:
                    print(f"📦 Found {total_collected} responses in {raw_responses_json.name}")
                    # Convert raw_responses format to the internal all_runs_raw_results format
                    resumed_runs = []
                    for run in old_raw_data.get('all_runs_raw_responses', []):
                        individual_data = []
                        for p in run.get('participants', []):
                            # Construct back the individual response objects
                            for resp in p.get('raw_responses', []):
                                individual_data.append(resp)
                        resumed_runs.append({"individual_data": individual_data})
                    
                    all_runs_raw_results = resumed_runs
                    existing_progress_data = old_raw_data
            except Exception as e:
                print(f"⚠️  Could not read existing raw_responses.json: {e}")

        # 2. If still no data, try full_benchmark.json
        if not all_runs_raw_results and incremental_output_file.exists():
            try:
                with open(incremental_output_file, 'r', encoding='utf-8') as f:
                    existing_progress_data = json.load(f)
                if existing_progress_data.get('all_runs_raw_results'):
                    print(f"📦 Found existing data in {incremental_output_file.name}, will attempt to resume...")
                    all_runs_raw_results = existing_progress_data['all_runs_raw_results']
            except Exception:
                pass

        # 3. Last resort: recovery from raw_responses.jsonl
        if log_file_jsonl.exists():
            # ... (rest of log recovery logic) ...
            pass

        # PRE-FLIGHT CHECK: Is this study actually already finished?
        if existing_progress_data:
            # Calculate total participants we actually have across all repeats
            total_we_have = sum(len(run.get('individual_data', [])) for run in all_runs_raw_results)
            # Calculate how many we expect (roughly)
            # This is a safety check: if we have 504 participants and it asks for 504, we are done.
            expected_n = n_participants or study.specification.get("participants", {}).get("n", 30)
            if total_we_have >= (expected_n * repeats):
                print(f"✅ Study {study_id} already has complete data ({total_we_have} responses). Skipping simulation.")
                return incremental_output_file

        # Create initial file ONLY if not resuming
        if not existing_progress_data:
            from datetime import datetime as _datetime_module
            try:
                initial_data = {
                    "timestamp": _datetime_module.now().strftime("%Y%m%d_%H%M%S"),
                    "study_id": study_id,
                    "title": study.metadata.get('title', ''),
                    "model": model,
                    "reasoning": reasoning,
                    "use_real_llm": use_real_llm,
                    "system_prompt_preset": system_prompt_preset,
                    "random_seed": random_seed,
                    "repeats_completed": 0,
                    "repeats_total": repeats,
                    "status": "starting",
                    "all_runs_raw_results": []
                }
                with open(incremental_output_file, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(initial_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Created initial save file: {incremental_output_file}", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not create initial save file: {e}", flush=True)
        
        for r_idx in range(len(all_runs_raw_results), repeats):
            if repeats > 1:
                print(f"\n>>> Run {r_idx + 1}/{repeats}")
            
            r_tag = f"_r{r_idx}" if repeats > 1 else ""
            cache_path = Path(f"{cache_path_base}{r_tag}.json")
            
            current_run_raw_results = None
            
            # RESUME: Extract existing responses for this specific repeat
            existing_repeat_responses = None
            if existing_progress_data and len(all_runs_raw_results) > r_idx:
                existing_repeat_responses = all_runs_raw_results[r_idx].get('individual_data', [])
                print(f"🔄 Resuming repeat {r_idx + 1}/{repeats} with {len(existing_repeat_responses)} existing responses")

            if use_cache and cache_path.exists():
                print(f"🔄 Loading cached results from {cache_path}")
                try:
                    with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                        cached = json.load(f)
                    current_run_raw_results = cached.get('raw_results') or cached
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"⚠️  Cache file encoding/JSON error at position {getattr(e, 'start', '?')}, will regenerate: {e}")
                    # Delete corrupted cache file
                    try:
                        cache_path.unlink()
                        print(f"  → Deleted corrupted cache file")
                    except:
                        pass
                except Exception as e:
                    print(f"⚠️  Cache file error: {e}, will regenerate")
            
            if current_run_raw_results is None:
                # Check if study requires group trials (multi-participant, sequential rounds)
                requires_group_trials = getattr(study_config, 'REQUIRES_GROUP_TRIALS', False)
                
                if requires_group_trials:
                    # Special handling for studies requiring group interaction (e.g., study_009)
                    print(f"Using group experiment runner for {study_id}")
                    
                    # Prepare participant pool kwargs
                    participant_pool_kwargs = {
                        "study_specification": study.specification,
                        "use_real_llm": use_real_llm,
                        "model": model,
                        "random_seed": random_seed + r_idx,
                        "num_workers": num_workers or 1,
                        "profiles": loaded_profiles,
                        "prompt_builder": builder,
                        "system_prompt_override": system_prompt_override,
                        "system_prompt_preset": system_prompt_preset,
                        "reasoning": reasoning,
                        "enable_reasoning": enable_reasoning,
                        "temperature": temperature
                    }
                    
                    # Call custom group experiment runner
                    current_run_raw_results = study_config.run_group_experiment(
                        trials=trials,
                        instructions=instructions,
                        participant_pool_kwargs=participant_pool_kwargs,
                        prompt_builder=builder
                    )
                
                # Standard experiment flow: 1 participant per trial (one-to-one mapping)
                elif n_participants is not None or by_sub_study_spec:
                    # Use one-to-one mode: create a single pool with n_participants = len(trials)
                    # This is more efficient than creating multiple pools (avoids thread contention)
                    # Ensure n_participants matches len(trials) for one_to_one mode
                    actual_n_participants = n_participants if n_participants is not None else len(trials)
                    
                    if actual_n_participants != len(trials):
                        print(f"⚠️  Warning: n_participants ({actual_n_participants}) != len(trials) ({len(trials)}). Adjusting to {len(trials)} for one-to-one mode.")
                        actual_n_participants = len(trials)
                    
                    if n_participants is not None:
                        print(f"Using {actual_n_participants} participants: 1 participant per trial (one-to-one mode)")
                    else:
                        print(f"Using paper-faithful distribution: {actual_n_participants} participants (one-to-one mode)")
                    
                    # Create a single pool with all participants
                    pool = ParticipantPool(
                        study_specification=study.specification,
                        n_participants=actual_n_participants,
                        use_real_llm=use_real_llm,
                        model=model,
                        random_seed=random_seed + r_idx,
                        num_workers=num_workers,  # Use full parallelism here
                        profiles=loaded_profiles[:actual_n_participants] if loaded_profiles else None,
                        prompt_builder=builder,
                        system_prompt_override=system_prompt_override,
                        system_prompt_preset=system_prompt_preset,
                        study_id=study_id,
                        reasoning=reasoning,
                        enable_reasoning=enable_reasoning,
                        existing_responses=existing_repeat_responses,
                        temperature=temperature
                    )
                    
                    # Create save callback that saves after each API call returns
                    # Use throttling to avoid saving too frequently (max once per second)
                    # Import datetime at function level to avoid scoping issues in closures
                    from datetime import datetime as _dt_module
                    last_save_time = [0]  # Use list to allow modification in closure
                    last_progress_print = [0]  # Track last progress print time
                    def save_after_api_call(new_resp_data=None):
                        """Save current state after each API call completes"""
                        import time as _time
                        current_time = _time.time()
                        
                        try:
                            # 1. PROGRESSIVE LOGGING: Append to JSONL immediately (No throttle!)
                            if new_resp_data:
                                try:
                                    with open(log_file_jsonl, 'a', encoding='utf-8') as f:
                                        f.write(json.dumps(new_resp_data, ensure_ascii=False) + "\n")
                                except Exception as e:
                                    pass # Don't let log failure stop the run

                            # 2. TERMINAL UPDATE: Show progress immediately
                            # Get current count from pool
                            current_progress = sum(len(p.trial_responses) for p in pool.participants)
                            total_trials = len(trials)
                            
                            # Print progress immediately whenever there's a change
                            progress_pct = (current_progress / total_trials * 100) if total_trials > 0 else 0
                            print(f"\r   📊 Progress: {current_progress}/{total_trials} trials ({progress_pct:.1f}%) - Repeat {r_idx + 1}/{repeats}", end='', flush=True)
                            
                            # 3. THROTTLED DISK SAVE: Update full_benchmark.json every 5 seconds
                            if current_time - last_save_time[0] < 5.0:
                                return
                            last_save_time[0] = current_time
                            
                            # Get current partial results from pool for the main file
                            partial_results = pool.aggregate_results()
                            
                            # Combine with previous repeats
                            all_runs_so_far = all_runs_raw_results + [partial_results]
                            incremental_runs = [{"individual_data": run_data.get('individual_data', [])} for run_data in all_runs_so_far]
                            incremental_data = {
                                "timestamp": _dt_module.now().strftime("%Y%m%d_%H%M%S"),
                                "study_id": study_id,
                                "title": study.metadata.get('title', ''),
                                "model": model,
                                "reasoning": reasoning,
                                "use_real_llm": use_real_llm,
                                "system_prompt_preset": system_prompt_preset,
                                "random_seed": random_seed,
                                "repeats_completed": len(all_runs_raw_results),
                                "repeats_total": repeats,
                                "current_repeat_progress": current_progress,
                                "status": "in_progress",
                                "all_runs_raw_results": incremental_runs
                            }
                            # Atomic save: write to tmp then rename
                            tmp_file = incremental_output_file.with_suffix('.tmp')
                            with open(tmp_file, 'w', encoding='utf-8', errors='replace') as f:
                                json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                            tmp_file.replace(incremental_output_file)
                        except Exception as e:
                            pass
                    
                    # Run experiment in one-to-one mode (each participant runs exactly one trial)
                    # This uses ParticipantPool's internal ThreadPoolExecutor with num_workers
                    # This is much faster than creating multiple pools with num_workers=1
                    print(f"\n🚀 Starting simulation: {len(trials)} trials, {actual_n_participants} participants, {num_workers or 1} worker(s)", flush=True)
                    
                    # Add a progress monitor thread to show updates more frequently
                    import threading
                    progress_stop = threading.Event()
                    def progress_monitor():
                        """Monitor and print progress every 0.5 seconds"""
                        last_count = 0
                        while not progress_stop.is_set():
                            try:
                                current_count = sum(len(p.trial_responses) for p in pool.participants)
                                if current_count > last_count:
                                    progress_pct = (current_count / len(trials) * 100) if len(trials) > 0 else 0
                                    print(f"\r   📊 Progress: {current_count}/{len(trials)} trials ({progress_pct:.1f}%) - Repeat {r_idx + 1}/{repeats}", end='', flush=True)
                                    last_count = current_count
                            except:
                                pass
                            progress_stop.wait(0.5)  # Check every 0.5 seconds
                    
                    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
                    monitor_thread.start()
                    
                    try:
                        current_run_raw_results = pool.run_experiment(
                            trials, 
                            instructions, 
                            prompt_builder=builder,
                            one_to_one=True,  # Enable one-to-one mode for better performance
                            save_callback=save_after_api_call  # Save after each API call
                        )
                    finally:
                        progress_stop.set()  # Stop the monitor
                
                # Case 3: Fallback default
                else:
                    n_def = study.specification.get('participants', {}).get('n') or 30
                    print(f"Using default N={n_def}")
                    pool = ParticipantPool(
                        study_specification=study.specification,
                        n_participants=n_def,
                        use_real_llm=use_real_llm,
                        model=model,
                        random_seed=random_seed + r_idx,
                        num_workers=num_workers,
                        profiles=loaded_profiles,
                        prompt_builder=builder,
                        system_prompt_override=system_prompt_override,
                        system_prompt_preset=system_prompt_preset,
                        study_id=study_id,
                        reasoning=reasoning,
                        enable_reasoning=enable_reasoning,
                        existing_responses=existing_repeat_responses,
                        temperature=temperature
                    )
                    
                    # Create save callback that saves after each API call returns
                    # Use throttling to avoid saving too frequently (max once per second)
                    # Import datetime at function level to avoid scoping issues in closures
                    from datetime import datetime as _dt_module_fallback
                    last_save_time_fallback = [0]  # Use list to allow modification in closure
                    last_progress_print_fallback = [0]  # Track last progress print time
                    def save_after_api_call_fallback(new_resp_data=None):
                        """Save current state after each API call completes"""
                        import time as _time
                        current_time = _time.time()
                        
                        try:
                            # 1. PROGRESSIVE LOGGING: Append to JSONL immediately
                            if new_resp_data:
                                try:
                                    with open(log_file_jsonl, 'a', encoding='utf-8') as f:
                                        f.write(json.dumps(new_resp_data, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass

                            # 2. TERMINAL UPDATE: Show progress immediately
                            current_progress = sum(len(p.trial_responses) for p in pool.participants)
                            total_expected = n_def * len(trials) if n_def else len(trials) * len(pool.participants)
                            
                            # Print progress immediately
                            progress_pct = (current_progress / total_expected * 100) if total_expected > 0 else 0
                            print(f"\r   📊 Progress: {current_progress}/{total_expected} responses ({progress_pct:.1f}%) - Repeat {r_idx + 1}/{repeats}", end='', flush=True)
                            
                            # 3. THROTTLED DISK SAVE: Update full_benchmark.json every 5 seconds
                            if current_time - last_save_time_fallback[0] < 5.0:
                                return
                            last_save_time_fallback[0] = current_time
                            
                            # Get current partial results from pool
                            partial_results = pool.aggregate_results()
                            
                            # Combine with previous repeats
                            all_runs_so_far = all_runs_raw_results + [partial_results]
                            incremental_runs = [{"individual_data": run_data.get('individual_data', [])} for run_data in all_runs_so_far]
                            incremental_data = {
                                "timestamp": _dt_module_fallback.now().strftime("%Y%m%d_%H%M%S"),
                                "study_id": study_id,
                                "title": study.metadata.get('title', ''),
                                "model": model,
                                "reasoning": reasoning,
                                "use_real_llm": use_real_llm,
                                "system_prompt_preset": system_prompt_preset,
                                "random_seed": random_seed,
                                "repeats_completed": len(all_runs_raw_results),
                                "repeats_total": repeats,
                                "current_repeat_progress": current_progress,
                                "status": "in_progress",
                                "all_runs_raw_results": incremental_runs
                            }
                            # Atomic save: write to tmp then rename
                            tmp_file = incremental_output_file.with_suffix('.tmp')
                            with open(tmp_file, 'w', encoding='utf-8', errors='replace') as f:
                                json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                            tmp_file.replace(incremental_output_file)
                        except Exception as e:
                            pass
                    
                    print(f"\n🚀 Starting simulation: {len(trials)} trials per participant, {n_def} participants, {num_workers or 1} worker(s)", flush=True)
                    
                    # Add a progress monitor thread to show updates more frequently
                    import threading
                    progress_stop_fallback = threading.Event()
                    def progress_monitor_fallback():
                        """Monitor and print progress every 0.5 seconds"""
                        last_count = 0
                        total_expected = n_def * len(trials)
                        while not progress_stop_fallback.is_set():
                            try:
                                current_count = sum(len(p.trial_responses) for p in pool.participants)
                                if current_count > last_count:
                                    progress_pct = (current_count / total_expected * 100) if total_expected > 0 else 0
                                    print(f"\r   📊 Progress: {current_count}/{total_expected} responses ({progress_pct:.1f}%) - Repeat {r_idx + 1}/{repeats}", end='', flush=True)
                                    last_count = current_count
                            except:
                                pass
                            progress_stop_fallback.wait(0.5)  # Check every 0.5 seconds
                    
                    monitor_thread_fallback = threading.Thread(target=progress_monitor_fallback, daemon=True)
                    monitor_thread_fallback.start()
                    
                    try:
                        current_run_raw_results = pool.run_experiment(
                            trials, 
                            instructions, 
                            prompt_builder=builder,
                            save_callback=save_after_api_call_fallback  # Save after each API call
                        )
                    finally:
                        progress_stop_fallback.set()  # Stop the monitor
                
                # Save cache
                if use_cache:
                    payload = {
                        "version": 1,
                        "study_id": study_id,
                        "repeat_idx": r_idx,
                        "raw_results": current_run_raw_results,
                    }
                    with open(cache_path, 'w', encoding='utf-8', errors='replace') as f:
                        json.dump(payload, f, ensure_ascii=False)
            
            # Print completion message
            num_responses = len(current_run_raw_results.get('individual_data', []))
            print(f"✅ Repeat {r_idx + 1}/{repeats} complete: {num_responses} responses collected", flush=True)
            
            all_runs_raw_results.append(current_run_raw_results)
            
            # INCREMENTAL SAVE: Save results after each repeat to prevent data loss
            # (Output directory structure is set up before the loop)
            try:
                # Prepare incremental save data
                # Use function-level datetime import to avoid scoping issues
                from datetime import datetime as _dt_module_incremental
                incremental_runs = [{"individual_data": run_data.get('individual_data', [])} for run_data in all_runs_raw_results]
                incremental_data = {
                    "timestamp": _dt_module_incremental.now().strftime("%Y%m%d_%H%M%S"),
                    "study_id": study_id,
                    "title": study.metadata.get('title', ''),
                    "model": model,
                    "reasoning": reasoning,
                    "use_real_llm": use_real_llm,
                    "system_prompt_preset": system_prompt_preset,
                    "random_seed": random_seed,
                    "repeats_completed": len(all_runs_raw_results),
                    "repeats_total": repeats,
                    "status": "in_progress" if len(all_runs_raw_results) < repeats else "complete",
                    "all_runs_raw_results": incremental_runs
                }
                
                # Save incrementally (will be overwritten at final save)
                with open(incremental_output_file, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Incremental save: {len(all_runs_raw_results)}/{repeats} repeats completed", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not save incrementally: {e}", flush=True)
        
        # Try to aggregate results (optional)
        try:
            # Use the first run's data for aggregation stats
            aggregation_source = all_runs_raw_results[0] if all_runs_raw_results else {"individual_data": []}
            results = study_config.aggregate_results(aggregation_source)
        except Exception as e:
            print(f"Warning: Failed to aggregate results: {e}")
            results = {"descriptive_statistics": {}, "inferential_statistics": {}, "error": str(e)}
        
        # Save results - simplified structure
        # Use global datetime import from top of file
        import os as _os_module
        
        # Use run_name or default to benchmark folder
        if run_name:
            # Named runs go to results/runs/{run_name}/
            output_dir = Path("results/runs")
            run_dir = output_dir / run_name
        else:
            # Default: save directly to results/benchmark/ (baseline)
            benchmark_folder = _os_module.getenv("BENCHMARK_FOLDER", "benchmark")
            output_dir = Path("results")
            run_dir = output_dir / benchmark_folder
        
        # Create config subfolder: {model}_{reasoning}_{prompt_preset}
        model_slug = model.replace("/", "_").replace("-", "_")
        prompt_slug = system_prompt_preset.replace("_", "-")
        
        # Include temperature in folder name if it's not the default
        temp_suffix = f"_temp{temperature}" if temperature != 1.0 else ""
        
        # Don't add reasoning postfix for "default", "low", or "minimal" (treat as default)
        if reasoning and reasoning != "default" and reasoning != "low" and reasoning != "minimal":
            config_folder = f"{model_slug}_{reasoning}{temp_suffix}_{prompt_slug}"
        else:
            config_folder = f"{model_slug}{temp_suffix}_{prompt_slug}"
        
        # Structure: results/{benchmark|runs/{run_name}}/{study_id}/{config_folder}/
        study_dir = run_dir / study_id
        config_dir = study_dir / config_folder
        config_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = config_dir / "full_benchmark.json"
        
        # Load existing results if merging repeats
        existing_runs = []
        existing_metadata = {}
        if merge_existing_repeats and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Extract existing runs
                if existing_data.get('all_runs_raw_results'):
                    existing_runs = existing_data['all_runs_raw_results']
                elif existing_data.get('individual_data'):
                    # Convert single run to list format
                    existing_runs = [{"individual_data": existing_data['individual_data']}]
                
                # Preserve metadata from existing file
                existing_metadata = {
                    "timestamp": existing_data.get('timestamp', timestamp),
                    "title": existing_data.get('title', study.metadata['title']),
                    "model": existing_data.get('model', model),
                    "use_real_llm": existing_data.get('use_real_llm', use_real_llm),
                    "system_prompt_preset": existing_data.get('system_prompt_preset', system_prompt_preset),
                    "random_seed": existing_data.get('random_seed', random_seed),
                }
                
                print(f"📦 Found {len(existing_runs)} existing run(s), adding {repeats} more...")
            except Exception as e:
                print(f"⚠️  Could not load existing results for merging: {e}")
                existing_runs = []
        
        # Merge new runs with existing runs
        all_merged_runs = existing_runs + [
            {"individual_data": run_data.get('individual_data', [])}
            for run_data in all_runs_raw_results
        ]
        
        total_repeats = len(all_merged_runs)
        
        # Helper function to remove raw_response_text from response data (keep full_benchmark.json small)
        def clean_response_data(data):
            """Remove raw_response_text from response objects and sanitize text fields for UTF-8"""
            if isinstance(data, dict):
                cleaned = {k: v for k, v in data.items() if k != 'raw_response_text'}
                for k, v in cleaned.items():
                    cleaned[k] = clean_response_data(v)
                return cleaned
            elif isinstance(data, list):
                return [clean_response_data(item) for item in data]
            elif isinstance(data, str):
                # Sanitize string values to ensure UTF-8 compatibility
                try:
                    # Try to encode as UTF-8 to check validity
                    data.encode('utf-8')
                    return data
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # Replace invalid characters
                    return data.encode('utf-8', errors='replace').decode('utf-8')
            else:
                return data
        
        # Extract raw responses before cleaning
        raw_responses_data = {
            "timestamp": existing_metadata.get("timestamp", timestamp),
            "study_id": study_id,
            "title": existing_metadata.get("title", study.metadata['title']),
            "model": existing_metadata.get("model", model),
            "reasoning": existing_metadata.get("reasoning", reasoning),
            "use_real_llm": existing_metadata.get("use_real_llm", use_real_llm),
            "system_prompt_preset": existing_metadata.get("system_prompt_preset", system_prompt_preset),
            "random_seed": existing_metadata.get("random_seed", random_seed),
            "repeats": total_repeats,
            "all_runs_raw_responses": []
        }
        
        # Extract raw responses from all runs
        for run_idx, run_data in enumerate(all_merged_runs):
            participants_data = run_data.get('individual_data', [])
            run_raw_responses = []
            
            # Check if data is flat structure (legacy format) or nested structure (from Stage 5)
            # Flat structure: individual_data is a list of trial responses (each has participant_id, response_text, etc.)
            # Nested structure: individual_data is a list of participants (each has participant_id, responses[])
            # Note: Flat structure is kept for backward compatibility with legacy results
            is_flat_structure = len(participants_data) > 0 and 'responses' not in participants_data[0]
            
            if is_flat_structure:
                # Handle flat structure: group responses by participant_id
                participant_responses_map = defaultdict(list)
                
                for response_item in participants_data:
                    participant_id = response_item.get('participant_id', 0)
                    # Make a deep copy to avoid modifying the original
                    raw_resp = copy.deepcopy(response_item)
                    
                    # Remove items field from trial_info to reduce file size
                    if "trial_info" in raw_resp and isinstance(raw_resp["trial_info"], dict):
                        if "items" in raw_resp["trial_info"]:
                            del raw_resp["trial_info"]["items"]
                    
                    participant_responses_map[participant_id].append(raw_resp)
                
                # Convert to nested format for raw_responses.json
                for participant_id, responses in participant_responses_map.items():
                    participant_raw_responses = {
                        "participant_id": participant_id,
                        "raw_responses": responses
                    }
                    run_raw_responses.append(participant_raw_responses)
            else:
                # Handle nested structure (original Stage 5 format)
                for participant_data in participants_data:
                    participant_id = participant_data.get('participant_id', 0)
                    responses = participant_data.get('responses', [])
                    participant_raw_responses = {
                        "participant_id": participant_id,
                        "raw_responses": []
                    }
                    
                    for response in responses:
                        # Save complete response dictionary to prevent information loss
                        # Make a deep copy to avoid modifying the original
                        raw_resp = copy.deepcopy(response)
                        
                        # Remove items field from trial_info to reduce file size
                        if "trial_info" in raw_resp and isinstance(raw_resp["trial_info"], dict):
                            if "items" in raw_resp["trial_info"]:
                                del raw_resp["trial_info"]["items"]
                        
                        participant_raw_responses["raw_responses"].append(raw_resp)
                    
                    run_raw_responses.append(participant_raw_responses)
            
            raw_responses_data["all_runs_raw_responses"].append({
                "run_index": run_idx,
                "participants": run_raw_responses
            })
        
        # Prepare study data (cleaned, without raw_response_text)
        # Use all_merged_runs[0] as the primary data source for summary stats
        primary_run_data = all_merged_runs[0] if all_merged_runs else {"individual_data": []}
        
        cleaned_individual_data = clean_response_data(primary_run_data.get('individual_data', []))
        cleaned_all_runs = [{"individual_data": clean_response_data(run_data.get('individual_data', []))} for run_data in all_merged_runs] if total_repeats > 1 else None
        
        # Calculate overall usage statistics for the saved data
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        total_participants_all_runs = 0
        
        for run in all_merged_runs:
            participants_data = run.get('individual_data', [])
            is_flat = len(participants_data) > 0 and 'responses' not in participants_data[0]
            
            if is_flat:
                # Handle flat structure
                total_participants_all_runs += len(set(resp.get('participant_id') for resp in participants_data))
                for resp in participants_data:
                    usage = resp.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                    total_completion_tokens += usage.get('completion_tokens', 0) or 0
                    total_tokens += usage.get('total_tokens', 0) or 0
                    total_cost += usage.get('cost', 0.0) or 0.0
            else:
                # Handle nested structure
                total_participants_all_runs += len(participants_data)
                for participant in participants_data:
                    for resp in participant.get('responses', []):
                        usage = resp.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                        total_completion_tokens += usage.get('completion_tokens', 0) or 0
                        total_tokens += usage.get('total_tokens', 0) or 0
                        total_cost += usage.get('cost', 0.0) or 0.0
        
        avg_tokens_per_participant = total_tokens / total_participants_all_runs if total_participants_all_runs > 0 else 0
        avg_cost_per_participant = total_cost / total_participants_all_runs if total_participants_all_runs > 0 else 0

        save_data = {
            "timestamp": existing_metadata.get("timestamp", timestamp),
            "study_id": study_id,
            "title": existing_metadata.get("title", study.metadata['title']),
            "model": existing_metadata.get("model", model),
            "reasoning": existing_metadata.get("reasoning", reasoning),
            "use_real_llm": existing_metadata.get("use_real_llm", use_real_llm),
            "system_prompt_preset": existing_metadata.get("system_prompt_preset", system_prompt_preset),
            "random_seed": existing_metadata.get("random_seed", random_seed),
            "elapsed_time": time.time() - start_time,
            "repeats": total_repeats,
            "usage_stats": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost": float(total_cost),
                "avg_tokens_per_participant": float(avg_tokens_per_participant),
                "avg_cost_per_participant": float(avg_cost_per_participant)
            },
            "descriptive_statistics": results.get('descriptive_statistics', {}),
            "inferential_statistics": results.get('inferential_statistics', {}),
            "individual_data": cleaned_individual_data,
            "all_runs_raw_results": cleaned_all_runs,
            "summary": {
                "total_participants": len(cleaned_individual_data) if cleaned_individual_data else 0
            }
        }
        
        # Save (overwrite with merged data)
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save raw responses to separate file
        raw_responses_file = config_dir / "raw_responses.json"
        with open(raw_responses_file, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(raw_responses_data, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Stage 5 complete!")
        print(f"  - Simulation time: {elapsed:.1f}s")
        print(f"  - Results saved to: {output_file}")
        print(f"  - Raw responses saved to: {raw_responses_file}")
        if merge_existing_repeats and total_repeats > repeats:
            print(f"  - Participants: {n_participants}, Total Runs: {total_repeats} ({len(existing_runs)} existing + {repeats} new)")
        else:
            print(f"  - Participants: {n_participants}, Runs: {total_repeats}")
        print(f"\nNext step: Run Stage 6 to generate scores")
        print(f"  python generation_pipeline/run.py --stage 6 --study-id {study_id}")
        
        return output_file
    
    def run_stage6(
        self,
        study_id: str,
        study_dir: Optional[Path] = None,
        skip_generation: bool = False,
        run_name: Optional[str] = None,
        config_folder: Optional[str] = None,
        disable_formatter: bool = True
    ) -> Path:
        """
        Run stage 6 (Evaluator generation and scoring).
        
        Args:
            study_id: Study ID (e.g., "study_001")
            study_dir: Study directory path
            skip_generation: Skip evaluator code generation
            run_name: Name of the run
            config_folder: Specific config folder to evaluate (optional)
            
        Returns:
            Path to generated evaluator file
        """
        print(f"Running Stage 6: Generating Evaluator and Computing Scores for {study_id}")
        import json
        import os as _os_module
        from pathlib import Path
        
        # Determine study directory
        if study_dir is None:
            study_dir = Path("data/studies") / study_id
        
        study_dir = Path(study_dir)
        if not study_dir.exists():
            raise FileNotFoundError(f"Study directory not found: {study_dir}")
        
        # Check required files exist
        required_files = ["ground_truth.json", "specification.json"]
        for fname in required_files:
            if not (study_dir / fname).exists():
                raise FileNotFoundError(f"Required file not found: {study_dir / fname}")
        
        # Generate evaluator (unless skipping)
        evaluator_path = Path("src/studies") / f"{study_id}_evaluator.py"
        evaluator_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not skip_generation:
            from src.generators.evaluator_generator import EvaluatorGenerator
            generator = EvaluatorGenerator(llm_client=self.client)
            success = generator.generate_evaluator(study_id, study_dir, evaluator_path)
            
            if not success:
                raise RuntimeError(f"Failed to generate evaluator for {study_id}")
            
            print(f"  - Generated: {evaluator_path}")
        else:
            if not evaluator_path.exists():
                raise FileNotFoundError(f"Evaluator not found: {evaluator_path}. Run without --skip-generation to generate it first.")
            print(f"  - Using existing evaluator: {evaluator_path}")
        
        # Find results base directory
        if run_name:
            results_base_dir = Path("results/runs") / run_name
        else:
            benchmark_folder = _os_module.getenv("BENCHMARK_FOLDER", "benchmark")
            results_base_dir = Path("results") / benchmark_folder
        
        results_study_dir = results_base_dir / study_id
        
        if not results_study_dir.exists():
            raise FileNotFoundError(f"Study results directory not found: {results_study_dir}. Run Stage 5 first.")
        
        # Find the config folders to process
        all_config_dirs = [d for d in results_study_dir.iterdir() if d.is_dir()]
        
        if not all_config_dirs:
            raise FileNotFoundError(f"No config folders found in {results_study_dir}")
        
        if config_folder:
            config_dirs_to_process = [d for d in all_config_dirs if d.name == config_folder]
            if not config_dirs_to_process:
                raise FileNotFoundError(f"Specific config folder not found: {config_folder}")
        else:
            config_dirs_to_process = sorted(all_config_dirs)
            if len(config_dirs_to_process) > 1:
                print(f"  - Found {len(config_dirs_to_process)} config folders, evaluating all...")
        
        # Process each config folder
        from src.evaluation.evaluator_runner import run_evaluator, load_evaluator
        import csv
        import numpy as np
        from scipy import stats
        from src.evaluation.stats_lib import aggregate_study_pas
        
        last_evaluator_path = evaluator_path
        
        print(f"  Processing {len(config_dirs_to_process)} config folder(s)...")
        for idx, cfg_dir in enumerate(config_dirs_to_process, 1):
            print(f"\n  >>> [{idx}/{len(config_dirs_to_process)}] Evaluating: {cfg_dir.name}")
            
            benchmark_file = cfg_dir / "full_benchmark.json"
            if not benchmark_file.exists():
                print(f"  ⚠️  Benchmark file not found in {cfg_dir.name}, skipping.")
                continue
            
            # ===== Load benchmark data =====
            print(f"  - Loading benchmark data...")
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
            print(f"    ✓ Loaded benchmark data")
            
            # ===== Calculate Raw Failure Rate =====
            print(f"  - Calculating raw failure rate...")
            from src.evaluation.sanity_check import calculate_raw_failure_rate
            
            raw_failure_stats = calculate_raw_failure_rate(benchmark_data)
            raw_failure_rate = raw_failure_stats.get("raw_failure_rate", 0.0)
            print(f"    → Raw failure rate: {raw_failure_rate:.2f}% ({raw_failure_stats.get('raw_failed', 0)}/{raw_failure_stats.get('raw_total', 0)})")
            if raw_failure_stats.get('raw_failure_breakdown'):
                breakdown = raw_failure_stats['raw_failure_breakdown']
                print(f"      Breakdown: {breakdown.get('empty', 0)} empty, {breakdown.get('refusal', 0)} refusal, {breakdown.get('other', 0)} other")
            
            # ===== Sanity Check: Verify response extraction (Final Failure Rate) =====
            print(f"  - Running sanity check for response extraction...")
            from src.evaluation.sanity_check import run_sanity_check, format_failed_responses
            
            # Count total responses for progress indication
            total_resp_count = len(benchmark_data.get('individual_data', []))
            if total_resp_count > 0:
                print(f"    → Checking {total_resp_count} responses...")
            
            sanity_check_result = run_sanity_check(study_id, benchmark_file, evaluator_path)
            
            if not sanity_check_result.get("all_passed", True):
                failed_responses = sanity_check_result.get("failed_responses", [])
                print(f"  ⚠️  Found {len(failed_responses)}/{sanity_check_result.get('total_checked', 0)} responses that cannot be fully extracted")
                
                # 打印一个失败的例子
                if failed_responses:
                    example = failed_responses[0]
                    print(f"\n  Example failed response:")
                    print(f"    Participant {example['participant_id']}, Response {example['response_index']}")
                    print(f"    Missing Q numbers: {example['missing_q_numbers']}")
                    print(f"    Required: {example['required_q_numbers']}")
                    print(f"    Extracted: {example['extracted_q_numbers']}")
                    print(f"    Response preview: {example['response_text_preview'][:300]}")
                
                # Formatter disabled - skip automatic formatting of failed responses
                # disable_formatter=True means formatter is disabled (default)
                # Only run formatter if disable_formatter is False (explicitly enabled)
                if not disable_formatter:
                    print(f"  - Activating formatter for failed responses (multithreading enabled)...")
                    
                    # 使用多线程格式化失败的响应
                    formatted_count = format_failed_responses(
                        study_id=study_id,
                        benchmark_file=benchmark_file,
                        failed_responses=failed_responses,
                        evaluator_path=evaluator_path,
                        num_workers=32
                    )
                    
                    print(f"  ✓ Formatted {formatted_count}/{len(failed_responses)} responses")
                    
                    # 重新运行sanity check验证格式化结果
                    sanity_check_result = run_sanity_check(study_id, benchmark_file, evaluator_path)
                    if not sanity_check_result.get("all_passed", True):
                        remaining_failed = len(sanity_check_result.get("failed_responses", []))
                        print(f"  ⚠️  Warning: {remaining_failed} responses still cannot be extracted after formatting")
                    else:
                        print(f"  ✓ All responses passed sanity check after formatting")
                else:
                    print(f"  ⚠️  Skipping formatter (disabled) - {len(failed_responses)} responses failed extraction")
            else:
                total_checked = sanity_check_result.get('total_checked', 0)
                passed = sanity_check_result.get('passed', 0)
                skipped = sanity_check_result.get('skipped_responses', 0)
                total = sanity_check_result.get('total_responses', total_checked)
                if skipped > 0:
                    print(f"  ✓ All responses passed sanity check ({passed}/{total_checked} checked, {skipped} skipped out of {total} total)")
                else:
                    print(f"  ✓ All responses passed sanity check ({passed}/{total_checked})")
            
            # ===== Use already loaded benchmark data for evaluation =====
            individual_data = benchmark_data.get('individual_data', [])
            all_runs_data = []
            if benchmark_data.get('all_runs_raw_results'):
                for run_data in benchmark_data['all_runs_raw_results']:
                    all_runs_data.append(run_data.get('individual_data', []))
            else:
                all_runs_data = [individual_data] if individual_data else []
            
            if not all_runs_data or not all_runs_data[0]:
                print(f"  ⚠️  No raw response data found in benchmark results for {cfg_dir.name}")
                continue
            
            # Load evaluator module once (for bootstrap efficiency)
            print(f"  - Loading evaluator module...")
            # #region agent log
            import json
            import time
            from pathlib import Path as _Path
            def _debug_log_pipe(location, message, data, hypothesis_id=None):
                try:
                    log_path = _Path("/Users/assassin808/Desktop/xuan-hs/HS_bench/.cursor/debug.log")
                    with open(log_path, "a", encoding="utf-8") as f:
                        log_entry = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000)}
                        f.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
            _debug_log_pipe("pipeline.py:1436", "Before load_evaluator call", {"study_id": study_id, "cfg_dir_name": cfg_dir.name}, "D")
            # #endregion
            evaluator_module = load_evaluator(study_id)
            # #region agent log
            _debug_log_pipe("pipeline.py:1438", "After load_evaluator call", {"study_id": study_id, "cfg_dir_name": cfg_dir.name, "evaluator_module_is_none": evaluator_module is None}, "D")
            # #endregion
            if evaluator_module is None:
                print(f"    ❌ Failed to load evaluator for {cfg_dir.name}")
                # #region agent log
                _debug_log_pipe("pipeline.py:1441", "Evaluator load failed", {"study_id": study_id, "cfg_dir_name": cfg_dir.name}, "D")
                # #endregion
                continue
            print(f"    ✓ Evaluator module loaded")
            
            # Combine all participant data from all runs into a single pool
            combined_participant_pool = []
            for run_individual_data in all_runs_data:
                if run_individual_data:
                    combined_participant_pool.extend(run_individual_data)
            
            if not combined_participant_pool:
                print(f"    ❌ No participant data available for bootstrap")
                continue
            
            # Check if data is flat structure (legacy format) or nested structure (from Stage 5)
            # Note: Flat structure is kept for backward compatibility with legacy results
            is_flat_structure = len(combined_participant_pool) > 0 and 'responses' not in combined_participant_pool[0]
            
            if is_flat_structure:
                # Convert flat structure to nested structure for evaluator
                print(f"    - Converting flat structure to nested structure for evaluator...")
                nested_participants = defaultdict(lambda: {"responses": [], "profile": {}})
                
                for item_response in combined_participant_pool:
                    p_id = item_response.get("participant_id", 0)
                    # Ensure profile is copied from the trial_info if available
                    if not nested_participants[p_id]["profile"] and "trial_info" in item_response and "profile" in item_response["trial_info"]:
                        nested_participants[p_id]["profile"] = item_response["trial_info"]["profile"]
                    # Add the response to the participant's responses list
                    nested_participants[p_id]["responses"].append(item_response)
                
                # Convert defaultdict to regular list with participant_id set
                combined_participant_pool = []
                for p_id, p_data in nested_participants.items():
                    p_data["participant_id"] = p_id
                    combined_participant_pool.append(p_data)
                
                print(f"    - Converted {len(nested_participants)} flat responses to {len(combined_participant_pool)} nested participants")
            
            n_runs = len(all_runs_data)
            print(f"    - Processing {n_runs} run(s) with {len(combined_participant_pool)} total participants...")
            
            # Get evaluation result (main metric: ECS_corr = CCC)
            print(f"  - Computing detailed evaluation results...")
            full_raw_results = {"individual_data": combined_participant_pool}
            pas_result = evaluator_module.evaluate_study(full_raw_results)
            
            if "error" in pas_result:
                print(f"    ❌ Evaluation failed on full data: {pas_result.get('error')}")
                continue
            print(f"    ✓ Detailed evaluation complete")
            
            test_results = pas_result.get('test_results', [])
            from src.evaluation.stats_lib import compute_ecs_corr
            
            STUDY_GROUPS = {
                "Cognition": ["study_001", "study_002", "study_003", "study_004"],
                "Strategic": ["study_009", "study_010", "study_011", "study_012"],
                "Social": ["study_005", "study_006", "study_007", "study_008"]
            }
            
            # Compute ECS_corr (CCC = Lin's Concordance Correlation Coefficient) — primary metric only
            ecs_corr_result = compute_ecs_corr(test_results, study_groups=STUDY_GROUPS)
            pas_result['ecs_corr'] = ecs_corr_result.get('ecs_overall')  # CCC overall (for this study's tests)
            pas_result['ecs_corr_details'] = {
                'n_tests_overall': ecs_corr_result.get('n_tests_overall', 0),
                'n_tests_per_study': ecs_corr_result.get('n_tests_per_study', {}),
                'ecs_per_study': ecs_corr_result.get('ecs_per_study', {}),
                'ecs_domain': ecs_corr_result.get('ecs_domain', {}),
                'caricature_overall': ecs_corr_result.get('caricature_overall', {'a': None, 'b': None}),
            }
            result_study_id = pas_result.get('study_id')
            if result_study_id and result_study_id in ecs_corr_result.get('ecs_per_study', {}):
                pas_result['ecs_corr_study'] = ecs_corr_result['ecs_per_study'][result_study_id]
            else:
                pas_result['ecs_corr_study'] = pas_result.get('ecs_corr')
            
            # ECS missing rate: fraction of tests without valid human_effect_d / agent_effect_d
            n_total = len(test_results)
            n_valid = ecs_corr_result.get('n_tests_overall', 0)
            pas_result['ecs_corr_details']['ecs_missing_rate'] = (1.0 - (n_valid / n_total)) if n_total > 0 else None
            
            pas_result.update({
                'n_participants': len(combined_participant_pool),
                'n_runs': n_runs,
            })

            # ===== Calculate Usage Statistics =====
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            total_cost = 0.0
            
            # Check if data is flat or nested to iterate correctly
            is_flat_structure = len(combined_participant_pool) > 0 and 'responses' not in combined_participant_pool[0]
            
            if is_flat_structure:
                # Handle flat structure: each item is a trial response
                for resp in combined_participant_pool:
                    usage = resp.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                    total_completion_tokens += usage.get('completion_tokens', 0) or 0
                    total_tokens += usage.get('total_tokens', 0) or 0
                    total_cost += usage.get('cost', 0.0) or 0.0
                n_participants_count = len(set(resp.get('participant_id') for resp in combined_participant_pool))
            else:
                # Handle nested structure: each item is a participant with multiple responses
                for participant in combined_participant_pool:
                    for resp in participant.get('responses', []):
                        usage = resp.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                        total_completion_tokens += usage.get('completion_tokens', 0) or 0
                        total_tokens += usage.get('total_tokens', 0) or 0
                        total_cost += usage.get('cost', 0.0) or 0.0
                n_participants_count = len(combined_participant_pool)
            
            avg_tokens_per_participant = total_tokens / n_participants_count if n_participants_count > 0 else 0
            avg_cost_per_participant = total_cost / n_participants_count if n_participants_count > 0 else 0
            
            # ===== Calculate Final Failure Rate (from sanity check) =====
            final_failed = sanity_check_result.get('failed', 0)
            final_total = sanity_check_result.get('total_checked', 0)
            final_failure_rate = (final_failed / final_total * 100.0) if final_total > 0 else 0.0
            
            print(f"      → Final failure rate: {final_failure_rate:.2f}% ({final_failed}/{final_total})")
            
            pas_result.update({
                'usage_stats': {
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': float(total_cost),
                    'avg_tokens_per_participant': float(avg_tokens_per_participant),
                    'avg_cost_per_participant': float(avg_cost_per_participant)
                },
                'failure_rates': {
                    'raw_failure_rate': float(raw_failure_rate),
                    'raw_failed': raw_failure_stats.get('raw_failed', 0),
                    'raw_total': raw_failure_stats.get('raw_total', 0),
                    'raw_failure_breakdown': raw_failure_stats.get('raw_failure_breakdown', {}),
                    'final_failure_rate': float(final_failure_rate),
                    'final_failed': final_failed,
                    'final_total': final_total
                }
            })
            
            # Print usage summary
            print(f"      → Usage: Total {total_tokens} tokens, Total Cost ${total_cost:.4f}")
            print(f"      → Average per participant: {avg_tokens_per_participant:.1f} tokens, ${avg_cost_per_participant:.6f}")
            
            # Save detailed CSV
            print(f"  - Saving results...")
            csv_path = cfg_dir / "detailed_stats.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = [
                    "Study_ID", "Sub_Study_ID", "Finding_ID", "Test_Name", "Scenario",
                    "Statistical_Test_Type", "Human_Test_Statistic", "Agent_Test_Statistic",
                    "Z_Diff", "ECS_Test", "Agent_Effect_Size", "Human_Effect_Size",
                    "Agent_Effect_d", "Human_Effect_d"
                ]
                writer.writerow(header)
                for test in pas_result.get('test_results', []):
                    z_diff = test.get('z_diff')
                    rep_cons = test.get('replication_consistency')
                    agent_es = test.get('agent_effect_size')
                    human_es = test.get('human_effect_size')
                    agent_ed = test.get('agent_effect_d')
                    human_ed = test.get('human_effect_d')
                    z_diff_str = f"{z_diff:.4f}" if z_diff is not None else ""
                    rep_cons_str = f"{rep_cons:.4f}" if rep_cons is not None else ""
                    agent_es_str = f"{agent_es:.4f}" if agent_es is not None else ""
                    human_es_str = f"{human_es:.4f}" if human_es is not None else ""
                    agent_ed_str = f"{agent_ed:.4f}" if agent_ed is not None else ""
                    human_ed_str = f"{human_ed:.4f}" if human_ed is not None else ""
                    row = [
                        study_id,
                        test.get('sub_study_id', ''),
                        test.get('finding_id', ''),
                        test.get('test_name', ''),
                        test.get('scenario', ''),
                        test.get('statistical_test_type', ''),
                        test.get('human_test_statistic', ''),
                        test.get('agent_test_statistic', ''),
                        z_diff_str,
                        rep_cons_str,
                        agent_es_str,
                        human_es_str,
                        agent_ed_str,
                        human_ed_str
                    ]
                    writer.writerow(row)
            
            json_path = cfg_dir / "evaluation_results.json"
            from src.utils.io import atomic_write_json
            atomic_write_json(json_path, pas_result, indent=2, ensure_ascii=False, encoding='utf-8', errors='replace')
            
            ecs_val = pas_result.get('ecs_corr')
            ecs_str = f"{ecs_val:.4f}" if ecs_val is not None else "N/A (need ≥3 tests)"
            print(f"    ✅ Results saved: ECS_corr (CCC) = {ecs_str}")
            print(f"    ✓ Saved to: {csv_path.name}, {json_path.name}")
        
        return last_evaluator_path

