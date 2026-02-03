"""
Main Validation Pipeline Orchestrator

Coordinates all validation agents to perform comprehensive study validation.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from legacy.validation_pipeline.utils.document_loader import DocumentLoader
from legacy.validation_pipeline.utils.gemini_client import GeminiClient
from legacy.validation_pipeline.agents import (
    ExperimentCompletenessAgent,
    ExperimentConsistencyAgent,
    DataValidationAgent,
    ChecklistGeneratorAgent,
)
from legacy.validation_pipeline.agents.material_validation_agent import MaterialValidationAgent
from legacy.validation_pipeline.agents.evaluator_validation_agent import EvaluatorValidationAgent


class ValidationPipeline:
    """Main pipeline orchestrator for study validation"""
    
    def __init__(
        self,
        model: str = "models/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize validation pipeline.
        
        Args:
            model: Gemini model to use
            api_key: Optional API key (if None, reads from env)
            output_dir: Directory to save validation results
        """
        self.model = model
        self.client = GeminiClient(model=model, api_key=api_key)
        self.api_key = self.client.api_key
        self.output_dir = Path(output_dir) if output_dir else Path("legacy/validation_pipeline/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        self.completeness_agent = ExperimentCompletenessAgent(gemini_client=self.client)
        self.consistency_agent = ExperimentConsistencyAgent(gemini_client=self.client)
        self.data_agent = DataValidationAgent(gemini_client=self.client)
        self.checklist_agent = ChecklistGeneratorAgent(gemini_client=self.client)
        self.material_agent = MaterialValidationAgent(gemini_client=self.client)
        self.evaluator_agent = EvaluatorValidationAgent(gemini_client=self.client)
    
    def validate_study(
        self,
        study_id: str,
        study_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        save_results: bool = True,
        num_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Validate a study.
        
        Args:
            study_id: Study ID (e.g., "study_001")
            study_path: Path to study directory (if None, constructs from study_id)
            config_path: Path to study config Python file
            save_results: Whether to save results to file
            
        Returns:
            Complete validation results
        """
        print(f"\n{'='*80}")
        print(f"Starting validation for {study_id}")
        print(f"{'='*80}\n")
        
        # Determine study path
        if study_path is None:
            study_path = Path("data/studies") / study_id
        else:
            study_path = Path(study_path)
        
        if not study_path.exists():
            raise FileNotFoundError(f"Study path not found: {study_path}")
        
        # Load documents
        print("Loading documents...")
        loader = DocumentLoader()
        documents = loader.load_study_files(study_path)

        # Pre-upload PDFs once so parallel agents don't each re-upload.
        # Stored in documents["uploaded_pdfs"] and reused by agents via BaseValidationAgent._get_pdf_files().
        documents["uploaded_pdfs"] = {}
        for pdf_name, pdf_data in documents.get("pdfs", {}).items():
            try:
                pdf_path = Path(pdf_data["path"])
                documents["uploaded_pdfs"][pdf_name] = self.client.upload_file(pdf_path)
            except Exception:
                continue
        
        # Load config code if available
        if config_path is None:
            config_path = Path(f"src/studies/{study_id}_config.py")
        
        if config_path.exists():
            print(f"Loading config code from {config_path}...")
            documents["config_code"] = loader.load_python_file(config_path)
        else:
            print(f"Warning: Config file not found at {config_path}")
            documents["config_code"] = ""
        
        # Run validation agents
        results = {
            "study_id": study_id,
            "validation_timestamp": datetime.now().isoformat(),
            "study_path": str(study_path),
        }
        
        # Parallelizable agents (independent)
        # Checklist must run last because it depends on previous results.
        print(f"\nRunning validation agents (parallel={num_workers > 1}, workers={num_workers})...")
        agent_jobs = {}

        def _run_agent(name: str):
            print(f"  → Started {name}...")
            # Use shared client. genai.Client is generally thread-safe.
            # Creating new clients per thread can cause issues with uploaded file visibility.
            if name == "completeness":
                return ExperimentCompletenessAgent(gemini_client=self.client).validate(documents)
            if name == "consistency":
                return ExperimentConsistencyAgent(gemini_client=self.client).validate(documents)
            if name == "data_validation":
                return DataValidationAgent(gemini_client=self.client).validate(documents)
            if name == "materials_validation":
                return MaterialValidationAgent(gemini_client=self.client).validate(documents)
            if name == "evaluator_validation":
                return EvaluatorValidationAgent(gemini_client=self.client).validate(documents)
            raise ValueError(f"Unknown agent name: {name}")

        agent_names = [
            "completeness",
            "consistency",
            "data_validation",
            "materials_validation",
            "evaluator_validation",
        ]

        if num_workers <= 1:
            for name in agent_names:
                print(f"\nRunning {name}...")
                try:
                    results[name] = _run_agent(name)
                    print(f"✓ {name} completed")
                except Exception as e:
                    print(f"✗ Error in {name}: {e}")
                    results[name] = {"error": str(e)}
        else:
            import time
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                future_map = {}
                for name in agent_names:
                    future_map[ex.submit(_run_agent, name)] = name
                    # Small throttle between agent starts to avoid hitting rate limits 
                    # and ensure sequential logging of start messages.
                    time.sleep(1) 
                
                for fut in as_completed(future_map):
                    name = future_map[fut]
                    try:
                        results[name] = fut.result()
                        print(f"✓ {name} completed")
                    except Exception as e:
                        print(f"✗ Error in {name}: {e}")
                        results[name] = {"error": str(e)}
        
        # Generate Checklist (depends on earlier results)
        print("\nGenerating Validation Checklist...")
        try:
            checklist_result = self.checklist_agent.validate(
                documents,
                previous_results={
                    "completeness": results.get("completeness", {}),
                    "consistency": results.get("consistency", {}),
                    "data_validation": results.get("data_validation", {}),
                }
            )
            results["checklist"] = checklist_result
            print("✓ Checklist generation completed")
        except Exception as e:
            print(f"✗ Error in checklist generation: {e}")
            results["checklist"] = {"error": str(e)}
        
        # Save results
        if save_results:
            output_file = self.output_dir / f"{study_id}_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_file}")
            
            # Also save a human-readable summary
            summary_file = self.output_dir / f"{study_id}_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            self._save_summary(results, summary_file)
            print(f"✓ Summary saved to: {summary_file}")

            # Generate Stage 2 Regeneration Instructions (main output for generation pipeline)
            modification_plan = self._collect_modifications(results)
            stage2_instructions = self._generate_stage2_instructions(results, modification_plan)
            instructions_file = self.output_dir / f"{study_id}_stage2_regeneration_instructions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(instructions_file, 'w', encoding='utf-8') as f:
                json.dump(stage2_instructions, f, indent=2, ensure_ascii=False)
            print(f"✓ Stage 2 regeneration instructions saved to: {instructions_file}")
        
        print(f"\n{'='*80}")
        print(f"Validation completed for {study_id}")
        print(f"{'='*80}\n")
        
        return results
    
    def _save_summary(self, results: Dict[str, Any], output_file: Path):
        """Save human-readable summary"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Validation Summary for {results['study_id']}\n\n")
            f.write(f"**Validation Date:** {results['validation_timestamp']}\n\n")
            f.write(f"**Study Path:** {results['study_path']}\n\n")
            
            # Completeness summary
            if "completeness" in results and "results" in results["completeness"]:
                comp = results["completeness"]["results"]
                if "completeness_summary" in comp:
                    f.write("## Experiment Completeness\n\n")
                    f.write(f"- **Completeness Score:** {comp['completeness_summary'].get('completeness_score', 'N/A')}\n")
                    f.write(f"- **Total Experiments:** {comp['completeness_summary'].get('total_experiments', 'N/A')}\n")
                    f.write(f"- **Included Experiments:** {comp['completeness_summary'].get('included_experiments', 'N/A')}\n")
                    f.write(f"\n{comp['completeness_summary'].get('completeness_notes', '')}\n\n")
            
            # Consistency summary
            if "consistency" in results and "results" in results["consistency"]:
                cons = results["consistency"]["results"]
                if "consistency_summary" in cons:
                    f.write("## Experimental Setup Consistency\n\n")
                    f.write(f"- **Consistency Score:** {cons['consistency_summary'].get('consistency_score', 'N/A')}\n")
                    f.write(f"- **Consistent Aspects:** {cons['consistency_summary'].get('consistent_aspects', 'N/A')}\n")
                    f.write(f"- **Total Aspects Checked:** {cons['consistency_summary'].get('total_aspects_checked', 'N/A')}\n")
                    f.write(f"\n{cons['consistency_summary'].get('overall_assessment', '')}\n\n")
            
            # Data validation summary
            if "data_validation" in results and "results" in results["data_validation"]:
                data = results["data_validation"]["results"]
                if "validation_summary" in data:
                    f.write("## Human Data Validation\n\n")
                    f.write(f"- **Data Accuracy Score:** {data['validation_summary'].get('data_accuracy_score', 'N/A')}\n")
                    f.write(f"- **Matching Data Points:** {data['validation_summary'].get('matching_data_points', 'N/A')}\n")
                    f.write(f"- **Total Data Points Checked:** {data['validation_summary'].get('total_data_points_checked', 'N/A')}\n")
                    f.write(f"\n{data['validation_summary'].get('overall_assessment', '')}\n\n")
            
            # Checklist summary
            if "checklist" in results and "results" in results["checklist"]:
                checklist = results["checklist"]["results"]
                if "checklist_summary" in checklist:
                    f.write("## Validation Checklist\n\n")
                    f.write(f"- **Total Items:** {checklist['checklist_summary'].get('total_items', 'N/A')}\n")
                    f.write(f"- **Critical Items:** {checklist['checklist_summary'].get('critical_items', 'N/A')}\n")
                    f.write(f"- **Estimated Validation Time:** {checklist['checklist_summary'].get('estimated_validation_time', 'N/A')}\n\n")

            # RECOMMENDED ACTIONS SECTION
            f.write("## 🛠 Recommended File Modifications\n\n")
            actions = self._get_recommended_actions(results)
            if not actions:
                f.write("✅ No specific file modifications recommended based on current validation.\n\n")
            else:
                for file_name, file_actions in actions.items():
                    f.write(f"### `{file_name}`\n")
                    for action in file_actions:
                        f.write(f"- {action}\n")
                    f.write("\n")

    def _get_recommended_actions(self, results: Dict[str, Any]) -> Dict[str, list]:
        """Aggregate issues from all agents and map them to specific files"""
        actions = {}
        study_id = results.get("study_id")
        
        def add_action(file, msg):
            if file not in actions: actions[file] = []
            if msg not in actions[file]: actions[file].append(msg)

        # 1. Completeness Issues
        if "completeness" in results and "results" in results["completeness"]:
            comp = results["completeness"]["results"]
            missing = comp.get("completeness_summary", {}).get("missing_experiments", [])
            if missing:
                msg = f"Missing experiments detected: {', '.join(missing)}. Please update the benchmark to include these."
                add_action("specification.json", msg)
                add_action("ground_truth.json", msg)
                for exp_id in missing:
                    # Specific sub-study ID usually maps to materials/{id}.json
                    clean_id = exp_id.lower().replace(" ", "_")
                    add_action(f"materials/{clean_id}.json", f"Missing material file for {exp_id}.")

        # 2. Consistency Issues
        if "consistency" in results and "results" in results["consistency"]:
            cons = results["consistency"]["results"]
            aspects = cons.get("comparison_by_aspect", {})
            for aspect, data in aspects.items():
                if not data.get("consistent", True):
                    msg = f"Inconsistency in **{aspect}**: {data.get('notes', 'Details not provided')}"
                    if aspect in ["participants", "procedure", "conditions"]:
                        add_action("specification.json", msg)
                    elif aspect == "materials":
                        # Try to find which specific material file is referenced
                        # If the agent mentioned a sub_study in the notes or results, we use it
                        benchmark_text = data.get('benchmark', '')
                        # Heuristic: if we can't find a specific file, we mention materials/
                        add_action("materials/", f"Check instructions/scenario: {msg}")
                    elif aspect == "measures" or aspect == "analyses":
                        add_action("ground_truth.json", msg)

        # 3. Data Validation Issues
        if "data_validation" in results and "results" in results["data_validation"]:
            data_val = results["data_validation"]["results"]
            
            # Participant sample size mismatch
            part_val = data_val.get("participant_data_validation", {})
            for key, val in part_val.items():
                if isinstance(val, dict) and not val.get("match", True):
                    add_action("specification.json", f"Participant {key} mismatch: Paper says {val.get('paper')}, Benchmark has {val.get('benchmark')}")
            
            # Experimental data mismatches
            exp_val = data_val.get("experimental_data_validation", [])
            for item in exp_val:
                if not item.get("match", True) and not item.get("acceptable", True):
                    exp_id = item.get('experiment_id', 'unknown')
                    msg = f"Data mismatch in **{exp_id}** ({item.get('metric_name')}): Paper={item.get('paper_value')}, Benchmark={item.get('benchmark_value')}"
                    add_action("ground_truth.json", msg)
                    
                    # Also flag the specific material file if it's an item-level issue
                    if "item" in item.get('metric_name', '').lower() or "mean" in item.get('metric_name', '').lower():
                        clean_id = exp_id.lower().replace(" ", "_")
                        add_action(f"materials/{clean_id}.json", f"Verify item settings or descriptions for {exp_id} due to result mismatch.")

            # Critical issues
            for issue in data_val.get("critical_issues", []):
                msg = f"[{issue.get('severity', 'high').upper()}] {issue.get('issue')} - *Rec: {issue.get('recommendation')}*"
                # If the issue context contains a sub-study name, try to map it
                issue_text = issue.get('issue', '').lower()
                target_file = "ground_truth.json" # Default
                
                # Check for common sub-study patterns in the issue text
                if "study" in issue_text or "experiment" in issue_text:
                    # Heuristic mapping
                    import re
                    match = re.search(r'(study|experiment|sub_study)[\s_]*(\d+|[a-z_]+)', issue_text)
                    if match:
                        found_id = match.group(0).replace(" ", "_")
                        target_file = f"materials/{found_id}.json"
                
                add_action(target_file, msg)

        return actions

    def _collect_modifications(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect file modifications from all agents into a single plan usable by FileModifier.
        """
        plan = {
            "study_id": results.get("study_id"),
            "generated_at": datetime.now().isoformat(),
            "file_modifications": []
        }

        def extend_from_agent(agent_key: str):
            agent_res = results.get(agent_key, {})
            res = agent_res.get("results") if isinstance(agent_res, dict) else {}
            mods = res.get("file_modifications", [])
            if isinstance(mods, list):
                plan["file_modifications"].extend(mods)

        # From existing agents
        for key in ["completeness", "consistency", "data_validation", "materials_validation", "evaluator_validation", "checklist"]:
            extend_from_agent(key)

        # Also fold in recommended actions (legacy) as generic modifications
        recommended = self._get_recommended_actions(results)
        for file_path, msgs in recommended.items():
            for msg in msgs:
                plan["file_modifications"].append({
                    "file": file_path,
                    "reason": msg,
                    "change_type": "review",
                    "proposed_content": ""
                })

        return plan

    def _generate_stage2_instructions(self, results: Dict[str, Any], modification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specific instructions for Generation Stage 2 (Extraction) 
        based on validation findings.
        """
        instructions = {
            "study_id": results.get("study_id"),
            "generated_at": datetime.now().isoformat(),
            "missing_experiments": [],
            "exact_stats_needed": [],  # NEW: Specific section for missing exact statistics
            "data_corrections": [],
            "material_issues": [],
            "evaluator_issues": [],
            "other_feedback": []
        }

        # 1. Missing Experiments (from completeness results)
        comp = results.get("completeness", {}).get("results", {})
        missing = comp.get("completeness_summary", {}).get("missing_experiments", [])
        if missing:
            instructions["missing_experiments"] = missing

        # 2. Map file modifications to categories
        for mod in modification_plan.get("file_modifications", []):
            target_file = mod.get("file", "").lower()
            reason = mod.get("reason", "").lower()
            entry = {
                "reason": mod.get("reason"),
                "suggested_fix": mod.get("proposed_content")
            }

            # Check for exact statistics related issues (priority categorization)
            if any(kw in reason for kw in ["exact", "t-stat", "t(", "f(", "bf", "p-value"]):
                instructions["exact_stats_needed"].append(entry)
            elif "ground_truth.json" in target_file:
                instructions["data_corrections"].append(entry)
            elif "materials/" in target_file:
                instructions["material_issues"].append(entry)
            elif "evaluator" in target_file:
                instructions["evaluator_issues"].append(entry)
            else:
                instructions["other_feedback"].append(entry)

        return instructions

