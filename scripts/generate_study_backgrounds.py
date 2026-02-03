"""
Generate and store Generative Agents-style backgrounds for study participants.

This script:
1. Loads study specification and metadata
2. Generates backgrounds using Gemini 3 for each participant
3. Stores backgrounds per study/participant/trial combination
4. Ensures each participant-profile pair is unique per experiment/trial
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.generate_agent_background import AgentBackgroundGenerator
from src.core.study import Study


class StudyBackgroundGenerator:
    """Generate and store backgrounds for study participants."""
    
    def __init__(
        self,
        model: str = "gemini-flash-3",
        api_key: Optional[str] = None,
        storage_dir: Path = Path("data/backgrounds")
    ):
        """
        Initialize the background generator.
        
        Args:
            model: LLM model to use (default: gemini-flash-3)
            api_key: API key (if None, reads from environment)
            storage_dir: Directory to store generated backgrounds
        """
        self.generator = AgentBackgroundGenerator(
            model=model,
            api_key=api_key,
            use_gemini=True
        )
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_for_study(
        self,
        study_path: Path,
        n_participants: Optional[int] = None,
        overwrite: bool = False,
        max_workers: int = 16
    ) -> Dict[str, Any]:
        """
        Generate backgrounds for all participants in a study.
        
        Args:
            study_path: Path to study directory (e.g., data/studies/study_001)
            n_participants: Number of participants to generate (if None, uses spec)
            overwrite: If True, regenerate existing backgrounds
            max_workers: Number of parallel workers (default: 16)
        
        Returns:
            Dict with generation statistics
        """
        study_path = Path(study_path)
        study_id = study_path.name
        
        # Load study data
        try:
            study = Study.load(study_path)
        except Exception as e:
            raise ValueError(f"Failed to load study {study_id}: {e}")
        
        metadata = study.metadata
        specification = study.specification
        
        # Calculate total trials using study config logic (same as pipeline.py)
        try:
            from src.core.study_config import get_study_config
            repo_root = Path(__file__).parent.parent
            study_config = get_study_config(study_id, study_path, specification)
            
            # Replicate the trial counting logic from pipeline.py
            by_sub_study_spec = specification.get("participants", {}).get("by_sub_study", {})
            if n_participants is not None:
                trials = study_config.create_trials(n_trials=n_participants)
            elif by_sub_study_spec:
                trials = study_config.create_trials(n_trials=None)
            else:
                n_def = specification.get('participants', {}).get('n') or 30
                trials = study_config.create_trials(n_trials=n_def)
            
            total_trials = len(trials)
            print(f"Calculated {total_trials} total trials/participants needed for {study_id} (using config logic)")
        except Exception as e:
            print(f"Warning: Could not calculate trials from config, falling back to basic logic: {e}")
            # Fallback to previous logic
            participant_spec = specification.get("participants", {})
            by_sub_study = participant_spec.get("by_sub_study", {})
            total_trials = 0
            if by_sub_study:
                for group_spec in by_sub_study.values():
                    n = group_spec.get("n", 0) if isinstance(group_spec, dict) else group_spec
                    total_trials += n
            if total_trials == 0:
                total_trials = participant_spec.get("n", 50) if n_participants is None else n_participants
            if n_participants is not None:
                total_trials = n_participants

        # Get trial groups for experimental context building
        trial_groups = []
        by_sub_study = specification.get("participants", {}).get("by_sub_study", {})
        if by_sub_study:
            for trial_group_id, group_spec in by_sub_study.items():
                n = group_spec.get("n", 0) if isinstance(group_spec, dict) else group_spec
                if n > 0 or total_trials > 156: # Include default groups if we know they exist
                    trial_groups.append({"trial_group_id": trial_group_id, "n": n})
        
        # Build experimental context
        experimental_context = self._build_experimental_context(metadata, specification, trial_groups)
        
        # Get participant demographics from spec
        participant_spec = specification.get("participants", {})
        age_range = participant_spec.get("age_range", [18, 65])
        age_mean = participant_spec.get("age_mean", (age_range[0] + age_range[1]) / 2)
        age_sd = participant_spec.get("age_sd", 5.0)
        gender_dist = participant_spec.get("gender_distribution", {"male": 50, "female": 50})
        education = participant_spec.get("recruitment_source", "college student")
        
        # Storage path for this study
        study_storage = self.storage_dir / study_id
        study_storage.mkdir(parents=True, exist_ok=True)
        
        # Helper lists for name generation
        FIRST_NAMES_MALE = ["James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua"]
        FIRST_NAMES_FEMALE = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"]
        LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzales", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

        def generate_name(gender):
            first = np.random.choice(FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE)
            last = np.random.choice(LAST_NAMES)
            return f"{first} {last}"

        # Generate profiles and backgrounds
        import numpy as np
        np.random.seed(42)  # For reproducibility
        
        generated = 0
        skipped = 0
        backgrounds = {}
        trial_counter = 0
        lock = threading.Lock()  # For thread-safe counters
        
        def generate_single_background(args):
            """Generate background for a single participant."""
            participant_id, trial_group_id, trial_number, trial_context, gender, real_name, age, n_trials = args
            
            # Check if already exists
            participant_storage = study_storage / f"participant_{participant_id:04d}.json"
            if participant_storage.exists() and not overwrite:
                with lock:
                    nonlocal skipped
                    skipped += 1
                try:
                    with open(participant_storage, 'r', encoding='utf-8') as f:
                        return participant_id, json.load(f), 'skipped'
                except:
                    return participant_id, None, 'error'
            
            # Generate background (NO experimental_context - only pure bio)
            try:
                background = self.generator.generate_background(
                    name=real_name,
                    age=age,
                    gender=gender,
                    education=education,
                    experimental_context=None,  # Do not include experimental context
                    additional_demographics={
                        "study_id": study_id,
                        "participant_id": participant_id,
                        "trial_group_id": trial_group_id,
                        "trial_number": trial_number
                    }
                )
                
                # Store background data
                background_data = {
                    "name": real_name,
                    "study_id": study_id,
                    "participant_id": participant_id,
                    "trial_group_id": trial_group_id,
                    "trial_number": trial_number,
                    "age": age,
                    "gender": gender,
                    "education": education,
                    "background": background,
                    "experimental_context": trial_context,
                    "generated_at": str(Path(__file__).stat().st_mtime)
                }
                
                # Save to file
                with open(participant_storage, 'w', encoding='utf-8') as f:
                    json.dump(background_data, f, indent=2, ensure_ascii=False)
                
                with lock:
                    nonlocal generated
                    generated += 1
                
                print(f"Generated background for {study_id} participant {participant_id:04d} (trial group: {trial_group_id}, trial {trial_number+1}/{n_trials})")
                return participant_id, background_data, 'generated'
                
            except Exception as e:
                print(f"Error generating background for participant {participant_id}: {e}")
                return participant_id, None, 'error'
        
        # Prepare all tasks
        tasks = []
        
        # Track which participant IDs we've assigned tasks for
        assigned_participant_ids = set()

        # 1. First, generate backgrounds for each trial group defined in spec
        if trial_groups:
            for trial_group in trial_groups:
                trial_group_id = trial_group["trial_group_id"]
                n_trials = trial_group["n"]
                
                if n_trials <= 0:
                    continue

                # Build trial-specific context
                trial_context = self._build_trial_group_context(
                    experimental_context, trial_group_id, trial_group
                )
                
                for i in range(n_trials):
                    participant_id = trial_counter
                    trial_counter += 1
                    assigned_participant_ids.add(participant_id)
                    
                    # Sample gender
                    total_gender = sum(gender_dist.values())
                    rand = np.random.random() * total_gender
                    cumsum = 0
                    gender = "male"
                    for g, count in gender_dist.items():
                        cumsum += count
                        if rand < cumsum:
                            gender = g
                            break
                    
                    # Generate name and age
                    real_name = generate_name(gender)
                    age = int(np.clip(np.random.normal(age_mean, age_sd), age_range[0], age_range[1]))
                    
                    tasks.append((
                        participant_id, trial_group_id, i, trial_context, gender, real_name, age, n_trials
                    ))

        # 2. If we still haven't reached total_trials, fill the rest (for default groups or extra trials)
        remaining = total_trials - trial_counter
        if remaining > 0:
            print(f"Adding {remaining} extra tasks to reach total_trials ({total_trials})")
            for i in range(remaining):
                participant_id = trial_counter
                trial_counter += 1
                
                # Sample gender
                total_gender = sum(gender_dist.values())
                rand = np.random.random() * total_gender
                cumsum = 0
                gender = "male"
                for g, count in gender_dist.items():
                    cumsum += count
                    if rand < cumsum:
                        gender = g
                        break
                
                real_name = generate_name(gender)
                age = int(np.clip(np.random.normal(age_mean, age_sd), age_range[0], age_range[1]))
                
                # Use a generic group ID if needed
                tasks.append((
                    participant_id, "default_group", i, experimental_context, gender, real_name, age, remaining
                ))
        
        # Execute in parallel
        print(f"Generating {len(tasks)} backgrounds with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_single_background, task): task for task in tasks}
            
            for future in as_completed(futures):
                participant_id, result, status = future.result()
                if result:
                    backgrounds[participant_id] = result
        
        # Save study-level index
        index_path = study_storage / "index.json"
        index_data = {
            "study_id": study_id,
            "total_trials": total_trials,
            "trial_groups": trial_groups,
            "generated": generated,
            "skipped": skipped,
            "participants": list(backgrounds.keys())
        }
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        return {
            "study_id": study_id,
            "generated": generated,
            "skipped": skipped,
            "total": total_trials,
            "trial_groups": len(trial_groups) if trial_groups else 0
        }
    
    def _build_experimental_context(
        self, 
        metadata: Dict[str, Any], 
        specification: Dict[str, Any],
        trial_groups: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build experimental context description from metadata and specification."""
        context_parts = []
        
        # Study title and description
        if metadata.get("title"):
            context_parts.append(f"Study: {metadata['title']}")
        
        if metadata.get("description"):
            desc = metadata.get("description", "")
            # No truncation - send full description
            context_parts.append(f"Description: {desc}")
        
        # Procedure
        procedure = specification.get("procedure", {})
        if isinstance(procedure, dict):
            steps = procedure.get("steps", [])
            if steps:
                context_parts.append("Procedure:")
                for i, step in enumerate(steps, 1):  # No limit - include all steps
                    context_parts.append(f"  {i}. {step}")
        
        # Design
        design = specification.get("design", {})
        if design.get("type"):
            context_parts.append(f"Design: {design['type']}")
        
        # Trial groups information
        if trial_groups:
            context_parts.append("")
            context_parts.append("Trial Groups:")
            for group in trial_groups:
                context_parts.append(f"  - {group['trial_group_id']}: {group['n']} participants")
        
        return "\n".join(context_parts)
    
    def _build_trial_group_context(
        self,
        base_context: str,
        trial_group_id: str,
        trial_group_spec: Dict[str, Any]
    ) -> str:
        """Build context specific to a trial group."""
        context_parts = [base_context]
        context_parts.append("")
        context_parts.append(f"Trial Group: {trial_group_id}")
        if isinstance(trial_group_spec, dict) and "n" in trial_group_spec:
            context_parts.append(f"This participant is part of {trial_group_id} with {trial_group_spec['n']} total participants.")
        return "\n".join(context_parts)
    
    def get_background(
        self,
        study_id: str,
        participant_id: int,
        trial_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a stored background.
        
        Args:
            study_id: Study identifier
            participant_id: Participant ID
            trial_id: Optional trial ID (for trial-specific backgrounds)
        
        Returns:
            Background data dict or None if not found
        """
        if trial_id:
            background_path = self.storage_dir / study_id / f"participant_{participant_id:04d}_trial_{trial_id}.json"
        else:
            background_path = self.storage_dir / study_id / f"participant_{participant_id:04d}.json"
        
        if not background_path.exists():
            return None
        
        with open(background_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate backgrounds for study participants")
    parser.add_argument("--study", type=str, required=True, help="Study ID (e.g., study_001)")
    parser.add_argument("--n-participants", type=int, help="Number of participants (default: from spec)")
    parser.add_argument("--model", type=str, default="gemini-flash-3", help="LLM model to use")
    parser.add_argument("--api-key", type=str, help="API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--storage-dir", type=str, default="data/backgrounds", help="Storage directory")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing backgrounds")
    parser.add_argument("--data-dir", type=str, default="data/studies", help="Studies directory")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    
    args = parser.parse_args()
    
    study_path = Path(args.data_dir) / args.study
    if not study_path.exists():
        print(f"Error: Study path not found: {study_path}")
        sys.exit(1)
    
    generator = StudyBackgroundGenerator(
        model=args.model,
        api_key=args.api_key,
        storage_dir=Path(args.storage_dir)
    )
    
    print(f"Generating backgrounds for {args.study}...")
    result = generator.generate_for_study(
        study_path=study_path,
        n_participants=args.n_participants,
        overwrite=args.overwrite,
        max_workers=args.max_workers
    )
    
    print(f"\nCompleted:")
    print(f"  Generated: {result['generated']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Total: {result['total']}")


if __name__ == "__main__":
    main()

