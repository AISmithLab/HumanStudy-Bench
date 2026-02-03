"""
Main script to run validation pipeline for a study
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path so we can import legacy.validation_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from legacy.validation_pipeline.pipeline import ValidationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Validate a study implementation against the original paper"
    )
    parser.add_argument(
        "study_id",
        type=str,
        help="Study ID to validate (e.g., study_001)",
    )
    parser.add_argument(
        "--study-path",
        type=str,
        default=None,
        help="Path to study directory (default: data/studies/{study_id})",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to study config Python file (default: src/studies/{study_id}_config.py)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-3-flash-preview",
        help="Gemini model to use (default: models/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="legacy/validation_pipeline/outputs",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Max parallel workers within a study (default: 1). "
             "Runs independent validation agents concurrently, then checklist last.",
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ValidationPipeline(
        model=args.model,
        output_dir=Path(args.output_dir),
    )
    
    # Run validation
    study_path = Path(args.study_path) if args.study_path else None
    config_path = Path(args.config_path) if args.config_path else None
    
    results = pipeline.validate_study(
        study_id=args.study_id,
        study_path=study_path,
        config_path=config_path,
        save_results=not args.no_save,
        num_workers=max(1, int(args.num_workers)),
    )
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if "completeness" in results and "results" in results["completeness"]:
        comp = results["completeness"]["results"]
        if "completeness_summary" in comp:
            print(f"\nCompleteness Score: {comp['completeness_summary'].get('completeness_score', 'N/A')}")
    
    if "consistency" in results and "results" in results["consistency"]:
        cons = results["consistency"]["results"]
        if "consistency_summary" in cons:
            print(f"Consistency Score: {cons['consistency_summary'].get('consistency_score', 'N/A')}")
    
    if "data_validation" in results and "results" in results["data_validation"]:
        data = results["data_validation"]["results"]
        if "validation_summary" in data:
            print(f"Data Accuracy Score: {data['validation_summary'].get('data_accuracy_score', 'N/A')}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

