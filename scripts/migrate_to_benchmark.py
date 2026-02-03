#!/usr/bin/env python3
"""
Migrate study_001 to results/benchmark/ as baseline reference.

Usage:
    python scripts/migrate_to_benchmark.py --run-name batch_20260106
"""

import argparse
import json
import shutil
from pathlib import Path
import os

def get_benchmark_folder():
    """Get benchmark folder name from .env or use default."""
    # Try to load .env if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required
    
    return os.getenv("BENCHMARK_FOLDER", "benchmark")

def migrate_to_benchmark(source_run: str, study_id: str = "study_001"):
    """
    Migrate a study to benchmark folder.
    
    Args:
        source_run: Source run directory name (e.g., "batch_20260106")
        study_id: Study ID to migrate (default: study_001)
    """
    source_dir = Path("results/runs") / source_run / study_id
    if not source_dir.exists():
        print(f"❌ Source study not found: {source_dir}")
        return False
    
    benchmark_folder = get_benchmark_folder()
    benchmark_base = Path("results") / benchmark_folder
    benchmark_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Migrating {study_id} from {source_run} to benchmark...")
    print(f"  Benchmark folder: {benchmark_folder}")
    
    # Copy all config folders
    config_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    if not config_dirs:
        print(f"❌ No config folders found in {source_dir}")
        return False
    
    for config_dir in config_dirs:
        config_name = config_dir.name
        target_dir = benchmark_base / study_id / config_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Copying {config_name}...")
        
        # Copy all files
        for file in config_dir.glob("*"):
            if file.is_file():
                target_file = target_dir / file.name
                shutil.copy2(file, target_file)
                print(f"    ✓ {file.name}")
    
    # Create benchmark metadata
    metadata_file = benchmark_base / "benchmark_metadata.json"
    metadata = {
        "benchmark_folder": benchmark_folder,
        "source_run": source_run,
        "studies": [study_id],
        "note": "This is the baseline benchmark. All new runs will be compared against this."
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Migration complete!")
    print(f"   Benchmark: {benchmark_base}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Migrate study to benchmark folder")
    parser.add_argument(
        "--run-name",
        type=str,
        default="batch_20260106",
        help="Source run directory name"
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default="study_001",
        help="Study ID to migrate (default: study_001)"
    )
    
    args = parser.parse_args()
    
    success = migrate_to_benchmark(args.run_name, args.study_id)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()

