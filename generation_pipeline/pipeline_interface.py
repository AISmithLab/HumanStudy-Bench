"""
Unified pipeline interface: stages 1-6 + final.

Stages:
  1 - Replicability filter (PDF → stage1 review)
  2 - Study & data extraction (stage1 → stage2 review)
  3 - Generate study files (stage2 → data/studies/{study_id}/)
  4 - Generate study config (stage2 → src/studies/{study_id}_config.py)
  5 - Simulation (LLM agents → full_benchmark.json)
  6 - Evaluation (full_benchmark → evaluation_results.json, detailed_stats.csv)
  final - Finding explanations (evaluation + metadata → finding_explanations.json/.md)
"""

from pathlib import Path
from typing import Any, Optional

# Re-export pipeline and run stages via the main pipeline class.
# Actual execution is in run.py; this module documents the interface.
__all__ = ["STAGES"]

STAGES = {
    1: "Filter (replicability)",
    2: "Extraction (study & data)",
    3: "Generate study files",
    4: "Generate study config",
    5: "Simulation",
    6: "Evaluation",
    "final": "Finding explanations",
}
