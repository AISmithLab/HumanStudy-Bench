#!/usr/bin/env python3
"""Thin wrapper: delegates to scripts/advanced/legacy_generate_results_table_pas_ecs.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from advanced.legacy_generate_results_table_pas_ecs import main

if __name__ == "__main__":
    sys.exit(main() or 0)
