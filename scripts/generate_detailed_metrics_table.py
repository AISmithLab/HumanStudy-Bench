#!/usr/bin/env python3
"""Thin wrapper: delegates to scripts/advanced/generate_detailed_metrics_table.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from advanced.generate_detailed_metrics_table import main

if __name__ == "__main__":
    sys.exit(main() or 0)
