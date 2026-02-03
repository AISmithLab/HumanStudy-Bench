#!/usr/bin/env python3
"""Thin wrapper: delegates to scripts/advanced/generate_production_results.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from advanced.generate_production_results import main

if __name__ == "__main__":
    sys.exit(main() or 0)
