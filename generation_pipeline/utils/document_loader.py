"""
Document Loader - Reuses legacy.validation_pipeline's DocumentLoader
"""

import sys
from pathlib import Path

# Add repo root to path so legacy is importable
_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root))

from legacy.validation_pipeline.utils.document_loader import DocumentLoader

__all__ = ["DocumentLoader"]

