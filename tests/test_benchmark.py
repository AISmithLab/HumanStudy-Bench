# HumanStudyBench Tests

import pytest
from pathlib import Path


def test_import():
    """Test that main modules can be imported."""
    from src.core.benchmark import HumanStudyBench
    from src.core.study import Study
    from src.agents.base_agent import BaseAgent
    
    assert HumanStudyBench is not None
    assert Study is not None
    assert BaseAgent is not None


def test_data_directory_exists():
    """Test that data directory exists."""
    data_dir = Path("data")
    assert data_dir.exists()
    assert (data_dir / "registry.json").exists()
    assert (data_dir / "schemas").exists()
    assert (data_dir / "studies").exists()
