"""
Study-specific configurations.

This module imports all study configurations to ensure they are registered.
"""

# Dynamically import available study configs
import os
import importlib

__all__ = []

for i in range(1, 13):
    study_id = f"study_{i:03d}"
    config_file = f"src.studies.{study_id}_config"
    config_class = f"StudyStudy{i:03d}Config"

    try:
        module = importlib.import_module(config_file)
        cls = getattr(module, config_class)
        __all__.append(config_class)
        globals()[config_class] = cls
    except (ImportError, AttributeError):
        # Skip if config file doesn't exist yet
        pass

__all__ = [
    'StudyStudy001Config',
    'StudyStudy002Config',
    'StudyStudy003Config',
    'StudyStudy004Config',
    'StudyStudy005Config',
    'StudyStudy006Config',
    'StudyStudy007Config',
    'StudyStudy008Config',
    'StudyStudy009Config',
    'StudyStudy010Config',
    'StudyStudy011Config',
    'StudyStudy012Config',
]

