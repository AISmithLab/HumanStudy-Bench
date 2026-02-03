"""
Utility functions for safe file I/O operations.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(file_path: Path, data: Any, indent: int = 2, ensure_ascii: bool = False, encoding: str = 'utf-8', errors: str = 'replace') -> None:
    """
    Atomically write JSON data to a file.
    
    This function writes to a temporary file first, then atomically replaces
    the target file. This prevents corruption if the process is interrupted
    mid-write or if multiple processes try to write simultaneously.
    
    Args:
        file_path: Path to the target JSON file
        data: Data to serialize to JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
        encoding: File encoding (default: 'utf-8')
        errors: Error handling mode (default: 'replace')
    
    Raises:
        OSError: If file operations fail
        TypeError: If data is not JSON serializable
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory as target
    # This ensures atomic replace works even across filesystems
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.json.tmp',
        dir=str(file_path.parent),
        text=True
    )
    
    try:
        # Write JSON to temporary file
        with os.fdopen(temp_fd, 'w', encoding=encoding, errors=errors) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Atomically replace target file with temporary file
        os.replace(temp_path, str(file_path))
    except Exception:
        # Clean up temporary file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

