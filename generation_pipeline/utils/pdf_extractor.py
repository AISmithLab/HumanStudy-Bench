"""
PDF text extraction for multi-provider pipeline (no upload_file dependency).
"""

from pathlib import Path
from typing import Optional


def extract_pdf_text(pdf_path: Path, max_chars: Optional[int] = None) -> str:
    """
    Extract text from PDF with page markers. Uses pypdf.

    Args:
        pdf_path: Path to PDF file
        max_chars: If set, truncate to this many characters (for context limits)

    Returns:
        Text with "--- Page N ---" markers between pages
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf required for PDF text extraction. Install with: pip install pypdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    parts = []
    total = 0
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        block = f"--- Page {i} ---\n{text}"
        if max_chars and total + len(block) > max_chars:
            remaining = max_chars - total - 50
            if remaining > 0:
                block = block[:remaining] + "\n[... truncated]"
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
