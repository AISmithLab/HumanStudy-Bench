"""
Document Loader for Validation Pipeline

Loads PDF files, markdown files, and Python configuration files.
PDFs are stored as file paths for direct upload to Gemini API.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import PyPDF2


class DocumentLoader:
    """Loads various document types for validation"""
    
    @staticmethod
    def get_pdf_pages(pdf_path: Path) -> List[int]:
        """
        Get total number of pages in PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page numbers (1-indexed)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return list(range(1, len(pdf_reader.pages) + 1))
        except Exception as e:
            raise RuntimeError(f"Error reading PDF {pdf_path}: {e}")
    
    @staticmethod
    def load_markdown(md_path: Path) -> str:
        """
        Load markdown file content.
        
        Args:
            md_path: Path to markdown file
            
        Returns:
            File content as string
        """
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_json(json_path: Path) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Parsed JSON as dictionary
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_python_file(py_path: Path) -> str:
        """
        Load Python file content.
        
        Args:
            py_path: Path to Python file
            
        Returns:
            File content as string
        """
        if not py_path.exists():
            raise FileNotFoundError(f"Python file not found: {py_path}")
        
        with open(py_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_study_files(study_path: Path) -> Dict[str, Any]:
        """
        Load all relevant files for a study.
        PDFs are stored as file paths (not extracted text) for direct upload to Gemini API.
        
        Args:
            study_path: Path to study directory
            
        Returns:
            Dictionary containing all loaded documents
        """
        study_path = Path(study_path)
        
        # Find PDF files - store paths and page info instead of extracting text
        pdf_files = list(study_path.glob("*.pdf"))
        pdf_info = {}
        for pdf_file in pdf_files:
            try:
                pages = DocumentLoader.get_pdf_pages(pdf_file)
                pdf_info[pdf_file.name] = {
                    "path": str(pdf_file.absolute()),
                    "page_count": len(pages),
                    "pages": pages,
                }
            except Exception as e:
                print(f"Warning: Could not read PDF {pdf_file.name}: {e}")
        
        # Load STUDY_INFO.md
        study_info = None
        study_info_path = study_path / "STUDY_INFO.md"
        if study_info_path.exists():
            study_info = DocumentLoader.load_markdown(study_info_path)
        
        # Load markdown files (paper summaries)
        md_files = list(study_path.glob("*.md"))
        md_content = {}
        for md_file in md_files:
            if md_file.name != "STUDY_INFO.md":
                md_content[md_file.name] = DocumentLoader.load_markdown(md_file)
        
        # Load JSON files
        json_files = {}
        for json_file in ["metadata.json", "specification.json", "ground_truth.json"]:
            json_path = study_path / json_file
            if json_path.exists():
                json_files[json_file] = DocumentLoader.load_json(json_path)
        
        # Try to find and load stage1_filter.json from generation_pipeline/outputs
        # by matching paper_title from metadata.json
        metadata = json_files.get("metadata.json")
        if metadata and "title" in metadata:
            study_title = metadata["title"].strip().lower()
            outputs_dir = Path("generation_pipeline/outputs")
            if outputs_dir.exists():
                for f in outputs_dir.glob("*_stage1_filter.json"):
                    try:
                        stage1_data = DocumentLoader.load_json(f)
                        if stage1_data.get("paper_title", "").strip().lower() == study_title:
                            json_files["stage1_filter.json"] = stage1_data
                            break
                    except Exception:
                        continue

        return {
            "pdfs": pdf_info,  # Now contains file paths and page info, not text
            "study_info": study_info,
            "markdown": md_content,
            "json": json_files,
            "study_path": str(study_path),
        }

