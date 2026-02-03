"""
File Modifier - Use LLM to modify files based on context
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm.factory import get_client
from generation_pipeline.utils.pdf_extractor import extract_pdf_text

PDF_TEXT_MAX_CHARS = 400000


class FileModifier:
    """Use LLM to modify files based on context"""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "models/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize file modifier.

        Args:
            provider: One of gemini, openai, anthropic, xai, openrouter
            model: Model name
            api_key: Optional API key
            api_base: Optional API base URL
        """
        self.client = get_client(provider=provider, model=model, api_key=api_key, api_base=api_base)
    
    def modify_files(
        self,
        file_paths: List[Path],
        context: str,
        output_dir: Optional[Path] = None,
        apply_changes: bool = False,
        pdf_path: Optional[Path] = None,
        auto_find_pdf: bool = True
    ) -> Dict[str, Any]:
        """
        Modify files using LLM based on context.
        
        Args:
            file_paths: List of file paths to modify
            context: Description of what changes to make
            output_dir: Directory to save modified files (default: create .modified versions)
            apply_changes: If True, apply changes directly. If False, save to new files.
            pdf_path: Optional path to PDF file (original paper)
            auto_find_pdf: If True, automatically find PDF in study directory
            
        Returns:
            Dictionary with modification results
        """
        # Read all files
        file_contents = {}
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            file_contents[str(file_path)] = file_path.read_text(encoding='utf-8')
        
        # Auto-find PDF if needed
        if auto_find_pdf and pdf_path is None:
            pdf_path = self._find_pdf_for_files(file_paths)
        
        # Build prompt
        prompt = self._build_prompt(file_contents, context, pdf_path)
        
        # Call LLM (with PDF if available)
        try:
            if pdf_path and pdf_path.exists():
                pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
                full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"
                response = self.client.generate_content(prompt=full_prompt)
            else:
                response = self.client.generate_content(prompt=prompt)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM API: {e}. Check API key and provider env.")
        
        if response is None:
            raise ValueError("LLM returned None response. Check API key and network connection.")
        
        # Parse response
        modifications = self._parse_response(response, file_paths)
        
        # Apply or save modifications
        results = {}
        for file_path in file_paths:
            file_str = str(file_path)
            if file_str in modifications:
                modified_content = modifications[file_str]
                
                if apply_changes:
                    # Backup original
                    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                    file_path.rename(backup_path)
                    
                    # Write modified content
                    file_path.write_text(modified_content, encoding='utf-8')
                    results[file_str] = {
                        "status": "applied",
                        "backup": str(backup_path),
                        "modified": str(file_path)
                    }
                else:
                    # Save to new file
                    if output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / file_path.name
                    else:
                        output_path = file_path.with_suffix(file_path.suffix + '.modified')
                    
                    output_path.write_text(modified_content, encoding='utf-8')
                    results[file_str] = {
                        "status": "saved",
                        "original": str(file_path),
                        "modified": str(output_path)
                    }
            else:
                results[file_str] = {
                    "status": "no_changes",
                    "message": "LLM did not provide modifications for this file"
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "pdf_used": str(pdf_path) if pdf_path else None,
            "files": results
        }
    
    def _find_pdf_for_files(self, file_paths: List[Path]) -> Optional[Path]:
        """Auto-find PDF file related to the files being modified"""
        # Try to find study directory from file paths
        for file_path in file_paths:
            # Check if file is in a study directory
            parts = file_path.parts
            if 'studies' in parts:
                study_idx = parts.index('studies')
                if study_idx + 1 < len(parts):
                    study_id = parts[study_idx + 1]
                    study_dir = Path(*parts[:study_idx + 2])
                    
                    # Look for PDF in study directory
                    pdf_files = list(study_dir.glob("*.pdf"))
                    if pdf_files:
                        return pdf_files[0]
                    
                    # Also check backup directory
                    backup_study_dir = Path("backup") / "data" / "studies" / study_id
                    if backup_study_dir.exists():
                        pdf_files = list(backup_study_dir.glob("*.pdf"))
                        if pdf_files:
                            return pdf_files[0]
        
        return None
    
    def _build_prompt(self, file_contents: Dict[str, str], context: str, pdf_path: Optional[Path] = None) -> str:
        """Build prompt for LLM"""
        files_section = ""
        for file_path, content in file_contents.items():
            files_section += f"\n\n## File: {file_path}\n```\n{content}\n```"
        
        pdf_section = ""
        if pdf_path and pdf_path.exists():
            pdf_section = f"""

ORIGINAL PAPER PDF:
The original research paper PDF is attached. Please refer to it to ensure accuracy when making modifications.
PDF file: {pdf_path.name}
"""
        
        return f"""You are a code/file modification assistant. Your task is to modify the provided files based on the given context and the original research paper.

CONTEXT (what changes to make):
{context}
{pdf_section}
FILES TO MODIFY:
{files_section}

INSTRUCTIONS:
1. Understand the context and what changes are needed
2. Modify each file accordingly
3. Preserve the original structure and formatting as much as possible
4. Only make the changes requested in the context
5. If a file doesn't need changes, return it as-is

OUTPUT FORMAT:
Provide your response in JSON format:
{{
    "files": {{
        "{list(file_contents.keys())[0] if file_contents else "file_path"}": "modified content here",
        ...
    }},
    "summary": "Brief summary of changes made",
    "notes": "Any additional notes or warnings"
}}

IMPORTANT:
- Return the COMPLETE modified content for each file
- Maintain original file encoding and line endings
- Do not add explanations outside the JSON structure
- If a file should not be changed, return it exactly as provided
"""
    
    def _parse_response(self, response: str, file_paths: List[Path]) -> Dict[str, str]:
        """Parse LLM response"""
        response_text = response.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
        
        # Extract file modifications
        modifications = {}
        if 'files' in result:
            for file_path in file_paths:
                file_str = str(file_path)
                if file_str in result['files']:
                    modifications[file_str] = result['files'][file_str]
                elif file_path.name in result['files']:
                    modifications[file_str] = result['files'][file_path.name]
        
        return modifications

