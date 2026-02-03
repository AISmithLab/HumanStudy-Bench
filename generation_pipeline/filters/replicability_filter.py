"""
Replicability Filter - Stage 1

Filters papers based on whether they can be replicated using LLM agents.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.filters.base_filter import BaseFilter
from generation_pipeline.utils.document_loader import DocumentLoader
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


# Max PDF text length to avoid context limits (chars)
PDF_TEXT_MAX_CHARS = 400000


class ReplicabilityFilter(BaseFilter):
    """Filter papers for LLM replicability"""

    def process(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process PDF and determine replicability.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with filter results:
            {
                "paper_id": str,
                "paper_title": str,
                "paper_authors": List[str],
                "paper_abstract": str,
                "experiments": [...],
                "overall_replicable": bool,
                "confidence": float,
                "notes": str
            }
        """
        loader = DocumentLoader()
        pdf_info = loader.get_pdf_pages(pdf_path)
        pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
        prompt = self._build_prompt(pdf_path.name, len(pdf_info))
        full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"

        try:
            response = self.client.generate_content(prompt=full_prompt)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM API: {e}. Check API key and provider env (e.g. GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY).")
        
        if response is None:
            raise ValueError("LLM returned None response. Check API key and network connection.")
        
        # Parse response
        result = self._parse_response(response, pdf_path)
        
        return result
    
    def _build_prompt(self, pdf_name: str, num_pages: int) -> str:
        """Build prompt for LLM"""
        return f"""Analyze the research paper in the attached PDF file: {pdf_name} ({num_pages} pages)

Your task is to:
1. Extract the paper's title, authors, and abstract
2. Identify all experiments/studies in the paper
3. For each experiment, determine if it can be replicated using LLM agents
4. Assess whether the paper contains self-contained materials (all test items, questions, stimuli, and raw data needed for replication)

EXCLUSION CRITERIA (experiment is NOT replicable if):
- Requires visual input (images, videos, visual stimuli)
- Requires time perception/measurement (reaction time, duration judgments)
- Requires real-world physical actions or authentic in-person/social interaction that cannot be faithfully simulated in text-only prompts
- Requires "authentic conflict" / real behavioral compliance (e.g., actually wearing a sign in public, real-world tasks on campus) as the measured dependent variable
- Participant profile is too vague to construct (cannot determine demographics, recruitment source, etc.)
- Has no quantitative/statistical data

SELF-CONTAINED MATERIALS ASSESSMENT:
An experiment is considered to have "self-contained materials" if the paper includes:
- All test items/questions/stimuli used in the experiment (e.g., all 50 quotations, all 16 prose passages)
- Complete instructions given to participants
- Raw data or sufficient statistical information to reconstruct the experiment

IMPORTANT: An experiment can be REPLICABLE even if it lacks self-contained materials. For example:
- A paper may describe an experiment methodologically but reference external materials (e.g., "50 quotations from Lorge (1936)")
- A paper may re-analyze data from another study without including the original test items
- In such cases, mark the experiment as "replicable: YES" but set "has_self_contained_materials: false"
- Note in the experiment's notes what materials are missing and whether they can be reconstructed or need to be obtained from other sources

For each experiment, provide:
- Experiment name/number
- Input: What participants receive/see
- Participants: Brief description of participant characteristics
- Output: What is measured/collected
- Replicable: YES/NO/UNCERTAIN
- Has Self-Contained Materials: true/false (whether all test items, questions, and data are present in the paper)
- Exclusion Reasons: If not replicable, list the reasons
- Missing Materials: If replicable but lacks self-contained materials, describe what is missing (e.g., "test items not included, referenced from Lorge 1936")

Provide your analysis in JSON format:
{{
    "paper_title": "Title of the paper",
    "paper_authors": ["Author 1", "Author 2", ...],
    "paper_abstract": "Full abstract text",
    "experiments": [
        {{
            "experiment_id": "Experiment 1",
            "experiment_name": "Name or description",
            "input": "What participants receive/see",
            "participants": "Brief description",
            "output": "What is measured/collected",
            "replicable": "YES/NO/UNCERTAIN",
            "has_self_contained_materials": true/false,
            "exclusion_reasons": ["reason1", "reason2"] or [],
            "missing_materials": "Description of missing materials if has_self_contained_materials is false, or empty string"
        }}
    ],
    "overall_replicable": true/false,
    "confidence": 0.0-1.0,
    "notes": "Additional notes or observations. If experiments are replicable but lack self-contained materials, note this here and explain whether materials can be reconstructed or need external sources."
}}

IMPORTANT: 
- Only include experiments that have quantitative/statistical data. If an experiment is purely qualitative or lacks statistical analysis, mark it as NOT replicable.
- Distinguish between "replicable" (methodologically feasible) and "has_self_contained_materials" (all materials present in paper). An experiment can be replicable even without self-contained materials."""
    
    def _parse_response(self, response: str, pdf_path: Path) -> Dict[str, Any]:
        """Parse LLM response"""
        if response is None:
            raise ValueError("LLM response is None")
        
        # Extract JSON from response (may have markdown code blocks)
        response_text = response.strip() if isinstance(response, str) else str(response).strip()
        
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
        
        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {}
        
        # Add paper_id (derived from PDF filename)
        paper_id = pdf_path.stem.replace(' ', '_').replace('-', '_').lower()
        result['paper_id'] = paper_id
        
        return result

