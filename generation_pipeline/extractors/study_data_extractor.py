"""
Study & Data Extractor - Stage 2

Extracts study information, phenomena, research questions, statistical data,
and participant profiles from papers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.extractors.base_extractor import BaseExtractor
from generation_pipeline.utils.document_loader import DocumentLoader
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 400000


class StudyDataExtractor(BaseExtractor):
    """Extract study and statistical data from papers"""

    def process(self, stage1_json: Dict[str, Any], pdf_path: Path, regeneration_instructions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract study data from PDF.

        Args:
            stage1_json: Results from stage1 filter
            pdf_path: Path to PDF file
            regeneration_instructions: Optional validation feedback from previous run

        Returns:
            Dictionary with extracted data (studies list, etc.)
        """
        loader = DocumentLoader()
        pdf_info = loader.get_pdf_pages(pdf_path)
        pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
        prompt = self._build_prompt(stage1_json, pdf_path.name, len(pdf_info), regeneration_instructions)
        full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"

        try:
            response = self.client.generate_content(prompt=full_prompt)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM API: {e}. Check API key and provider env.")
        
        if response is None:
            raise ValueError("LLM returned None response. Check API key and network connection.")
        
        # Parse response
        result = self._parse_response(response, stage1_json)
        
        if result is None:
            raise ValueError("Parsed result is None")
        
        return result
    
    def _build_prompt(self, stage1_json: Dict[str, Any], pdf_name: str, num_pages: int, regeneration_instructions: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for LLM"""
        experiments_info = json.dumps(stage1_json.get("experiments", []), indent=2)
        
        # Build validation feedback section if provided
        feedback_section = ""
        if regeneration_instructions:
            feedback_section = "\n\n" + "="*80 + "\n"
            feedback_section += "VALIDATION FEEDBACK FROM PREVIOUS EXTRACTION:\n"
            feedback_section += "="*80 + "\n"
            feedback_section += "Please address the following issues identified during validation:\n\n"
            
            if regeneration_instructions.get("missing_experiments"):
                feedback_section += f"MISSING EXPERIMENTS:\n"
                for exp in regeneration_instructions["missing_experiments"]:
                    feedback_section += f"  - {exp}\n"
                feedback_section += "\n"
            
            if regeneration_instructions.get("exact_stats_needed"):
                feedback_section += f"EXACT STATISTICS REQUIRED (CRITICAL):\n"
                for item in regeneration_instructions["exact_stats_needed"]:
                    feedback_section += f"  - {item.get('reason', '')}\n"
                    if item.get('suggested_fix'):
                        feedback_section += f"    Fix: {item['suggested_fix']}\n"
                feedback_section += "\n"
            
            if regeneration_instructions.get("data_corrections"):
                feedback_section += f"DATA CORRECTIONS NEEDED:\n"
                for item in regeneration_instructions["data_corrections"]:
                    feedback_section += f"  - {item.get('reason', '')}\n"
                    if item.get('suggested_fix'):
                        feedback_section += f"    Fix: {item['suggested_fix']}\n"
                feedback_section += "\n"
            
            feedback_section += "="*80 + "\n\n"
        
        return f"""Analyze the research paper in the attached PDF file: {pdf_name} ({num_pages} pages)

STAGE 1 FILTER RESULTS:
{experiments_info}
{feedback_section}
Extract complete information for each replicable experiment/study to enable replication and evaluation.

EXTRACTION REQUIREMENTS:
1. Label each finding as "Finding 1", "Finding 2", etc. (or use paper's notation like "F1", "F2")
2. Extract all statistical tests for each finding (significant, non-significant, marginal, interactions, follow-ups)
3. Include complete raw data for each test (means, SDs, sample sizes, differences)

For EACH study/experiment, extract:

1. STUDY STRUCTURE:
   - Study ID, name, phenomenon
   - Findings: List all findings with IDs (Finding 1, Finding 2, etc.) and their hypotheses
   - All sub-studies/scenarios/conditions

2. MATERIALS:
   - Actual text of questions, scenarios, instructions, stimuli
   - Item-level details: question text, response options, scales

3. PARTICIPANTS:
   - Sample sizes, demographics, group assignments, exclusion criteria

4. STATISTICAL RESULTS:
   For each test, extract:
   - finding_id: Which finding this addresses (e.g., "Finding 1", "F2")
   - test_name: Exact test name (e.g., "t-test", "ANOVA", "correlation")
   - statistic: Complete string (e.g., "t(23) = 4.66", "F(1, 68) = 6.38", "t < 1")
   - p_value: Exact value (e.g., "p < .001", "p = .04", "not significant")
   - raw_data: Means, SDs, sample sizes for all groups/conditions
   - claim: What the test evaluates
   - location: Page and section (e.g., "Page 489, Table 1")

Extract all tests from Results, Discussion, Tables, and Footnotes. List each test separately. Include main effects, interactions, post-hoc comparisons, and follow-up analyses.

Provide your analysis in JSON format:
{{
    "studies": [
        {{
            "study_id": "Experiment 1",
            "study_name": "...",
            "phenomenon": "...",
            "findings": [
                {{ "finding_id": "Finding 1", "finding_description": "...", "hypothesis": "..." }},
                {{ "finding_id": "Finding 2", "finding_description": "...", "hypothesis": "..." }}
            ],
            "sub_studies": [
                {{
                    "sub_study_id": "...",
                    "type": "task",
                    "content": "...",
                    "items": [...],
                    "participants": {{ "n": 100, ... }},
                    "human_data": {{
                        "item_level_results": [...],
                        "statistical_results": [
                            {{
                                "finding_id": "Finding 1",
                                "test_name": "t-test",
                                "statistic": "t(98) = 4.5",
                                "p_value": "p < .001",
                                "raw_data": {{ "group_1": {{ "mean": 45.2, "sd": 12.3, "n": 50 }}, "group_2": {{ "mean": 32.1, "sd": 10.8, "n": 50 }} }},
                                "claim": "...",
                                "location": "Page 4, Table 1"
                            }}
                        ]
                    }}
                }}
            ]
        }}
    ]
}}"""
    
    def _parse_response(self, response: str, stage1_json: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response"""
        if response is None:
            raise ValueError("LLM response is None")
        
        # Extract JSON from response
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
        
        # Add paper_id from stage1
        result['paper_id'] = stage1_json.get('paper_id', 'unknown')
        result['paper_title'] = stage1_json.get('paper_title', '')
        result['paper_authors'] = stage1_json.get('paper_authors', [])
        result['paper_abstract'] = stage1_json.get('paper_abstract', '')
        
        return result

