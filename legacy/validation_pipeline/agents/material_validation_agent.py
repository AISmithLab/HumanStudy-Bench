"""
Material Validation Agent

Validates materials/*.json (instructions, items, options) against the original paper.
Focuses on missing items, wording drift, and option mismatches.
"""

from pathlib import Path
from typing import Dict, Any, List

from legacy.validation_pipeline.agents.base_agent import BaseValidationAgent
from legacy.validation_pipeline.utils.document_loader import DocumentLoader


class MaterialValidationAgent(BaseValidationAgent):
    """Agent that validates study materials (instructions/items/options)"""

    def validate(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate materials against the PDF and extracted ground truth.

        Args:
            documents: Dictionary containing PDF info, json files, and study_path.

        Returns:
            Validation results with file-specific modification suggestions.
        """
        pdf_files = self._get_pdf_files(documents)
        ground_truth = documents.get("json", {}).get("ground_truth.json", {})
        specification = documents.get("json", {}).get("specification.json", {})
        study_path = Path(documents.get("study_path", "."))

        if not pdf_files:
            raise ValueError("No PDF files found in documents")

        # Load all materials/*.json
        materials_dir = study_path / "materials"
        materials = {}
        if materials_dir.exists():
            for path in materials_dir.glob("*.json"):
                try:
                    materials[str(path.name)] = DocumentLoader.load_json(path)
                except Exception:
                    continue

        system_instruction = (
            "You are an expert validator. Compare benchmark materials with the original paper. "
            "Flag any missing items, wording drift, option mismatches, or instruction gaps. "
            "Be concise and actionable."
        )

        prompt_parts: List[Any] = []
        # Attach PDFs
        prompt_parts.extend(pdf_files)

        text_prompt = f"""
Validate benchmark materials against the original paper.

PDF files are attached.

BENCHMARK MATERIALS (from materials/*.json):
{materials}

GROUND TRUTH (for reference):
{ground_truth}

SPECIFICATION (for expected participants/conditions):
{specification}

Check for each material file:
- Missing questions/items or response options
- Wording differences that change meaning (not just formatting)
- Instruction mismatches (missing constraints, scales, units)
- Alignment with ground_truth hypotheses/tests (if items map to them)

Return JSON:
{{
  "material_validation": [
    {{
      "file": "materials/<name>.json",
      "issue": "description",
      "severity": "critical|high|medium|low",
      "paper_evidence": "page/section or quote",
      "suggested_change": "what to change",
      "proposed_content": "replacement text/options if applicable"
    }}
  ],
  "file_modifications": [
    {{
      "file": "materials/<name>.json",
      "reason": "why",
      "change_type": "update|add|remove",
      "proposed_content": "replacement snippet or data needed"
    }}
  ],
  "summary": "Overall assessment"
}}
"""
        prompt_parts.append(text_prompt)

        result = self._generate_response(
            prompt_parts, system_instruction=system_instruction, structured=True
        )

        return {
            "agent": "MaterialValidationAgent",
            "status": "completed",
            "results": result,
        }

