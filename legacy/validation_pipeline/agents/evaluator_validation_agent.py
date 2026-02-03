"""
Evaluator Validation Agent

Validates src/studies/study_XXX_evaluator.py against the original paper and ground_truth.json.
Focuses on whether statistical tests, directions, and parameters match the paper.
"""

from pathlib import Path
from typing import Dict, Any, List

from legacy.validation_pipeline.agents.base_agent import BaseValidationAgent
from legacy.validation_pipeline.utils.document_loader import DocumentLoader


class EvaluatorValidationAgent(BaseValidationAgent):
    """Agent that validates evaluator scripts"""

    def validate(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate evaluator code against paper and ground truth.

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

        study_id = Path(study_path).name
        evaluator_path = Path("src") / "studies" / f"{study_id}_evaluator.py"
        evaluator_code = ""
        if evaluator_path.exists():
            evaluator_code = DocumentLoader.load_python_file(evaluator_path)

        system_instruction = (
            "You are an expert statistician and code reviewer. "
            "Check that the evaluator script matches the paper's reported analyses exactly. "
            "CRITICAL: The evaluator must use EXACT test statistics (t, F, r, chi2 with df) from the paper, "
            "NOT generic p-value-based Bayes Factors. If the paper reports t(79)=2.66 for Study 2 Item 1, "
            "the evaluator should use that exact t-value to calculate bf_human, not a generic p<.05. "
            "Flag any test that uses p-value fallback when exact statistics are available in the paper."
        )

        prompt_parts: List[Any] = []
        prompt_parts.extend(pdf_files)

        text_prompt = f"""
Validate the evaluator script against the original paper and ground truth.

PDF files are attached.

EVALUATOR CODE:
```
{evaluator_code}
```

GROUND TRUTH (parsed findings):
{ground_truth}

SPECIFICATION (for N, conditions):
{specification}

Check:
1. **EXACT STATISTICS USAGE** (CRITICAL):
   - Does the evaluator use EXACT t/F/r/chi2 values from the paper for pi_human calculation?
   - If the evaluator uses `calc_bf_from_p(p_value, n)` but the paper reports the actual t or F statistic, 
     flag this as "SHOULD USE EXACT STAT" and provide the exact value from the paper.
   - Example: If paper says "t(79) = 2.66, p < .05" for Study 2 Item 1, evaluator should use 
     `calc_bf_t(2.66, n1, n2)` NOT `calc_bf_from_p(0.05, n)`.

2. **test_configs / test_results definitions**: align with the paper's reported tests (t/F/chi2/r, dfs, tails)
3. **Directionality**: one-tailed vs two-tailed assumptions and sign handling
4. **Sample sizes**: correct N, n1, n2, df in BF calculations
5. **Naming**: outputs include pi_human_source and consistent naming

Return JSON:
{{
  "evaluator_validation": [
    {{
      "aspect": "exact_stat|stat_test|direction|df|n|mapping",
      "issue": "description - if exact stats available in paper but evaluator uses p-value, say so",
      "severity": "critical|high|medium|low",
      "paper_evidence": "page/section or quote WITH THE EXACT STATISTIC",
      "suggested_change": "what to change in code, including the exact value to use",
      "code_reference": "function or test_name"
    }}
  ],
  "file_modifications": [
    {{
      "file": "src/studies/{study_id}_evaluator.py",
      "reason": "e.g., Replace generic p-value BF with exact t-stat BF for Study 2 items",
      "change_type": "update|add|remove",
      "proposed_content": "code snippet with exact values, e.g., study_2_t_stats = {{1: 2.66, 2: 2.58, ...}}"
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
            "agent": "EvaluatorValidationAgent",
            "status": "completed",
            "results": result,
        }

