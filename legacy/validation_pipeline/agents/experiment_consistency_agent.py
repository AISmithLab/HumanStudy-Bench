"""
Experiment Consistency Agent

Verifies that experimental setup matches the original paper, and identifies
any inconsistencies with special markings and validation methods.
"""

from typing import Dict, Any
from legacy.validation_pipeline.agents.base_agent import BaseValidationAgent


class ExperimentConsistencyAgent(BaseValidationAgent):
    """Agent that verifies experimental setup consistency"""
    
    def validate(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify experimental setup consistency.
        
        Args:
            documents: Dictionary containing PDF, README, and config files
            
        Returns:
            Validation results with consistency analysis
        """
        # Get PDF files (uploaded to Gemini API)
        pdf_files = self._get_pdf_files(documents)
        study_info = documents.get("study_info", "")
        specification = documents.get("json", {}).get("specification.json", {})
        metadata = documents.get("json", {}).get("metadata.json", {})
        config_code = documents.get("config_code", "")
        
        if not pdf_files:
            raise ValueError("No PDF files found in documents")
        
        system_instruction = """You are an expert in experimental design and methodology. 
Your task is to compare the experimental setup in the original paper with the benchmark 
implementation and identify any inconsistencies. Be precise and note both exact matches 
and intentional modifications. Pay special attention to ensuring that questions/tasks are 
almost exactly the same, with only formatting differences allowed."""
        
        # Build prompt with PDF files
        prompt_parts = []
        
        # Add PDF files
        for pdf_file in pdf_files:
            prompt_parts.append(pdf_file)
        
        # Add text content
        text_prompt = f"""Compare the experimental setup between the original paper and the benchmark implementation.

STUDY INFORMATION:
{study_info}

BENCHMARK METADATA:
{metadata}

BENCHMARK SPECIFICATION:
{specification}

IMPLEMENTATION CODE:
{config_code if config_code else "Not provided"}

For each aspect of the experimental design, compare:
1. Participant characteristics (N, demographics, recruitment)
2. Experimental procedure and steps
3. Materials and stimuli - CRITICAL: Questions/tasks must be almost exactly the same, only formatting differences are allowed
4. Dependent variables and measures
5. Statistical analyses
6. Conditions and manipulations

IMPORTANT: For questions, tasks, and stimuli:
- They must be almost exactly the same as in the original paper
- Only formatting differences (e.g., whitespace, line breaks, markdown formatting) are acceptable
- Any changes to wording, content, or meaning are considered inconsistencies

For each inconsistency found:
- Mark whether it's an intentional modification or an error
- Provide the original paper's description
- Provide the benchmark's implementation
- Suggest validation methods to verify the modification is acceptable
- Note if the modification affects replicability

Provide your analysis in JSON format:
{{
    "comparison_by_aspect": {{
        "participants": {{
            "original": "Description from paper",
            "benchmark": "Description from implementation",
            "consistent": true/false,
            "modification_type": "exact_match|intentional_modification|error|unclear",
            "validation_method": "How to validate this aspect",
            "notes": "Additional notes"
        }},
        "procedure": {{...}},
        "materials": {{
            "original": "Exact questions/tasks from paper",
            "benchmark": "Questions/tasks from implementation",
            "consistent": true/false,
            "modification_type": "exact_match|formatting_only|intentional_modification|error|unclear",
            "validation_method": "How to validate this aspect",
            "notes": "Note any wording or content differences (not just formatting)"
        }},
        "measures": {{...}},
        "analyses": {{...}},
        "conditions": {{...}}
    }},
    "consistency_summary": {{
        "total_aspects_checked": 0,
        "consistent_aspects": 0,
        "intentional_modifications": [],
        "potential_errors": [],
        "unclear_aspects": [],
        "consistency_score": 0.0,
        "overall_assessment": "Overall assessment of consistency"
    }},
    "validation_plan": [
        {{
            "aspect": "Name of aspect",
            "validation_method": "How to validate",
            "priority": "high|medium|low",
            "description": "Detailed validation steps"
        }}
    ],
    "recommendations": [
        "Recommendations for improving consistency or documenting modifications"
    ]
}}
"""
        prompt_parts.append(text_prompt)
        
        result = self._generate_response(prompt_parts, system_instruction, structured=True)
        
        return {
            "agent": "ExperimentConsistencyAgent",
            "status": "completed",
            "results": result,
        }

