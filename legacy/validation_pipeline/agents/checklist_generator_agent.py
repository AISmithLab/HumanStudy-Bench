"""
Checklist Generator Agent

Generates a detailed structured checklist for post-human validation.
"""

from typing import Dict, Any
from legacy.validation_pipeline.agents.base_agent import BaseValidationAgent


class ChecklistGeneratorAgent(BaseValidationAgent):
    """Agent that generates structured validation checklist"""
    
    def validate(self, documents: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured checklist based on all validation results.
        
        Args:
            documents: Dictionary containing all documents
            previous_results: Results from other validation agents
            
        Returns:
            Generated checklist
        """
        # Combine all previous validation results
        completeness_results = previous_results.get("completeness", {})
        consistency_results = previous_results.get("consistency", {})
        data_validation_results = previous_results.get("data_validation", {})
        
        system_instruction = """You are an expert in quality assurance and validation processes. 
Your task is to create a comprehensive, structured checklist that can be used for 
post-human validation of a research benchmark implementation. The checklist should be 
actionable, specific, and cover all critical aspects."""
        
        prompt = f"""Generate a comprehensive structured checklist for post-human validation.

Based on the validation results from other agents:

COMPLETENESS VALIDATION RESULTS:
{completeness_results}

CONSISTENCY VALIDATION RESULTS:
{consistency_results}

DATA VALIDATION RESULTS:
{data_validation_results}

Create a detailed, structured checklist that covers:
1. Experiment completeness verification
2. Experimental setup consistency checks
3. Human data accuracy verification
4. Implementation quality checks
5. Documentation completeness

Each checklist item should have:
- Clear description
- Verification method
- Expected outcome
- Priority level
- Reference to original paper section

Provide the checklist in JSON format:
{{
    "checklist_sections": [
        {{
            "section_id": "completeness",
            "section_name": "Experiment Completeness",
            "items": [
                {{
                    "item_id": "C1",
                    "description": "Clear description of what to check",
                    "verification_method": "How to verify this item",
                    "expected_outcome": "What the correct result should be",
                    "priority": "critical|high|medium|low",
                    "paper_reference": "Section/page in original paper",
                    "benchmark_reference": "Where to check in benchmark",
                    "status": "pending|passed|failed|needs_review",
                    "notes": "Additional notes"
                }}
            ]
        }},
        {{
            "section_id": "consistency",
            "section_name": "Experimental Setup Consistency",
            "items": [...]
        }},
        {{
            "section_id": "data_accuracy",
            "section_name": "Human Data Accuracy",
            "items": [...]
        }},
        {{
            "section_id": "implementation",
            "section_name": "Implementation Quality",
            "items": [...]
        }},
        {{
            "section_id": "documentation",
            "section_name": "Documentation Completeness",
            "items": [...]
        }}
    ],
    "checklist_summary": {{
        "total_items": 0,
        "critical_items": 0,
        "high_priority_items": 0,
        "medium_priority_items": 0,
        "low_priority_items": 0,
        "estimated_validation_time": "Estimated time to complete validation"
    }},
    "validation_workflow": [
        {{
            "step": 1,
            "description": "Step description",
            "required_items": ["item_ids"],
            "estimated_time": "Time estimate"
        }}
    ]
}}
"""
        
        result = self._generate_response(prompt, system_instruction, structured=True)
        
        return {
            "agent": "ChecklistGeneratorAgent",
            "status": "completed",
            "results": result,
        }

