"""
Experiment Completeness Agent

Verifies that all replicable experiments from the original paper are included
in the benchmark implementation.
"""

from typing import Dict, Any
from legacy.validation_pipeline.agents.base_agent import BaseValidationAgent


class ExperimentCompletenessAgent(BaseValidationAgent):
    """Agent that verifies all experiments are included"""
    
    def validate(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that all replicable experiments are included.
        
        Args:
            documents: Dictionary containing PDF, README, and config files
            
        Returns:
            Validation results with completeness analysis
        """
        # Get PDF files (uploaded to Gemini API)
        pdf_files = self._get_pdf_files(documents)
        study_info = documents.get("study_info", "")
        specification = documents.get("json", {}).get("specification.json", {})
        config_code = documents.get("config_code", "")
        stage1_filter = documents.get("json", {}).get("stage1_filter.json")
        
        # Extract list of replicable experiments from stage1_filter if available
        replicable_experiments = []
        if stage1_filter:
            for exp in stage1_filter.get("experiments", []):
                # Check various ways 'replicable' might be stored
                is_replicable = False
                rep_val = exp.get("replicable")
                if isinstance(rep_val, bool):
                    is_replicable = rep_val
                elif isinstance(rep_val, str):
                    is_replicable = rep_val.upper() in ["YES", "TRUE"]
                
                if is_replicable:
                    replicable_experiments.append(exp.get("experiment_id") or exp.get("experiment_name"))

        if not pdf_files:
            raise ValueError("No PDF files found in documents")
        
        system_instruction = """You are an expert in experimental psychology and research methodology. 
Your task is to analyze whether all experiments with statistical (quantitative) data from a research paper 
have been correctly included in a benchmark implementation. Focus on experiments that can be replicated using LLMs.
Be thorough and precise in your analysis."""
        
        # Build prompt with PDF files
        prompt_parts = []
        
        # Add PDF files
        for pdf_file in pdf_files:
            prompt_parts.append(pdf_file)
        
        # Add text content
        replicable_context = ""
        if replicable_experiments:
            replicable_context = f"\nSTAGE 1 REPLICABLE EXPERIMENTS (Only flag these if missing):\n{', '.join(replicable_experiments)}\n"

        text_prompt = f"""Analyze the completeness of experiment transfer from the original paper to the benchmark implementation.

STUDY INFORMATION:
{study_info}

BENCHMARK SPECIFICATION:
{specification}

IMPLEMENTATION CODE (if available):
{config_code if config_code else "Not provided"}
{replicable_context}

Please analyze:
1. List ALL experiments/studies mentioned in the original paper that contain statistical (quantitative) data
2. For each experiment, determine if it is replicable using LLM:
   - NOT replicable if it requires: visual input, time sense/perception, or lacks quantitative data
   - IS replicable if it involves: emotional simulation, problem-solving, role-playing, hypothetical scenarios, or any text-based tasks with quantitative outcomes
3. Check if each LLM-replicable experiment with statistical data is included in the benchmark implementation
4. Identify any missing experiments. **CRITICAL: Only flag experiments as 'missing' if they were identified as replicable in Stage 1 (if the list was provided above).**
5. Note any experiments that were intentionally excluded and why (if stated)

IMPORTANT: Only include experiments that have statistical (quantitative) data reported in the paper.

Provide your analysis in the following JSON format:
{{
    "experiments_in_paper": [
        {{
            "experiment_id": "Study 1",
            "description": "Brief description",
            "has_statistical_data": true/false,
            "replicable_using_llm": true/false,
            "replicable_reason": "Why it is or isn't replicable using LLM (note if requires visual/time sense/lacks quantitative data)",
            "included_in_benchmark": true/false,
            "implementation_details": "How it's implemented or why excluded",
            "notes": "Additional notes"
        }}
    ],
    "completeness_summary": {{
        "total_experiments_with_data": 0,
        "llm_replicable_experiments": 0,
        "included_experiments": 0,
        "missing_experiments": [],
        "intentionally_excluded": [],
        "completeness_score": 0.0,
        "completeness_notes": "Overall assessment"
    }},
    "recommendations": [
        "List of recommendations for improving completeness"
    ]
}}
"""
        prompt_parts.append(text_prompt)
        
        result = self._generate_response(prompt_parts, system_instruction, structured=True)
        
        return {
            "agent": "ExperimentCompletenessAgent",
            "status": "completed",
            "results": result,
        }

