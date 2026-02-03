"""
Output Formatter - Formats extraction results as markdown review files and JSON
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class OutputFormatter:
    """Format extraction results for review"""
    
    @staticmethod
    def format_stage1_review(filter_result: Dict[str, Any]) -> str:
        """Format stage1 filter results as markdown"""
        md = f"""# Stage 1: Replicability Filter Review

## Paper Information
- **Title**: {filter_result.get('paper_title', 'N/A')}
- **Authors**: {', '.join(filter_result.get('paper_authors', []))}
- **Abstract**: {filter_result.get('paper_abstract', 'N/A')}

## Experiments Overview

"""
        
        experiments = filter_result.get('experiments', [])
        for i, exp in enumerate(experiments, 1):
            md += f"""### Experiment {i}: {exp.get('experiment_name', exp.get('experiment_id', 'Unknown'))}
- **Input**: {exp.get('input', 'N/A')}
- **Participants**: {exp.get('participants', 'N/A')}
- **Output**: {exp.get('output', 'N/A')}
- **Replicable**: {exp.get('replicable', 'UNCERTAIN')}
- **Exclusion Reasons**: {', '.join(exp.get('exclusion_reasons', [])) or 'None'}

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

"""
        
        md += f"""## Overall Assessment
- **Overall Replicable**: {'YES' if filter_result.get('overall_replicable', False) else 'NO'}
- **Confidence**: {filter_result.get('confidence', 0.0):.2f}
- **Notes**: {filter_result.get('notes', 'N/A')}

## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Review Comments**: [填写]
- **Action**: [continue_to_stage2/refine_stage1/exclude]
"""
        
        return md
    
    @staticmethod
    def format_stage2_review(extraction_result: Dict[str, Any]) -> str:
        """Format stage2 extraction results as markdown"""
        if extraction_result is None:
            raise ValueError("extraction_result is None")
        
        if not isinstance(extraction_result, dict):
            raise ValueError(f"extraction_result is not a dict: {type(extraction_result)}")
        
        md = f"""# Stage 2: Study & Data Extraction Review

"""
        
        studies = extraction_result.get('studies', []) if extraction_result else []
        for study in studies:
            study_id = study.get('study_id', 'Unknown')
            study_name = study.get('study_name', '')
            
            md += f"""## {study_id}: {study_name}

### Phenomenon
{study.get('phenomenon', 'N/A')}

### Sub-Studies/Scenarios

"""
            
            sub_studies = study.get('sub_studies', [])
            for sub_study in sub_studies:
                sub_id = sub_study.get('sub_study_id', 'Unknown')
                sub_type = sub_study.get('type', 'scenario')
                content_preview = sub_study.get('content', '')[:2000] + "..." if len(sub_study.get('content', '')) > 200 else sub_study.get('content', 'N/A')
                
                md += f"""#### {sub_id} ({sub_type})
- **Content Preview**: {content_preview}
- **Participants N**: {sub_study.get('participants', {}).get('n', 'N/A')}
- **Human Data**: {json.dumps(sub_study.get('human_data', {}), indent=2)}
- **Statistical Tests**: {len(sub_study.get('statistical_tests', []))} test(s)

#### Checklist:
- [ ] Content correctly extracted
- [ ] Participant N correct
- [ ] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
[填写]

"""
            
            # Overall participants
            overall_participants = study.get('overall_participants', {})
            md += f"""### Overall Participant Profile
- **Total N**: {overall_participants.get('total_n', 'N/A')}
- **Population**: {overall_participants.get('population', 'N/A')}
- **Recruitment Source**: {overall_participants.get('recruitment_source', 'N/A')}
- **Demographics**: {json.dumps(overall_participants.get('demographics', {}), indent=2)}

#### Checklist:
- [ ] Total N correctly extracted
- [ ] Demographics correctly extracted
- [ ] All available information captured

#### Comments:
[填写]

---

"""
        
        md += """## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Review Comments**: [填写]
- **Action**: [continue_to_final/refine_stage2/back_to_stage1]
"""
        
        return md

