import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.study_config import get_study_config


def _default_llm():
    """Default LLM client for evaluator generation (gemini)."""
    from src.llm.factory import get_client
    return get_client(provider="gemini", model="models/gemini-3-flash-preview")


class EvaluatorGenerator:
    """
    Generates study-specific evaluation code using LLM.

    This agent reads the ground truth and data schema, then writes a Python script
    that performs the exact same statistical test as the original study,
    converting results to Bayesian Alignment Scores.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client if llm_client is not None else _default_llm()
        
    def generate_evaluator(self, study_id: str, study_dir: Path, output_path: Path) -> bool:
        """
        Generates the evaluator code for a study.
        
        Args:
            study_id: Study ID (e.g. 'study_001')
            study_dir: Path to study directory
            output_path: Path to save the generated evaluator.py
            
        Returns:
            bool: Success
        """
        # 1. Load Context
        ground_truth = self._load_json(study_dir / "ground_truth.json")
        specification = self._load_json(study_dir / "specification.json")
        metadata = self._load_json(study_dir / "metadata.json")
        stats_lib_docs = self._get_stats_lib_docs()
        
        # 2. Load Materials context (to get correct item IDs)
        materials_context = self._get_materials_context(study_dir / "materials")
        
        # 3. Get actual response_text sample from cache or simulation
        response_sample = self._get_response_sample(study_id)
        
        # 4. Read study config.py to understand PromptBuilder
        config_context = self._get_config_context(study_id)
        
        # 5. Construct Prompt
        prompt = self._build_prompt(study_id, ground_truth, specification, metadata, stats_lib_docs, materials_context, response_sample, config_context)
        
        # 4. Call LLM
        print(f"Generating evaluator for {study_id}...")
        response = self.llm.generate_content(prompt)
        code = self._extract_code(response)
        
        if not code or len(code.strip()) < 10:
            print(f"Failed to generate code for {study_id}.")
            if response:
                print(f"Raw Response snippet: {response[:200]}...")
            return False
            
        # 4. Save Code
        with open(output_path, 'w') as f:
            f.write(code)
            
        print(f"Evaluator saved to {output_path}")
        return True
        
    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
            
    def _get_stats_lib_docs(self) -> str:
        # Read the stats_lib.py file to get docstrings or just hardcode key API
        return """
        Available Statistical Functions in `src.evaluation.stats_lib`:
        
        def calc_bf_t(t_stat, n1, n2=None, independent=True):
            '''Calculate Bayes Factor (BF10) for t-test (JZS prior).'''
            
        def calc_bf_r(r, n):
            '''Calculate Bayes Factor (BF10) for Pearson correlation.'''
            
        def calc_bf_chisq(chi2, n, df=1):
            '''Calculate BF10 for Chi-Square test.'''
            
        def calc_bf_anova(f_stat, df1, df2, n_total):
            '''Calculate BF10 for ANOVA F-test.'''
            
        def prob_from_bf(bf, prior_odds=1.0):
            '''Convert Bayes Factor to Posterior Probability (pi).'''
            
        def calc_pas(pi_human, pi_agent):
            '''Calculate Bayesian Alignment Score: pi_h*pi_a + (1-pi_h)*(1-pi_a).'''
        """

    def _get_materials_context(self, materials_dir: Path) -> str:
        """Reads all material JSONs to extract sub_study_ids and item IDs."""
        if not materials_dir.exists():
            return "No materials found."
            
        context = "Available Materials (Sub-study IDs and Item IDs):\n"
        for path in materials_dir.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    sub_id = data.get('sub_study_id', path.stem)
                    items = data.get('items', [])
                    item_ids = [i.get('id') for i in items]
                    context += f"- Sub-study: {sub_id} | Item IDs: {', '.join(item_ids)}\n"
            except Exception:
                continue
        return context

    def _get_response_sample(self, study_id: str) -> str:
        """
        Gets actual response_text samples from the most recent successful benchmark run.
        This is CRITICAL for the LLM to understand the actual response format.

        Priority:
        1. Look for most recent benchmark run in results/runs/
        2. Fall back to cache files if no benchmark runs found
        """
        def _truncate_at_line_boundaries(text: str, max_chars: int) -> str:
            """
            Keep whole lines only (never cut a line mid-token like 'ESTIM...').
            If text is short, return as-is.
            """
            text = text or ""
            if len(text) <= max_chars:
                return text
            lines = text.splitlines()
            out_lines = []
            n = 0
            for line in lines:
                # +1 for newline join
                add = len(line) + (1 if out_lines else 0)
                if n + add > max_chars:
                    break
                out_lines.append(line)
                n += add
            return "\n".join(out_lines)

        # First, try to find the most recent benchmark run
        runs_dir = Path("results/runs")
        if runs_dir.exists():
            # Get all run directories, sorted by modification time (newest first)
            run_dirs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for run_dir in run_dirs:
                benchmark_file = run_dir / "full_benchmark.json"
                if benchmark_file.exists():
                    try:
                        with open(benchmark_file, 'r') as f:
                            data = json.load(f)
                            studies = data.get('studies', [])
                            
                            # Find this study in the results
                            for study_result in studies:
                                if study_result.get('study_id') == study_id:
                                    individual_data = study_result.get('individual_data', [])
                                    if individual_data:
                                        # Collect samples that cover EACH sub_study_id.
                                        # Pick ONE representative full response_text per sub-study.
                                        samples_by_sub: Dict[str, Dict[str, Any]] = {}

                                        # Collect ONE complete example per sub_study_id showing the FULL nested structure
                                        samples_by_sub: Dict[str, Dict[str, Any]] = {}

                                        for p in individual_data:
                                            for r in p.get('responses', []):
                                                trial_info = r.get('trial_info', {}) or {}
                                                sub_id = trial_info.get('sub_study_id') or trial_info.get('sub_id') or "unknown"
                                                txt = r.get('response_text', '') or ''
                                                if not txt.strip():
                                                    continue

                                                if sub_id not in samples_by_sub:
                                                    # Keep FULL nested structure - this is critical for the LLM to understand
                                                    samples_by_sub[sub_id] = {
                                                        "participant_id": p.get('participant_id'),
                                                        "responses": [
                                                            {
                                                                "response_text": txt,
                                                                "trial_info": {
                                                                    "sub_study_id": trial_info.get('sub_study_id'),
                                                                    "group_name": trial_info.get('group_name'),
                                                                    "items": trial_info.get('items', [])[:3]  # Show first 3 items as example
                                                                }
                                                            }
                                                        ]
                                                    }
                                                else:
                                                    # If we already have one, prefer one that looks more "complete" 
                                                    existing = samples_by_sub[sub_id]
                                                    existing_txt = existing['responses'][0]['response_text']
                                                    if txt.count('\n') > existing_txt.count('\n') or \
                                                       txt.count('ITEM_ID:') > existing_txt.count('ITEM_ID:'):
                                                        samples_by_sub[sub_id]['responses'][0]['response_text'] = txt
                                                        samples_by_sub[sub_id]['responses'][0]['trial_info'] = {
                                                            "sub_study_id": trial_info.get('sub_study_id'),
                                                            "group_name": trial_info.get('group_name'),
                                                            "items": trial_info.get('items', [])[:3]
                                                        }

                                        flattened = list(samples_by_sub.values())
                                        if flattened:
                                            return (
                                                f"**ACTUAL COMPLETE DATA STRUCTURE from benchmark run ({run_dir.name}) "
                                                f"(one representative participant per sub_study_id, showing FULL nesting)**:\n"
                                                f"```json\n{json.dumps(flattened, indent=2)}\n```\n\n"
                                                f"**CRITICAL**: Notice that `trial_info` is nested INSIDE each `response` object, "
                                                f"which is nested INSIDE each `participant` object. You MUST iterate as: "
                                                f"`for participant in results['individual_data']: for response in participant['responses']: ...`"
                                            )
                    except Exception as e:
                        continue
        
        # Fallback: Check cache directory
        cache_dir = Path("results/cache")
        if cache_dir.exists():
            # Check if config file exists and is newer than any cache files
            config_path = Path(f"src/studies/{study_id}_config.py")
            if config_path.exists():
                config_mtime = config_path.stat().st_mtime

                # Check if any cache files are older than the config
                cache_files = list(cache_dir.glob(f"{study_id}_*.json"))
                if cache_files:
                    # If config is newer than the newest cache file, skip cached samples
                    newest_cache_mtime = max(cache_file.stat().st_mtime for cache_file in cache_files)
                    if config_mtime > newest_cache_mtime:
                        return f"Config file is newer than cached results. Please run the benchmark first to generate raw responses, then re-run Stage 5."

            # Find any cache file for this study
            for cache_file in cache_dir.glob(f"{study_id}_*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        individual_data = data.get('individual_data', [])
                        if individual_data:
                            # Get first 2 participants with FULL nested structure
                            samples = []
                            for p in individual_data[:2]:
                                participant_sample = {
                                    "participant_id": p.get('participant_id'),
                                    "responses": []
                                }
                                for r in p.get('responses', [])[:1]:
                                    trial_info = r.get('trial_info', {})
                                    participant_sample["responses"].append({
                                        "response_text": r.get('response_text', '')[:500],  # Truncate for brevity
                                        "trial_info": {
                                        "sub_study_id": trial_info.get('sub_study_id') or trial_info.get('sub_id'),
                                            "group_name": trial_info.get('group_name'),
                                            "items": trial_info.get('items', [])[:2]  # Show first 2 items
                                        }
                                    })
                                if participant_sample["responses"]:
                                    samples.append(participant_sample)
                            if samples:
                                return (
                                    f"**ACTUAL COMPLETE DATA STRUCTURE from cached data (showing FULL nesting)**:\n"
                                    f"```json\n{json.dumps(samples, indent=2)}\n```\n\n"
                                    f"**CRITICAL**: Notice that `trial_info` is nested INSIDE each `response` object, "
                                    f"which is nested INSIDE each `participant` object."
                                )
                except Exception:
                    continue
        
        return "No response samples available. Please run the benchmark first to generate raw responses, then re-run Stage 5."

    def _get_config_context(self, study_id: str) -> str:
        """
        Reads the FULL study config.py to understand trial structure and prompt logic.
        """
        config_path = Path(f"src/studies/{study_id}_config.py")
        if not config_path.exists():
            return "Study config file not found."
        
        try:
            content = config_path.read_text(encoding='utf-8')
            return f"**FULL Study Config (`{study_id}_config.py`)**:\n```python\n{content}\n```"
        except Exception as e:
            return f"Error reading config: {e}"

    def _get_results_schema(self) -> str:
        """Returns a string representation of the results object key tree with explicit iteration instructions."""
        return """
**EXACT DATA STRUCTURE:**

results = {
    "individual_data": [  # List of participant objects
        {
            "participant_id": int,
            "profile": { "age": int, "gender": str, ... },
            "responses": [  # List of response objects (one per trial)
                {
                    "response_text": str,  # The raw string from the agent using standardized format (e.g., "Q1=A\\nQ2=75\\nQ3=3...")
                    "trial_info": {  # Metadata attached to this specific trial in config.py
                        "sub_study_id": str,  # e.g., "study_1_hypothetical_stories"
                        "group_name": str,  # e.g., "supermarket" (may differ from ground_truth keys like "supermarket_story")
                        "items": [  # List of item objects presented in this trial
                            { "id": str, "question": str, "type": str, "options": [...], ... }
                        ],
                        "profile": { "age": int, "gender": str, ... },  # Participant profile for this trial
                        ...
                    }
                }
            ]
        }
    ]
}

**CRITICAL ITERATION PATTERN (MUST FOLLOW THIS EXACTLY):**

```python
# CORRECT: Nested iteration through participants and their responses
for participant in results.get("individual_data", []):
    participant_id = participant.get("participant_id")
    for response in participant.get("responses", []):
        response_text = response.get("response_text", "")  # ← response_text is HERE
        trial_info = response.get("trial_info", {})      # ← trial_info is HERE (inside response!)
        sub_study_id = trial_info.get("sub_study_id")
        group_name = trial_info.get("group_name")
        items = trial_info.get("items", [])
        # ... process this response
```

**WRONG PATTERNS (DO NOT USE):**
```python
# WRONG: trial_info is NOT at participant level
for participant in results.get("individual_data", []):
    trial_info = participant.get("trial_info", {})  # ❌ WRONG!

# WRONG: response_text is NOT at participant level  
for participant in results.get("individual_data", []):
    response_text = participant.get("response_text", "")  # ❌ WRONG!
```

**IMPORTANT NOTES:**
1. The `results` dict contains ONLY `individual_data`. Ground truth data must be loaded separately from `data/studies/{study_id}/ground_truth.json`.
2. Each participant can have MULTIPLE responses (one per trial).
3. Each response has its own `trial_info` with `sub_study_id`, `group_name`, and `items`.
4. The `group_name` in `trial_info` may differ from keys in `ground_truth.json` (e.g., "supermarket" vs "supermarket_story").
"""

    def _build_prompt(self, study_id, ground_truth, specification, metadata, stats_lib_docs, materials_context, response_sample="", config_context="") -> str:
        template = """
You are an expert statistician and Python developer for HumanStudyBench.
Your task is to write `study_[[STUDY_ID]]_evaluator.py` to evaluate an AI agent's performance.

### Goal
Calculate **Bayesian Alignment Score (PAS)** by comparing agent statistical evidence against human ground truth.
PAS = pi_human * pi_agent + (1 - pi_human) * (1 - pi_agent)

### Core Principles
1. **Use HUMAN sample size for pi_human, AGENT sample size for pi_agent** - Never mix them
2. **Process ALL tests** - Each finding may have multiple statistical tests, process all of them
3. **Match the exact test** - Run the same statistical test on agent data as reported in ground truth
4. **Two-level weighted aggregation**:
   - **Finding Score** = Σ (Test PAS * Test Weight) / Σ Test Weights (for all tests in that finding)
   - **Study Score** = Σ (Finding Score * Finding Weight) / Σ Finding Weights (for all findings)

### Data Structure
**Input:** `results["individual_data"]` → `participant["responses"]` → `response["response_text"]` and `response["trial_info"]`

**Ground Truth:** Load from `data/studies/[[STUDY_ID]]/ground_truth.json` (NOT in results dict)

**Metadata:** Load from `data/studies/[[STUDY_ID]]/metadata.json` to get finding and test weights

### Available Functions
[[STATS_LIB_DOCS]]

### Context
**Study Config:** [[CONFIG_CONTEXT]]
**Ground Truth:** [[GROUND_TRUTH]]
**Metadata:** [[METADATA]]
**Response Samples:** [[RESPONSE_SAMPLE]]
**Materials:** [[MATERIALS_CONTEXT]]

### Required Functions

You MUST include these functions:

1. **`parse_agent_responses(response_text: str) -> Dict[str, str]`**
   - Parse Qk=<value> or Qk.n=<value> format from agent response
   - Use regex pattern: `r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)"`
   - Return a dictionary mapping Q numbers to their values

2. **`get_required_q_numbers(trial_info: Dict[str, Any]) -> set`**
   - Extract all required Q numbers from trial_info
   - This function is used by sanity_check to validate responses
   - Implementation depends on how Q numbers are assigned:
     - If Q numbers are based on item index: `Q{idx+1}` for each item in `trial_info["items"]`
     - If items have explicit `q_idx` field: use that field
     - Return a set of strings like `{"Q1", "Q2", "Q3"}` or `{"Q1.1", "Q1.2"}`
   - Example:
     ```python
     def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
         required = set()
         items = trial_info.get("items", [])
         for idx, item in enumerate(items):
             q_idx = item.get("q_idx")
             if q_idx:
                 if isinstance(q_idx, str) and q_idx.startswith("Q"):
                     required.add(q_idx)
                 else:
                     required.add(f"Q{q_idx}")
             else:
                 # Default: use index + 1
                 required.add(f"Q{idx + 1}")
         return required
     ```

3. **`evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]`**
   - Main evaluation function (see example below)

### Working Example Pattern

```python
import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from src.evaluation.stats_lib import (
    calc_bf_t, prob_from_bf, prob_from_bf_human, calc_pas,
    parse_p_value_from_reported, get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    \"\"\"Parse Qk=<value> or Qk.n=<value> format\"\"\"
    results = {}
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    \"\"\"Extract all required Q numbers from trial_info\"\"\"
    required = set()
    items = trial_info.get("items", [])
    for idx, item in enumerate(items):
        required.add(f"Q{idx + 1}")  # Adjust based on actual Q numbering scheme
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Load ground truth and metadata
    study_dir = Path(f"data/studies/study_002")
    with open(study_dir / "ground_truth.json", 'r') as f:
        ground_truth = json.load(f)
    
    # Load metadata for finding and test weights
    metadata = {}
    metadata_path = study_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Build weight maps: finding_id -> weight, (finding_id, test_name) -> weight
    finding_weights = {}
    test_weights = {}
    for finding in metadata.get("findings", []):
        finding_id = finding.get("finding_id")
        finding_weight = finding.get("weight", 1.0)
        if finding_id:
            finding_weights[finding_id] = finding_weight
        
        for test in finding.get("tests", []):
            test_name = test.get("test_name")
            test_weight = test.get("weight", 1.0)
            if finding_id and test_name:
                test_weights[(finding_id, test_name)] = test_weight
    
    # 2. Extract agent data
    agent_data = {"exp_1_calibration": [], "exp_1_anchored_estimation": []}
    
    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            items = trial_info.get("items", [])
            
            # Parse Qk=value format
            parsed = parse_agent_responses(response_text)
            
            # Map Q indices to items
            for item in items:
                q_est = item.get("q_idx_estimate")
                if q_est and q_est in parsed:
                    estimate = float(parsed[q_est])
                    agent_data[sub_id].append({
                        "estimate": estimate,
                        "label": item.get("metadata", {}).get("label")
                    })
    
    # 3. For each test in ground truth
    test_results = []
    for study in ground_truth["studies"]:
        for finding in study["findings"]:
            for test in finding["statistical_tests"]:
                # Extract human statistic: "t(102)=7.99" → t=7.99, n=103
                reported = test["reported_statistics"]
                # Parse t-stat and n from reported statistics
                t_human = 7.99  # Extract from reported
                n_human = 103   # Extract from reported
                
                # Calculate pi_human using HUMAN n
                pi_h = prob_from_bf_human(calc_bf_t(t_human, n_human, independent=False))
                
                # Run same test on agent data
                agent_group1 = [d["estimate"] for d in agent_data["exp_1_anchored_estimation"] if d["label"] == "high"]
                agent_group2 = [d["estimate"] for d in agent_data["exp_1_anchored_estimation"] if d["label"] == "low"]
                
                if len(agent_group1) > 2 and len(agent_group2) > 2:
                    t_stat, p_val_agent = stats.ttest_ind(agent_group1, agent_group2)
                    # Calculate pi_agent using AGENT n
                    pi_a = prob_from_bf(calc_bf_t(t_stat, len(agent_group1), len(agent_group2), independent=True))
                    reason = f"t={t_stat:.2f}, n1={len(agent_group1)}, n2={len(agent_group2)}"
                else:
                    pi_a = 0.5
                    p_val_agent = None
                    t_stat = None
                    reason = "Insufficient data"
                
                # Calculate PAS
                pas = calc_pas(pi_h, pi_a)
                
                # Get test weight (default to 1.0 if not found)
                test_weight = test_weights.get((finding["finding_id"], test["test_name"]), 1.0)
                
                # Create test result dict
                test_result = {
                    "study_id": study["study_id"],
                    "sub_study_id": "exp_1_anchored_estimation",
                    "finding_id": finding["finding_id"],
                    "test_name": test["test_name"],
                    "scenario": "Global",
                    "pi_human": pi_h,
                    "pi_agent": pi_a,
                    "pas": pas,
                    "test_weight": test_weight,
                    "pi_human_source": reported,
                    "agent_reason": reason,
                    "statistical_test_type": "t-test",
                    "human_test_statistic": str(t_human) if 't_human' in locals() else "",
                    "agent_test_statistic": f"{t_stat:.2f}" if t_stat is not None else ""
                }
                
                # Add statistical replication fields (p-values, significance, direction)
                add_statistical_replication_fields(test_result, test, p_val_agent, t_stat, "t-test")
                
                test_results.append(test_result)
    
    # 4. Two-level weighted aggregation
    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    finding_ids = sorted(list(set(tr["finding_id"] for tr in test_results)))
    for fid in finding_ids:
        fid_tests = [tr for tr in test_results if tr["finding_id"] == fid]
        sub_id = fid_tests[0]["sub_study_id"] if fid_tests else "unknown"
        
        # Weighted average: Σ (PAS * weight) / Σ weights
        total_weighted_pas = sum(tr["pas"] * tr.get("test_weight", 1.0) for tr in fid_tests)
        total_weight = sum(tr.get("test_weight", 1.0) for tr in fid_tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(fid, 1.0)
        finding_results.append({
            "sub_study_id": sub_id,
            "finding_id": fid,
            "finding_score": float(finding_score),
            "finding_weight": float(finding_weight),
            "n_tests": len(fid_tests)
        })
    
    # Level 2: Aggregate findings into study score (weighted by finding weights)
    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5
    
    # Also compute substudy_results for backward compatibility (unweighted average of findings per substudy)
    substudy_results = []
    substudy_ids = sorted(list(set(fr["sub_study_id"] for fr in finding_results)))
    for sid in substudy_ids:
        sid_findings = [fr for fr in finding_results if fr["sub_study_id"] == sid]
        score = np.mean([fr["finding_score"] for fr in sid_findings]) if sid_findings else 0.5
        substudy_results.append({
            "sub_study_id": sid,
            "substudy_score": float(score),
            "n_findings": len(sid_findings)
        })
    
    return {
        "score": float(overall_score),
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }
```

### Key Points
- **Required functions**: You MUST include `parse_agent_responses`, `get_required_q_numbers`, and `evaluate_study`
- **Data iteration**: `for participant in results["individual_data"]: for response in participant["responses"]: ...`
- **Response parsing**: Use regex `r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)"` to parse Qk=value or Qk.n=value
- **Q mapping**: Use `item["q_idx_estimate"]`, `item["q_idx_choice"]`, etc. from `trial_info["items"]`, or use item index if no explicit q_idx
- **get_required_q_numbers**: This function is critical for sanity checking - it tells the system which Q numbers should be present in each response
- **pi_human**: Always use human n from ground_truth (e.g., `calc_bf_t(7.99, 103, ...)`)
- **pi_agent**: Always use agent n from agent data (e.g., `calc_bf_t(t_stat, len(group1), len(group2), ...)`)
- **CRITICAL: Statistical Replication Fields & Effect Sizes**: 
  - You MUST capture the p-value from scipy.stats functions (e.g., `t_stat, p_val = stats.ttest_ind(...)`)
  - You MUST extract sample sizes for both human and agent data:
    * For **t-tests**: Extract `n_human` and `n2_human` from ground truth (e.g., "t(102)=7.99" means df=102, so n≈103 for one-sample or n1+n2≈103 for independent)
    * For **correlations**: Extract `n_human` from ground truth (e.g., "r=0.45, n=103" or infer from reported statistics)
    * For **chi-square**: Build the 2x2 contingency table `[[a, b], [c, d]]` from ground truth data
  - Use `add_statistical_replication_fields(test_result, test_gt, p_val_agent, test_stat_agent, test_type, n_agent=n1, n2_agent=n2, n_human=n_h1, n2_human=n_h2, contingency_agent=[[a,b],[c,d]], contingency_human=[[a_h,b_h],[c_h,d_h]], independent=True)` to add all required fields
  - This function automatically:
    * Parses human p-value from `test["reported_statistics"]` and determines significance and direction
    * Calculates Z-difference and Replication Consistency Score (frequentist effect size comparison)
    * Stores `agent_effect_size` and `human_effect_size` (Cohen's d for t-tests, Fisher's z for correlations, log OR for chi-square)
  - Required fields: `p_value_human`, `p_value_agent`, `is_significant_human`, `is_significant_agent`, `human_direction`, `agent_direction`, `direction_match`, `significance_level`, `z_diff`, `replication_consistency`, `agent_effect_size`, `human_effect_size`
- **Weighted aggregation**: 
  - Load weights from `metadata.json` (finding weights and test weights)
  - Finding Score = Σ (Test PAS * Test Weight) / Σ Test Weights
  - Study Score = Σ (Finding Score * Finding Weight) / Σ Finding Weights
  - If weights are missing, default to 1.0
- **Output format**: Return dict with `score`, `substudy_results`, `finding_results`, `test_results` as shown above

### Output Structure
```python
{
    "score": float,  # Weighted average: Σ (Finding Score * Finding Weight) / Σ Finding Weights
    "substudy_results": [{"sub_study_id": str, "substudy_score": float, "n_findings": int}, ...],
    "finding_results": [{
        "sub_study_id": str,
        "finding_id": str,
        "finding_score": float,  # Weighted average: Σ (Test PAS * Test Weight) / Σ Test Weights
        "finding_weight": float,
        "n_tests": int
    }, ...],
    "test_results": [{
        "study_id": str,
        "sub_study_id": str,
        "finding_id": str,
        "test_name": str,
        "scenario": str,  # Optional
        "pi_human": float,
        "pi_agent": float,
        "pas": float,
        "test_weight": float,  # Weight for this test (from metadata.json)
        "pi_human_source": str,  # e.g., "t(102)=7.99"
        "agent_reason": str,  # e.g., "t=5.23, n1=100, n2=95" or "Insufficient data"
        # Statistical replication fields (REQUIRED):
        "p_value_human": float or None,  # Parsed from reported_statistics
        "p_value_agent": float or None,  # From scipy.stats function (e.g., stats.ttest_ind returns p-value)
        "is_significant_human": bool,  # p_value_human < significance_level
        "is_significant_agent": bool,  # p_value_agent < significance_level
        "human_direction": int,  # 1 for positive, -1 for negative, 0 if unclear
        "agent_direction": int,  # 1 for positive, -1 for negative, 0 if unclear
        "direction_match": bool,  # True if human_direction == agent_direction
        "significance_level": float  # Usually 0.05
    }, ...]
}
```

Generate the complete `evaluate_study(results)` function now:
"""
        return template.replace("[[STUDY_ID]]", study_id)\
                       .replace("[[CONFIG_CONTEXT]]", config_context)\
                       .replace("[[GROUND_TRUTH]]", json.dumps(ground_truth, indent=2))\
                       .replace("[[METADATA]]", json.dumps(metadata, indent=2))\
                       .replace("[[RESPONSE_SAMPLE]]", response_sample)\
                       .replace("[[STATS_LIB_DOCS]]", stats_lib_docs)\
                       .replace("[[MATERIALS_CONTEXT]]", materials_context)

    def _extract_code(self, text: str) -> str:
        if not text:
            return ""
        
        # 1. Try python block
        python_pattern = r"```python\s*(.*?)```"
        match = re.search(python_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 2. Try generic block
        generic_pattern = r"```\s*(.*?)```"
        match = re.search(generic_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 3. Try raw code (if it looks like code)
        trimmed = text.strip()
        if "import " in trimmed[:50] or "def evaluate_study" in trimmed:
            return trimmed
            
        return ""

