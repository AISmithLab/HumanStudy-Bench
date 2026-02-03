import json
import re
import numpy as np
from scipy import stats
from pathlib import Path

from src.evaluation.stats_lib import (
    calc_bf_t,
    calc_bf_anova,
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)
from typing import Dict, Any, List

# Module-level cache for ground truth and metadata (loaded once, reused across bootstrap iterations)
_ground_truth_cache = None
_metadata_cache = None
_finding_weights_cache = None
_test_weights_cache = None

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """
    Parse standardized responses: Qk=Value or Qk: Value
    Regex handles lines, commas, or spaces as delimiters
    Supports both = and : separators
    """
    parsed_responses = {}
    matches = re.findall(r"Q(\d+)\s*[:=]\s*([^,\s\n]+)", response_text)
    for q_num, val in matches:
        parsed_responses[f"Q{q_num}"] = val.strip()
    return parsed_responses

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_001: 需要处理q_idx_choice, q_idx_est_a, q_idx_est_b, trait_q_map
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        # 基本Q编号
        for key in ['q_idx_choice', 'q_idx_est_a', 'q_idx_est_b', 'q_idx_estimate']:
            q_idx = item.get(key)
            if q_idx:
                # 如果q_idx已经包含"Q"前缀，直接使用；否则添加"Q"前缀
                if isinstance(q_idx, str) and q_idx.startswith("Q"):
                    required.add(q_idx)
                else:
                    required.add(f"Q{q_idx}")
        
        # Trait ratings (for Findings F2 and F5)
        trait_map = item.get("trait_q_map", {})
        for trait, qs in trait_map.items():
            if isinstance(qs, dict):
                for opt_key in ['opt_a', 'opt_b']:
                    q_idx = qs.get(opt_key)
                    if q_idx:
                        # 如果q_idx已经包含"Q"前缀，直接使用；否则添加"Q"前缀
                        if isinstance(q_idx, str) and q_idx.startswith("Q"):
                            required.add(q_idx)
                        else:
                            required.add(f"Q{q_idx}")
    
    return required

def evaluate_study(results):
    """
    Evaluates the agent's performance on Study 001 (False Consensus Effect) 
    by calculating the Bayesian Alignment Score (PAS).
    
    Uses module-level caching to avoid reloading ground truth and metadata
    on every bootstrap iteration (significant performance improvement).
    """
    global _ground_truth_cache, _metadata_cache, _finding_weights_cache, _test_weights_cache
    
    # 1. Load Ground Truth Data and Metadata (with caching)
    study_id = "study_001"
    
    # Load ground truth (cached)
    if _ground_truth_cache is None:
        study_dir = Path(f"data/studies/{study_id}")
        with open(study_dir / "ground_truth.json", 'r') as f:
            _ground_truth_cache = json.load(f)
    
    ground_truth = _ground_truth_cache
    
    # Load metadata (cached)
    if _metadata_cache is None:
        study_dir = Path(f"data/studies/{study_id}")
        metadata_path = study_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}
    
    metadata = _metadata_cache
    
    # Build weight maps (cached)
    if _finding_weights_cache is None or _test_weights_cache is None:
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
        
        _finding_weights_cache = finding_weights
        _test_weights_cache = test_weights
    
    finding_weights = _finding_weights_cache
    test_weights = _test_weights_cache

    # Cache for materials files to avoid reloading
    materials_cache = {}
    study_dir = Path(f"data/studies/{study_id}")
    
    def load_materials(sub_study_id: str) -> List[Dict[str, Any]]:
        """Load items from materials file if not in cache."""
        if sub_study_id not in materials_cache:
            materials_path = study_dir / "materials" / f"{sub_study_id}.json"
            if materials_path.exists():
                try:
                    with open(materials_path, 'r') as f:
                        material_data = json.load(f)
                        materials_cache[sub_study_id] = material_data.get("items", [])
                except Exception:
                    materials_cache[sub_study_id] = []
            else:
                materials_cache[sub_study_id] = []
        return materials_cache[sub_study_id]

    # 2. Extract and Aggregate Agent Data
    # Structure: aggregated[sub_study_id][scenario_key] = [list of data dicts]
    aggregated = {}

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_study_id = trial_info.get("sub_study_id")
            scenario_id = trial_info.get("scenario_id")
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_study_id:
                items = load_materials(sub_study_id)

            if not sub_study_id:
                continue

            if sub_study_id not in aggregated:
                aggregated[sub_study_id] = {}

            # Parse standardized responses: Qk=Value
            parsed_responses = parse_agent_responses(response_text)

            if sub_study_id == "study_2_personal_description_items":
                # Study 2 has multiple items in one trial
                # Re-calculate indices if missing (common if items loaded from materials)
                q_counter = 1
                for item in items:
                    if 'q_idx_choice' not in item:
                        item['q_idx_choice'] = q_counter
                        q_counter += 1
                        item['q_idx_estimate'] = q_counter
                        q_counter += 1
                    
                    gt_key = item.get("metadata", {}).get("gt_key")
                    if not gt_key:
                        continue
                    
                    if gt_key not in aggregated[sub_study_id]:
                        aggregated[sub_study_id][gt_key] = []
                    
                    q_choice_idx = item.get('q_idx_choice')
                    q_est_idx = item.get('q_idx_estimate')
                    
                    raw_choice = parsed_responses.get(f"Q{q_choice_idx}")
                    raw_est = parsed_responses.get(f"Q{q_est_idx}")
                    
                    if raw_choice and raw_est:
                        try:
                            # Normalize Choice (A/B) and Estimate (Numeric)
                            # Extract and validate choice (handles "A", "B", "Option A", etc.)
                            choice_raw = raw_choice.upper().strip()
                            # Find positions of A and B
                            pos_a = choice_raw.find("A")
                            pos_b = choice_raw.find("B")
                            
                            if pos_a != -1 and pos_b != -1:
                                # Both A and B found - use the one that appears first
                                choice = "A" if pos_a < pos_b else "B"
                            elif pos_a != -1:
                                choice = "A"
                            elif pos_b != -1:
                                choice = "B"
                            else:
                                # Fallback: use first character if it's A or B
                                first_char = choice_raw[0] if choice_raw else None
                                if first_char in ["A", "B"]:
                                    choice = first_char
                                else:
                                    # Invalid choice - skip this record
                                    continue
                            
                            est = float(raw_est)
                            aggregated[sub_study_id][gt_key].append({
                                "choice": choice,
                                "estimate_cat1": est
                            })
                        except (ValueError, IndexError):
                            continue
            else:
                # Study 1 and 3: Scenario-based (one item per trial)
                if not items:
                    continue
                item = items[0]
                
                # Re-calculate indices if missing
                if 'q_idx_choice' not in item:
                    item['q_idx_est_a'] = 1
                    item['q_idx_est_b'] = 2
                    item['q_idx_choice'] = 3
                    
                    # Trait ratings start at Q4
                    traits = item.get("metadata", {}).get("traits_to_rate", [])
                    if traits:
                        item["trait_q_map"] = {}
                        q_curr = 4
                        for trait in traits:
                            item["trait_q_map"][trait] = {"opt_a": q_curr, "opt_b": q_curr + 1}
                            q_curr += 2
                
                # Use scenario_id as key for mapping to GT
                if scenario_id not in aggregated[sub_study_id]:
                    aggregated[sub_study_id][scenario_id] = []
                
                q_choice_idx = item.get('q_idx_choice')
                q_est_a_idx = item.get('q_idx_est_a')
                q_est_b_idx = item.get('q_idx_est_b')
                
                raw_choice = parsed_responses.get(f"Q{q_choice_idx}")
                raw_est_a = parsed_responses.get(f"Q{q_est_a_idx}")
                raw_est_b = parsed_responses.get(f"Q{q_est_b_idx}")
                
                if raw_choice and raw_est_a:
                    # print(f"DEBUG: Found data for scenario {scenario_id}: Choice={raw_choice}, EstA={raw_est_a}")
                    try:
                        # Normalize Choice (A/B) - same logic as Study 2
                        choice_raw = raw_choice.upper().strip()
                        # Find positions of A and B
                        pos_a = choice_raw.find("A")
                        pos_b = choice_raw.find("B")
                        
                        if pos_a != -1 and pos_b != -1:
                            # Both A and B found - use the one that appears first
                            choice = "A" if pos_a < pos_b else "B"
                        elif pos_a != -1:
                            choice = "A"
                        elif pos_b != -1:
                            choice = "B"
                        else:
                            # Fallback: use first character if it's A or B
                            first_char = choice_raw[0] if choice_raw else None
                            if first_char in ["A", "B"]:
                                choice = first_char
                            else:
                                # Invalid choice - skip this record
                                continue
                        
                        est_a = float(raw_est_a)
                        # Estimate B is often 100 - A, but we parse it if available
                        est_b = float(raw_est_b) if raw_est_b else (100.0 - est_a)
                        
                        # Process Trait Ratings (for Findings F2 and F5)
                        trait_map = item.get("trait_q_map", {})
                        sum_abs_diff_a = 0
                        sum_abs_diff_b = 0
                        trait_count = 0
                        for trait, qs in trait_map.items():
                            val_a = parsed_responses.get(f"Q{qs['opt_a']}")
                            val_b = parsed_responses.get(f"Q{qs['opt_b']}")
                            if val_a and val_b:
                                try:
                                    # Extremity score: distance from midpoint (50)
                                    sum_abs_diff_a += abs(float(val_a) - 50)
                                    sum_abs_diff_b += abs(float(val_b) - 50)
                                    trait_count += 1
                                except ValueError:
                                    pass
                        
                        aggregated[sub_study_id][scenario_id].append({
                            "choice": choice,
                            "est_a": est_a,
                            "est_b": est_b,
                            "trait_diff_score": (sum_abs_diff_a - sum_abs_diff_b) if trait_count > 0 else None
                        })
                    except (ValueError, IndexError):
                        pass

    # 3. Statistical Replication and PAS Calculation
    test_results = []

    def normalize_key(k):
        """Helper to match scenario keys between Agent and GT."""
        return re.sub(r'[^a-z0-9]', '', str(k).lower().replace("version", ""))

    for study_gt in ground_truth.get("studies", []):
        study_label = study_gt.get("study_id") # e.g., "Study 1"
        for finding in study_gt.get("findings", []):
            finding_id = finding.get("finding_id") # e.g., "F1"
            
            # Get statistical test from finding (for SRS fields)
            # Study 001 typically has one test per finding
            statistical_tests = finding.get("statistical_tests", [])
            test_gt = statistical_tests[0] if statistical_tests else {}
            
            # Map Study Label to Sub-study ID
            if study_label == "Study 1":
                sub_study_id = "study_1_hypothetical_stories"
            elif study_label == "Study 2":
                sub_study_id = "study_2_personal_description_items"
            else:
                sub_study_id = "study_3_sandwich_board_hypothetical"

            gt_data_points = finding.get("original_data_points", {}).get("data", {})
            
            for scenario_key, scenario_gt in gt_data_points.items():
                # 3.1 Match Agent data to this scenario
                agent_scenario_key = None
                norm_scenario_key = normalize_key(scenario_key)
                
                if sub_study_id in aggregated:
                    for k in aggregated[sub_study_id].keys():
                        if normalize_key(k) == norm_scenario_key:
                            agent_scenario_key = k
                            break
                
                # 3.2 Calculate pi_human (3-way)
                pi_human = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                pi_human_source = "No statistical source"
                
                # Human expected direction: Studies 1, 2, 3 findings are typically positive (FCE)
                h_dir = 1
                
                # Check for item-level statistics (Preferred)
                if "consensus_f" in scenario_gt or "trait_f" in scenario_gt:
                    h_stat = scenario_gt.get("consensus_f") or scenario_gt.get("trait_f")
                    n1 = scenario_gt.get("n_choice_1") or scenario_gt.get("n_wear") or 40
                    n2 = scenario_gt.get("n_choice_2") or scenario_gt.get("n_not_wear") or 40
                    df2 = (n1 + n2) - 2
                    bf_h = calc_bf_anova(float(h_stat), 1, df2, n1 + n2)
                    pi_human = calc_posteriors_3way(bf_h, h_dir, prior_odds=10.0)
                    pi_human_source = f"F(1, {df2})={h_stat}"
                elif "t" in scenario_gt:
                    h_stat = scenario_gt["t"]
                    bf_h = calc_bf_t(float(h_stat), 40, 40)
                    pi_human = calc_posteriors_3way(bf_h, h_dir, prior_odds=10.0)
                    pi_human_source = f"t={h_stat}"
                else:
                    # Fallback to finding-level reported statistics
                    rep_stats = finding.get("statistical_tests", [{}])[0].get("reported_statistics", "")
                    if "F(" in rep_stats:
                        parts = re.findall(r"F\((\d+),\s*(\d+)\)\s*=\s*([\d\.]+)", rep_stats)
                        if parts:
                            df1, df2, f_val = parts[0]
                            bf_h = calc_bf_anova(float(f_val), int(df1), int(df2), int(df1)+int(df2)+1)
                            pi_human = calc_posteriors_3way(bf_h, h_dir, prior_odds=10.0)
                            pi_human_source = f"F({df1}, {df2})={f_val}"

                # 3.3 Calculate pi_agent (3-way)
                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                agent_reason = "No agent data found"
                p_val_agent = None
                t_stat_agent = None
                n_agent_1 = None
                n_agent_2 = None
                
                if agent_scenario_key and agent_scenario_key in aggregated[sub_study_id]:
                    data = aggregated[sub_study_id][agent_scenario_key]
                    
                    if finding_id in ["F1", "F3", "F4"]:
                        # False Consensus Effect: Compare estimates of Choice A between A-choosers and B-choosers
                        group_a = []
                        group_b = []
                        for d in data:
                            val = d.get("est_a") if finding_id != "F3" else d.get("estimate_cat1")
                            if val is not None:
                                if d["choice"] == "A": group_a.append(val)
                                else: group_b.append(val)
                        
                        # Check for potential extraction issues and failure cases
                        total_valid = len(group_a) + len(group_b)
                        extraction_warning = ""
                        is_failure_case = False
                        
                        if total_valid > 0:
                            # Flag if one group is empty or extremely small relative to the other
                            if len(group_b) == 0 and len(group_a) > 0:
                                extraction_warning = f" [EXTRACTION WARNING: nB=0 suggests all participants chose A - possible extraction issue]"
                                is_failure_case = True  # Cannot test False Consensus Effect without both groups
                            elif len(group_a) == 0 and len(group_b) > 0:
                                extraction_warning = f" [EXTRACTION WARNING: nA=0 suggests all participants chose B - possible extraction issue]"
                                is_failure_case = True  # Cannot test False Consensus Effect without both groups
                            elif len(group_a) > 0 and len(group_b) > 0:
                                ratio = min(len(group_a), len(group_b)) / max(len(group_a), len(group_b))
                                if ratio < 0.1:  # One group is less than 10% of the other
                                    extraction_warning = f" [EXTRACTION WARNING: Extreme imbalance (ratio={ratio:.2f}) - possible extraction issue]"
                                    is_failure_case = True  # Insufficient variation to test effect
                        
                        if len(group_a) >= 2 and len(group_b) >= 2:
                            t_stat_agent, p_val_agent = stats.ttest_ind(group_a, group_b)
                            n_agent_1 = len(group_a)
                            n_agent_2 = len(group_b)
                            if not np.isnan(t_stat_agent):
                                bf_a = calc_bf_t(t_stat_agent, len(group_a), len(group_b))
                                a_dir = 1 if t_stat_agent > 0 else -1
                                pi_agent = calc_posteriors_3way(bf_a, a_dir)
                                agent_reason = f"t({len(group_a)+len(group_b)-2})={t_stat_agent:.3f}{extraction_warning}"
                            else:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                agent_reason = f"T-test failed (NaN){extraction_warning}"
                                n_agent_1 = None
                                n_agent_2 = None
                        else:
                            # Distinguish between failure cases and uncertainty
                            if is_failure_case:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                            else:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 0.5}  # Uncertainty
                            agent_reason = f"Insufficient data: nA={len(group_a)}, nB={len(group_b)}{extraction_warning}"
                            n_agent_1 = len(group_a) if len(group_a) >= 2 else None
                            n_agent_2 = len(group_b) if len(group_b) >= 2 else None
                            
                    elif finding_id in ["F2", "F5"]:
                        # Trait Attribution: Compare trait extremity scores between A-choosers and B-choosers
                        group_a = [d["trait_diff_score"] for d in data if d["choice"] == "A" and d["trait_diff_score"] is not None]
                        group_b = [d["trait_diff_score"] for d in data if d["choice"] == "B" and d["trait_diff_score"] is not None]
                        
                        # Check for potential extraction issues and failure cases
                        total_valid = len(group_a) + len(group_b)
                        extraction_warning = ""
                        is_failure_case = False
                        
                        if total_valid > 0:
                            # Flag if one group is empty or extremely small relative to the other
                            if len(group_b) == 0 and len(group_a) > 0:
                                extraction_warning = f" [EXTRACTION WARNING: nB=0 suggests all participants chose A - possible extraction issue]"
                                is_failure_case = True  # Cannot test False Consensus Effect without both groups
                            elif len(group_a) == 0 and len(group_b) > 0:
                                extraction_warning = f" [EXTRACTION WARNING: nA=0 suggests all participants chose B - possible extraction issue]"
                                is_failure_case = True  # Cannot test False Consensus Effect without both groups
                            elif len(group_a) > 0 and len(group_b) > 0:
                                ratio = min(len(group_a), len(group_b)) / max(len(group_a), len(group_b))
                                if ratio < 0.1:  # One group is less than 10% of the other
                                    extraction_warning = f" [EXTRACTION WARNING: Extreme imbalance (ratio={ratio:.2f}) - possible extraction issue]"
                                    is_failure_case = True  # Insufficient variation to test effect
                        
                        if len(group_a) >= 2 and len(group_b) >= 2:
                            t_stat_agent, p_val_agent = stats.ttest_ind(group_a, group_b)
                            n_agent_1 = len(group_a)
                            n_agent_2 = len(group_b)
                            if not np.isnan(t_stat_agent):
                                bf_a = calc_bf_t(t_stat_agent, len(group_a), len(group_b))
                                a_dir = 1 if t_stat_agent > 0 else -1
                                pi_agent = calc_posteriors_3way(bf_a, a_dir)
                                agent_reason = f"t({len(group_a)+len(group_b)-2})={t_stat_agent:.3f}{extraction_warning}"
                            else:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                agent_reason = f"T-test failed (NaN){extraction_warning}"
                                n_agent_1 = None
                                n_agent_2 = None
                        else:
                            # Distinguish between failure cases and uncertainty
                            if is_failure_case:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                            else:
                                pi_agent = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 0.5}  # Uncertainty
                            agent_reason = f"Insufficient data: nA={len(group_a)}, nB={len(group_b)}{extraction_warning}"
                            n_agent_1 = len(group_a) if len(group_a) >= 2 else None
                            n_agent_2 = len(group_b) if len(group_b) >= 2 else None

                # Get test weight (default to 1.0 if not found)
                test_name = f"{scenario_key} - {finding_id}"
                test_weight = test_weights.get((finding_id, test_name), 1.0)
                
                # Extract human sample sizes from ground truth data
                n_human_1 = None
                n_human_2 = None
                
                # Get relevant study GT to find sample sizes for other findings
                study1_gt = ground_truth["studies"][0]
                study2_gt = ground_truth["studies"][1]
                study3_gt = ground_truth["studies"][2]
                
                if finding_id == "F1":
                    n_human_1 = scenario_gt.get("n_choice_1")
                    n_human_2 = scenario_gt.get("n_choice_2")
                elif finding_id == "F2":
                    # F2 uses same participants as F1
                    f1_gt = study1_gt["findings"][0]["original_data_points"]["data"].get(scenario_key, {})
                    n_human_1 = f1_gt.get("n_choice_1")
                    n_human_2 = f1_gt.get("n_choice_2")
                elif finding_id == "F3":
                    # Study 2 has N=80 total, but per-item n1/n2 not in JSON. 
                    # Use balanced approximation (40/40) for SE calculation.
                    n_human_1 = 40
                    n_human_2 = 40
                elif finding_id == "F4":
                    n_human_1 = scenario_gt.get("n_wear")
                    n_human_2 = scenario_gt.get("n_not_wear")
                elif finding_id == "F5":
                    # F5 uses same participants as F4
                    f4_gt = study3_gt["findings"][0]["original_data_points"]["data"].get(scenario_key, {})
                    n_human_1 = f4_gt.get("n_wear")
                    n_human_2 = f4_gt.get("n_not_wear")
                
                # Determine test type (f-test for F1, F2, F4, F5; t-test for F3)
                statistical_test_type = "t-test"
                if finding_id in ["F1", "F2", "F4", "F5"]:
                    statistical_test_type = "f-test"

                # Create test result dict
                test_result = {
                    "study_id": study_label,
                    "sub_study_id": sub_study_id,
                    "finding_id": finding_id,
                    "test_name": test_name,
                    "scenario": scenario_key,
                    "pi_human": float(pi_human['pi_plus'] + pi_human['pi_minus']),
                    "pi_agent": float(pi_agent['pi_plus'] + pi_agent['pi_minus']),
                    "pi_human_3way": pi_human,
                    "pi_agent_3way": pi_agent,
                    "pas": float(calc_pas(pi_human, pi_agent)),
                    "test_weight": float(test_weight),
                    "pi_human_source": pi_human_source,
                    "agent_reason": agent_reason,
                    "statistical_test_type": statistical_test_type,
                    "human_test_statistic": pi_human_source.split("=")[-1].split(",")[0] if "=" in pi_human_source else "",
                    "agent_test_statistic": f"{t_stat_agent:.3f}" if t_stat_agent is not None and not np.isnan(t_stat_agent) else ""
                }
                
                # Add statistical replication fields with sample sizes for frequentist consistency
                add_statistical_replication_fields(
                    test_result, test_gt, p_val_agent, t_stat_agent, statistical_test_type,
                    n_agent=n_agent_1, n2_agent=n_agent_2,
                    n_human=n_human_1, n2_human=n_human_2,
                    independent=True
                )
                
                test_results.append(test_result)

    # 4. Two-level Weighted Aggregation
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

    # substudy_results removed - using two-level aggregation (Tests -> Findings -> Study)
    substudy_results = []

    return {
        "score": float(overall_score),
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }