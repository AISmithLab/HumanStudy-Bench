#!/usr/bin/env python3
"""
Generate detailed metrics table with all hierarchical values (global/study/finding/subfield).

This script generates LaTeX tables showing:
1. All r correlation parameters (a and b) at global/study/finding/subfield levels
2. Global consistency (p_null) after Fisher-z pooling at all levels
3. All other metrics (PAS, ECS, APR) at all hierarchical levels
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
from scipy import stats

# Study groups
STUDY_GROUPS = {
    "Cognition": ["study_001", "study_002", "study_003", "study_004"],
    "Strategic": ["study_009", "study_010", "study_011", "study_012"],
    "Social": ["study_005", "study_006", "study_007", "study_008"]
}

METHOD_DISPLAY_MAP = {
    "v1-empty": "A1",
    "v2-human": "A2",
    "v3-human-plus-demo": "A3",
    "v4-background": "A4"
}


def fisher_z_transform(r: float) -> float:
    """Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))"""
    if abs(r) >= 1.0:
        return float('inf') if r > 0 else float('-inf')
    if abs(r) < 1e-10:
        return 0.0
    return 0.5 * math.log((1 + r) / (1 - r))


def fisher_z_inverse(z: float) -> float:
    """Inverse Fisher z-transform: r = tanh(z)"""
    if math.isinf(z) or math.isnan(z):
        return 1.0 if z > 0 else -1.0
    return math.tanh(z)


def compute_p_null_from_correlation(r: float, n: int) -> float:
    """
    Compute p-value for testing H0: rho = 0 (p_null).
    
    Uses Fisher-z transform and z-test:
    z = z_r / SE_z, where SE_z = 1/sqrt(n-3)
    p_null = 2 * (1 - Phi(|z|))
    """
    if r is None or math.isnan(r) or abs(r) >= 1.0:
        return None
    
    if n < 4:
        return None  # Need at least 4 samples for Fisher-z test
    
    # Fisher transform
    z_r = fisher_z_transform(r)
    if math.isinf(z_r) or math.isnan(z_r):
        return None
    
    # Standard error: SE_z = 1/sqrt(n-3)
    se_z = 1.0 / math.sqrt(n - 3)
    if se_z <= 0:
        return None
    
    # z-test statistic
    z_stat = z_r / se_z
    
    # Two-tailed p-value
    p_null = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(p_null)


def compute_p_null_fisher_pooled(z_values: List[float], weights: List[float]) -> float:
    """
    Compute p_null after Fisher-z pooling.
    
    Formula:
    1. Weighted average in z-space: z_bar = sum(z_i * w_i) / sum(w_i)
    2. Combined SE: SE_combined = 1/sqrt(sum(w_i) - 3)
    3. z-test: z_stat = z_bar / SE_combined
    4. p_null = 2 * (1 - Phi(|z_stat|))
    """
    if not z_values or not weights or len(z_values) != len(weights):
        return None
    
    # Filter valid values
    valid = [(z, w) for z, w in zip(z_values, weights) 
             if not (math.isnan(z) or math.isinf(z) or math.isnan(w) or w <= 0)]
    
    if not valid:
        return None
    
    z_vals, w_vals = zip(*valid)
    
    # Weighted average in z-space
    total_weight = sum(w_vals)
    if total_weight <= 0:
        return None
    
    z_bar = sum(z * w for z, w in zip(z_vals, w_vals)) / total_weight
    
    # Combined standard error: SE = 1/sqrt(n_eff - 3)
    # where n_eff is the effective sample size (sum of weights)
    n_eff = total_weight
    if n_eff < 4:
        return None
    
    se_combined = 1.0 / math.sqrt(n_eff - 3)
    
    # z-test statistic
    z_stat = z_bar / se_combined
    
    # Two-tailed p-value
    p_null = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(p_null)


def load_evaluation_results(results_dir: Path, model: str, method: str) -> Dict:
    """Load evaluation_results.json for a specific model-method combination."""
    # Try to find the evaluation_results.json file
    # Structure: results/benchmark/{study_id}/{model}_{method}/evaluation_results.json
    all_results = {}
    
    benchmark_dir = results_dir / "benchmark"
    if not benchmark_dir.exists():
        return all_results
    
    for study_dir in benchmark_dir.glob("study_*"):
        study_id = study_dir.name
        config_dir = study_dir / f"{model}_{method}"
        
        eval_file = config_dir / "evaluation_results.json"
        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_results[study_id] = data
            except Exception as e:
                print(f"Warning: Failed to load {eval_file}: {e}")
    
    return all_results


def aggregate_metrics_by_level(evaluation_results: Dict[str, Dict]) -> Dict:
    """
    Aggregate metrics at all hierarchical levels (global/study/finding/subfield).
    
    Returns a nested dictionary with all metrics at each level.
    """
    # Collect all test results
    all_test_results = []
    study_test_map = defaultdict(list)
    finding_test_map = defaultdict(list)
    domain_test_map = defaultdict(lambda: defaultdict(list))
    
    for study_id, eval_data in evaluation_results.items():
        test_results = eval_data.get("test_results", [])
        
        for test in test_results:
            # Use the study_id from the test itself if available, otherwise use directory name
            test_study_id = test.get("study_id", study_id)
            # Normalize study_id: convert "Study 004" -> "study_004" for matching
            # But keep original for display
            test["_study_id"] = test_study_id
            test["_study_id_normalized"] = study_id  # Use directory name for grouping
            
            finding_id = test.get("finding_id", "unknown")
            test["_finding_id"] = f"{test_study_id}_{finding_id}"
            
            all_test_results.append(test)
            # Use normalized study_id for grouping
            study_test_map[test_study_id].append(test)
            finding_test_map[f"{test_study_id}_{finding_id}"].append(test)
            
            # Map to domain
            for domain, study_list in STUDY_GROUPS.items():
                if study_id in study_list:
                    domain_test_map[domain][study_id].append(test)
                    break
    
    result = {
        "global": {},
        "study": {},
        "finding": {},
        "domain": {}
    }
    
    # === GLOBAL LEVEL ===
    if all_test_results:
        # Convert effect_size to effect_d if needed
        from src.evaluation.stats_lib import FrequentistConsistency
        for test in all_test_results:
            # If effect_d is missing but effect_size exists, convert it
            if test.get("human_effect_d") is None and test.get("human_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                h_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("human_effect_size"))
                if h_d is not None:
                    test["human_effect_d"] = h_d
            
            if test.get("agent_effect_d") is None and test.get("agent_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                a_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("agent_effect_size"))
                if a_d is not None:
                    test["agent_effect_d"] = a_d
        
        # ECS (CCC-based) overall
        from src.evaluation.stats_lib import compute_ecs_corr
        ecs_result = compute_ecs_corr(all_test_results, STUDY_GROUPS)
        result["global"]["ecs_corr"] = ecs_result.get("ecs_overall")  # CCC-based ECS (main metric)
        result["global"]["ecs_pearson_r"] = ecs_result.get("ecs_corr_overall")  # Pearson r (retained for reference)
        result["global"]["caricature_a"] = ecs_result.get("caricature_overall", {}).get("a")
        result["global"]["caricature_b"] = ecs_result.get("caricature_overall", {}).get("b")
        result["global"]["n_tests"] = ecs_result.get("n_tests_overall", 0)
        
        # Compute p_null for global correlation (using CCC for main metric, but can use Pearson for reference)
        if result["global"]["ecs_corr"] is not None:
            r_global = result["global"]["ecs_corr"]  # CCC
            n_global = result["global"]["n_tests"]
            result["global"]["p_null"] = compute_p_null_from_correlation(r_global, n_global)
        
        # PAS overall (simple mean of study scores)
        study_scores = []
        for study_id in set(t.get("_study_id") for t in all_test_results):
            study_tests = study_test_map.get(study_id, [])
            if study_tests:
                from src.evaluation.stats_lib import aggregate_study_pas
                pas_raw, pas_norm, _ = aggregate_study_pas(study_tests)
                study_scores.append({"pas_raw": pas_raw, "pas_norm": pas_norm})
        
        if study_scores:
            result["global"]["pas_raw"] = np.mean([s["pas_raw"] for s in study_scores])
            result["global"]["pas_norm"] = np.mean([s["pas_norm"] for s in study_scores])
        
        # APR overall
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from scripts.calculate_mixed_models_apr_ecs import calculate_apr_ecs
        # Combine all evaluation results into one dict for APR calculation
        combined_eval = {
            "test_results": all_test_results,
            "finding_results": []
        }
        # Collect finding results
        for finding_id in set(t.get("_finding_id") for t in all_test_results):
            finding_tests = finding_test_map.get(finding_id, [])
            if finding_tests:
                # Extract original finding_id from test (without study prefix)
                original_finding_id = finding_tests[0].get("finding_id", "unknown")
                # Create a finding result entry
                combined_eval["finding_results"].append({
                    "finding_id": original_finding_id,
                    "finding_weight": 1.0
                })
        
        apr_metrics = calculate_apr_ecs(combined_eval)
        result["global"]["apr"] = apr_metrics.get("apr", 0.0)
    
    # === STUDY LEVEL ===
    for study_id, tests in study_test_map.items():
        if not tests:
            continue
        
        # Convert effect_size to effect_d if needed
        from src.evaluation.stats_lib import FrequentistConsistency
        for test in tests:
            if test.get("human_effect_d") is None and test.get("human_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                h_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("human_effect_size"))
                if h_d is not None:
                    test["human_effect_d"] = h_d
            
            if test.get("agent_effect_d") is None and test.get("agent_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                a_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("agent_effect_size"))
                if a_d is not None:
                    test["agent_effect_d"] = a_d
        
        study_result = {}
        
        # ECS (CCC-based) per study
        from src.evaluation.stats_lib import compute_ecs_corr
        ecs_result = compute_ecs_corr(tests, None)
        study_result["ecs_corr"] = ecs_result.get("ecs_per_study", {}).get(study_id)  # CCC-based ECS (main metric)
        study_result["ecs_pearson_r"] = ecs_result.get("ecs_corr_per_study", {}).get(study_id)  # Pearson r (retained for reference)
        study_result["caricature_a"] = ecs_result.get("caricature_per_study", {}).get(study_id, {}).get("a")
        study_result["caricature_b"] = ecs_result.get("caricature_per_study", {}).get(study_id, {}).get("b")
        study_result["n_tests"] = len(tests)
        
        # p_null for study correlation (using CCC for main metric)
        if study_result["ecs_corr"] is not None:
            r_study = study_result["ecs_corr"]  # CCC
            n_study = study_result["n_tests"]
            study_result["p_null"] = compute_p_null_from_correlation(r_study, n_study)
        
        # PAS per study
        from src.evaluation.stats_lib import aggregate_study_pas
        pas_raw, pas_norm, _ = aggregate_study_pas(tests)
        study_result["pas_raw"] = pas_raw
        study_result["pas_norm"] = pas_norm
        
        # APR per study
        # calculate_apr_ecs already imported above
        # Collect unique finding IDs from tests
        unique_finding_ids = set(t.get("finding_id", "unknown") for t in tests)
        study_eval = {
            "test_results": tests,
            "finding_results": [{"finding_id": fid, "finding_weight": 1.0} 
                                for fid in unique_finding_ids]
        }
        apr_metrics = calculate_apr_ecs(study_eval)
        study_result["apr"] = apr_metrics.get("apr", 0.0)
        
        result["study"][study_id] = study_result
    
    # === FINDING LEVEL ===
    for finding_id, tests in finding_test_map.items():
        if not tests:
            continue
        
        finding_result = {}
        
        # Convert effect_size to effect_d if needed
        from src.evaluation.stats_lib import FrequentistConsistency
        for test in tests:
            if test.get("human_effect_d") is None and test.get("human_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                h_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("human_effect_size"))
                if h_d is not None:
                    test["human_effect_d"] = h_d
            
            if test.get("agent_effect_d") is None and test.get("agent_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                a_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("agent_effect_size"))
                if a_d is not None:
                    test["agent_effect_d"] = a_d
        
        # ECS_corr per finding (if >= 2 tests)
        if len(tests) >= 2:
            # For finding level, we need to compute correlation manually
            h_effects = [t.get("human_effect_d") for t in tests if t.get("human_effect_d") is not None]
            a_effects = [t.get("agent_effect_d") for t in tests if t.get("agent_effect_d") is not None]
            
            if len(h_effects) >= 2 and len(a_effects) >= 2:
                # Simple correlation (unweighted)
                # Guard against zero variance to avoid numpy RuntimeWarning
                h_std = np.std(h_effects)
                a_std = np.std(a_effects)
                if len(h_effects) == len(a_effects) and h_std > 1e-10 and a_std > 1e-10:
                    # Suppress warnings for edge cases (e.g., near-zero variance)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r_finding = np.corrcoef(h_effects, a_effects)[0, 1]
                    finding_result["ecs_corr"] = float(r_finding) if r_finding is not None and not (math.isnan(r_finding) or math.isinf(r_finding)) else None
                else:
                    finding_result["ecs_corr"] = None
                
                # Caricature regression
                if finding_result["ecs_corr"] is not None and len(h_effects) == len(a_effects):
                    from src.evaluation.stats_lib import weighted_linreg
                    a_finding, b_finding = weighted_linreg(h_effects, a_effects, [1.0] * len(h_effects))
                    if a_finding is not None and b_finding is not None:
                        finding_result["caricature_a"] = float(a_finding)
                        finding_result["caricature_b"] = float(b_finding)
                    else:
                        finding_result["caricature_a"] = None
                        finding_result["caricature_b"] = None
                    
                    # p_null for finding correlation
                    finding_result["p_null"] = compute_p_null_from_correlation(r_finding, len(h_effects))
        
        finding_result["n_tests"] = len(tests)
        
        # PAS per finding
        from src.evaluation.stats_lib import aggregate_finding_pas_raw, aggregate_finding_pas_norm
        finding_result["pas_raw"] = aggregate_finding_pas_raw(tests)
        finding_result["pas_norm"] = aggregate_finding_pas_norm(tests)
        
        # APR per finding
        sig_count = sum(1 for t in tests if t.get("is_significant_agent") and t.get("direction_match"))
        finding_result["apr"] = sig_count / len(tests) if tests else 0.0
        
        result["finding"][finding_id] = finding_result
    
    # === DOMAIN LEVEL ===
    for domain, study_map in domain_test_map.items():
        domain_tests = []
        for study_tests in study_map.values():
            domain_tests.extend(study_tests)
        
        if not domain_tests:
            continue
        
        # Convert effect_size to effect_d if needed
        from src.evaluation.stats_lib import FrequentistConsistency
        for test in domain_tests:
            if test.get("human_effect_d") is None and test.get("human_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                h_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("human_effect_size"))
                if h_d is not None:
                    test["human_effect_d"] = h_d
            
            if test.get("agent_effect_d") is None and test.get("agent_effect_size") is not None:
                test_type = test.get("statistical_test_type", "t-test")
                a_d = FrequentistConsistency.effect_to_d_equiv(test_type, test.get("agent_effect_size"))
                if a_d is not None:
                    test["agent_effect_d"] = a_d
        
        domain_result = {}
        
        # ECS (CCC-based) per domain
        from src.evaluation.stats_lib import compute_ecs_corr
        ecs_result = compute_ecs_corr(domain_tests, {domain: list(study_map.keys())})
        domain_result["ecs_corr"] = ecs_result.get("ecs_domain", {}).get(domain)  # CCC-based ECS (main metric)
        domain_result["ecs_pearson_r"] = ecs_result.get("ecs_corr_domain", {}).get(domain)  # Pearson r (retained for reference)
        domain_result["caricature_a"] = ecs_result.get("caricature_domain", {}).get(domain, {}).get("a")
        domain_result["caricature_b"] = ecs_result.get("caricature_domain", {}).get(domain, {}).get("b")
        domain_result["n_tests"] = len(domain_tests)
        
        # p_null for domain correlation (using CCC for main metric)
        if domain_result["ecs_corr"] is not None:
            r_domain = domain_result["ecs_corr"]  # CCC
            n_domain = domain_result["n_tests"]
            domain_result["p_null"] = compute_p_null_from_correlation(r_domain, n_domain)
        
        # PAS per domain (mean of study scores in domain)
        domain_study_scores = []
        for study_id in study_map.keys():
            if study_id in result["study"]:
                domain_study_scores.append({
                    "pas_raw": result["study"][study_id].get("pas_raw"),
                    "pas_norm": result["study"][study_id].get("pas_norm")
                })
        
        if domain_study_scores:
            domain_result["pas_raw"] = np.mean([s["pas_raw"] for s in domain_study_scores if s["pas_raw"] is not None])
            domain_result["pas_norm"] = np.mean([s["pas_norm"] for s in domain_study_scores if s["pas_norm"] is not None])
        
        # APR per domain (weighted average)
        domain_apr_scores = [result["study"][s].get("apr") for s in study_map.keys() if s in result["study"]]
        if domain_apr_scores:
            domain_result["apr"] = np.mean([a for a in domain_apr_scores if a is not None])
        
        result["domain"][domain] = domain_result
    
    return result


def generate_detailed_metrics_table(organized_data: Dict, results_dir: Path) -> str:
    """Generate a comprehensive table with all metrics at all hierarchical levels."""
    lines = []
    
    lines.append("\\begin{table*}[p]")
    lines.append("\\centering")
    lines.append("\\caption{Comprehensive Metrics at All Hierarchical Levels}")
    lines.append("\\label{tab:detailed-metrics-all-levels}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}ll")
    lines.append("S[table-format=1.3]")  # PAS_raw
    lines.append("S[table-format=1.3]")  # PAS_norm
    lines.append("S[table-format=1.3]")  # ECS_corr
    lines.append("S[table-format=+1.3]")  # Caricature a
    lines.append("S[table-format=+1.3]")  # Caricature b
    lines.append("S[table-format=1.4]")  # p_null
    lines.append("S[table-format=1.3]")  # APR
    lines.append("S[table-format=2.0]")  # n_tests
    lines.append("@{}")
    lines.append("}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Level} & \\textbf{PAS (Raw)} & \\textbf{PAS (Norm)} & \\textbf{ECS\\_corr} & \\textbf{Slope $a$} & \\textbf{Intercept $b$} & \\textbf{$p_{\\text{null}}$} & \\textbf{APR} & \\textbf{$n$} \\\\")
    lines.append("\\midrule")
    
    # Process each model-method combination
    base_models_sorted = sorted(organized_data.keys())
    for base_model in base_models_sorted:
        if 'temp' in base_model.lower():
            continue
        
        methods = organized_data[base_model]
        method_order = ["v1-empty", "v2-human", "v3-human-plus-demo", "v4-background"]
        sorted_methods = sorted(methods.keys(), key=lambda x: method_order.index(x) if x in method_order else 999)
        
        for method in sorted_methods:
            m_data = methods[method]
            model_key = f"{base_model}_{method}"
            method_display = METHOD_DISPLAY_MAP.get(method, method)
            
            # Load evaluation results
            eval_results = load_evaluation_results(results_dir, base_model, method)
            if not eval_results:
                continue
            
            # Aggregate metrics
            metrics = aggregate_metrics_by_level(eval_results)
            
            # Format model name
            model_display = base_model.replace("_", " ").title()
            
            # Helper function to format values with None handling
            def fmt_val(val, fmt_str, default=0.0):
                if val is None:
                    return fmt_str.format(default)
                return fmt_str.format(val)
            
            # GLOBAL LEVEL
            g = metrics.get("global", {})
            pas_raw = g.get('pas_raw') or 0.0
            pas_norm = g.get('pas_norm') or 0.0
            ecs_corr = g.get('ecs_corr') or 0.0
            car_a = g.get('caricature_a') or 0.0
            car_b = g.get('caricature_b') or 0.0
            p_null = g.get('p_null') or 1.0
            apr = g.get('apr') or 0.0
            n_tests = g.get('n_tests') or 0
            
            lines.append(f"{model_display} & Global & "
                        f"{pas_raw:.3f} & "
                        f"{pas_norm:.3f} & "
                        f"{ecs_corr:.3f} & "
                        f"{car_a:+.3f} & "
                        f"{car_b:+.3f} & "
                        f"{p_null:.4f} & "
                        f"{apr:.3f} & "
                        f"{n_tests:.0f} \\\\")
            
            # DOMAIN LEVEL
            for domain in ["Cognition", "Strategic", "Social"]:
                d = metrics.get("domain", {}).get(domain, {})
                if d:
                    pas_raw = d.get('pas_raw') or 0.0
                    pas_norm = d.get('pas_norm') or 0.0
                    ecs_corr = d.get('ecs_corr') or 0.0
                    car_a = d.get('caricature_a') or 0.0
                    car_b = d.get('caricature_b') or 0.0
                    p_null = d.get('p_null') or 1.0
                    apr = d.get('apr') or 0.0
                    n_tests = d.get('n_tests') or 0
                    
                    lines.append(f" & {domain} & "
                                f"{pas_raw:.3f} & "
                                f"{pas_norm:.3f} & "
                                f"{ecs_corr:.3f} & "
                                f"{car_a:+.3f} & "
                                f"{car_b:+.3f} & "
                                f"{p_null:.4f} & "
                                f"{apr:.3f} & "
                                f"{n_tests:.0f} \\\\")
            
            # STUDY LEVEL (show first few studies as example)
            study_items = list(metrics.get("study", {}).items())[:3]  # Limit to 3 studies per method
            for study_id, s in study_items:
                study_display = study_id.replace("_", " ").title()
                # Escape LaTeX special characters (especially &)
                study_display = study_display.replace("&", "\\&")
                pas_raw = s.get('pas_raw') or 0.0
                pas_norm = s.get('pas_norm') or 0.0
                ecs_corr = s.get('ecs_corr') or 0.0
                car_a = s.get('caricature_a') or 0.0
                car_b = s.get('caricature_b') or 0.0
                p_null = s.get('p_null') or 1.0
                apr = s.get('apr') or 0.0
                n_tests = s.get('n_tests') or 0
                
                lines.append(f" & {study_display} & "
                            f"{pas_raw:.3f} & "
                            f"{pas_norm:.3f} & "
                            f"{ecs_corr:.3f} & "
                            f"{car_a:+.3f} & "
                            f"{car_b:+.3f} & "
                            f"{p_null:.4f} & "
                            f"{apr:.3f} & "
                            f"{n_tests:.0f} \\\\")
            
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def main():
    import sys
    from pathlib import Path
    # Add project root to path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="results/benchmark_summary.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-latex", default="results/detailed_metrics_table.tex")
    args = parser.parse_args()
    
    # Load data
    from scripts.generate_production_results import load_benchmark_data, organize_data_by_model_method
    data = load_benchmark_data(Path(args.summary_json))
    organized = organize_data_by_model_method(data)
    
    # Generate table
    table_latex = generate_detailed_metrics_table(organized, Path(args.results_dir))
    
    # Write output
    with open(args.output_latex, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage[table]{xcolor}\n\n")
        f.write("\\begin{document}\n\n")
        f.write(table_latex)
        f.write("\n\\end{document}\n")
    
    print(f"✓ Detailed metrics table generated: {args.output_latex}")


if __name__ == "__main__":
    main()

