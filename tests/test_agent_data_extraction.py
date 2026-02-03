"""
Unit tests for agent data extraction: parse_agent_responses, get_required_q_numbers,
and agent p-value/effect-size extraction via scipy + add_statistical_replication_fields.
"""

import importlib.util
import sys
from pathlib import Path
import pytest
import numpy as np
from scipy import stats

from src.evaluation.stats_lib import add_statistical_replication_fields


# -----------------------------------------------------------------------------
# Helper: Load evaluator module
# -----------------------------------------------------------------------------


def _load_evaluator(study_id: str):
    """Load evaluator module dynamically (same pattern as evaluator_runner)."""
    evaluator_path = Path(f"src/studies/{study_id}_evaluator.py")
    if not evaluator_path.exists():
        pytest.skip(f"Evaluator not found: {evaluator_path}")
    
    spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{study_id}_evaluator"] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# parse_agent_responses
# -----------------------------------------------------------------------------


def test_parse_agent_responses_study_002(response_text_002):
    """Test study_002 parse_agent_responses with Q1.1=value format."""
    module = _load_evaluator("study_002")
    parsed = module.parse_agent_responses(response_text_002)
    assert "Q1.1" in parsed
    assert parsed["Q1.1"] == "2340"
    assert parsed["Q1.2"] == "7"
    assert parsed["Q2.1"] == "29029"


def test_parse_agent_responses_study_002_colon(response_text_002_colon):
    """Test study_002 with colon separator."""
    module = _load_evaluator("study_002")
    parsed = module.parse_agent_responses(response_text_002_colon)
    assert "Q1" in parsed
    assert parsed["Q1"] == "36"
    assert parsed["Q2"] == "75"


def test_parse_agent_responses_study_006(response_text_006):
    """Test study_006 parse_agent_responses with choice format (A/B)."""
    module = _load_evaluator("study_006")
    parsed = module.parse_agent_responses(response_text_006)
    assert "Q1" in parsed
    assert parsed["Q1"] == "A"
    assert parsed["Q2"] == "B"
    assert parsed["Q3"] == "A"


def test_parse_agent_responses_study_009(response_text_009):
    """Test study_009 parse_agent_responses with numeric Q1-Q4."""
    module = _load_evaluator("study_009")
    parsed = module.parse_agent_responses(response_text_009)
    assert "Q1" in parsed
    # study_009 returns Dict[str, float], not Dict[str, str]
    assert parsed["Q1"] == pytest.approx(50.0)
    assert parsed["Q2"] == pytest.approx(25.0)
    assert parsed["Q3"] == pytest.approx(30.0)
    assert parsed["Q4"] == pytest.approx(40.0)


def test_parse_agent_responses_empty(response_text_empty):
    """Test parse_agent_responses with empty/no-Q response."""
    module = _load_evaluator("study_002")
    parsed = module.parse_agent_responses(response_text_empty)
    assert parsed == {}


def test_parse_agent_responses_malformed():
    """Test parse_agent_responses skips malformed lines."""
    module = _load_evaluator("study_002")
    text = "Q1=5 Some prose Q2=10 Q3 malformed Q4: 20"
    parsed = module.parse_agent_responses(text)
    assert "Q1" in parsed
    assert "Q2" in parsed
    assert "Q4" in parsed
    # Q3 malformed should be skipped or handled gracefully
    assert "Q3" not in parsed or parsed.get("Q3") is not None


# -----------------------------------------------------------------------------
# get_required_q_numbers
# -----------------------------------------------------------------------------


def test_get_required_q_numbers_study_002(trial_info_002):
    """Test study_002 get_required_q_numbers from items."""
    module = _load_evaluator("study_002")
    required = module.get_required_q_numbers(trial_info_002)
    assert "Q1.1" in required
    assert "Q1.2" in required
    assert "Q2.1" in required


def test_get_required_q_numbers_study_009(trial_info_009):
    """Test study_009 get_required_q_numbers from round_feedbacks."""
    module = _load_evaluator("study_009")
    required = module.get_required_q_numbers(trial_info_009)
    assert "Q1" in required
    assert "Q2" in required
    assert "Q3" in required
    assert "Q4" in required


def test_get_required_q_numbers_study_009_default(trial_info_009_default):
    """Test study_009 get_required_q_numbers default (no round_feedbacks -> Q1-Q4)."""
    module = _load_evaluator("study_009")
    required = module.get_required_q_numbers(trial_info_009_default)
    assert "Q1" in required
    assert "Q2" in required
    assert "Q3" in required
    assert "Q4" in required


# -----------------------------------------------------------------------------
# Agent p-value and effect-size (via scipy + add_statistical_replication_fields)
# -----------------------------------------------------------------------------


def test_agent_p_value_ttest_extraction():
    """Test agent p-value extraction from scipy ttest_ind + add_statistical_replication_fields."""
    # Generate two groups
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 10)
    group2 = np.random.normal(0.8, 1, 10)
    
    # Run scipy t-test
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    # Build minimal test_gt and test_result
    test_gt = {
        "reported_statistics": "t(18) = 2.0, p < .05",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }
    
    test_result = {
        "statistical_test_type": "t-test",
        "agent_reason": f"t={t_stat:.2f}, n1={len(group1)}, n2={len(group2)}",
        "pi_human_source": "t=2.0, n=20",
    }
    
    # Call add_statistical_replication_fields
    add_statistical_replication_fields(
        test_result,
        test_gt,
        p_val_agent=p_val,
        test_stat_agent=t_stat,
        test_type="t-test",
        n_agent=len(group1),
        n2_agent=len(group2),
        n_human=10,
        n2_human=10,
        independent=True,
    )
    
    # Assertions
    assert "p_value_agent" in test_result
    assert test_result["p_value_agent"] == pytest.approx(p_val)
    assert "is_significant_agent" in test_result
    assert isinstance(test_result["is_significant_agent"], bool)
    assert "agent_effect_size" in test_result
    assert test_result["agent_effect_size"] is not None
    assert "agent_effect_d" in test_result
    assert test_result["agent_effect_d"] is not None


def test_agent_effect_size_ttest():
    """Test agent effect size (Cohen's d) is correctly calculated from t-stat."""
    np.random.seed(43)
    group1 = np.random.normal(0, 1, 15)
    group2 = np.random.normal(1.0, 1, 15)
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    test_gt = {
        "reported_statistics": "t(28) = 2.5, p < .05",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }
    
    test_result = {
        "statistical_test_type": "t-test",
        "agent_reason": f"t={t_stat:.2f}, n1={len(group1)}, n2={len(group2)}",
    }
    
    add_statistical_replication_fields(
        test_result,
        test_gt,
        p_val_agent=p_val,
        test_stat_agent=t_stat,
        test_type="t-test",
        n_agent=len(group1),
        n2_agent=len(group2),
        n_human=15,
        n2_human=15,
        independent=True,
    )
    
    # Check effect size is Cohen's d
    from src.evaluation.stats_lib import FrequentistConsistency
    
    expected_d = FrequentistConsistency.t_to_cohens_d(t_stat, len(group1), len(group2), independent=True)
    assert test_result["agent_effect_size"] == pytest.approx(expected_d)
    assert test_result["agent_effect_d"] == pytest.approx(expected_d)


def test_agent_p_value_correlation():
    """Test agent p-value extraction from scipy pearsonr + add_statistical_replication_fields."""
    np.random.seed(44)
    x = np.random.normal(0, 1, 50)
    y = 0.5 * x + np.random.normal(0, 0.5, 50)
    
    r_stat, p_val = stats.pearsonr(x, y)
    
    test_gt = {
        "reported_statistics": "r(48) = 0.5, p < .001",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }
    
    test_result = {
        "statistical_test_type": "correlation",
        "agent_reason": f"r={r_stat:.2f}, n={len(x)}",
    }
    
    add_statistical_replication_fields(
        test_result,
        test_gt,
        p_val_agent=p_val,
        test_stat_agent=r_stat,
        test_type="correlation",
        n_agent=len(x),
        n_human=50,
    )
    
    assert "p_value_agent" in test_result
    assert test_result["p_value_agent"] == pytest.approx(p_val)
    assert "agent_effect_size" in test_result
    # For correlation, effect_size is Fisher's z
    from src.evaluation.stats_lib import FrequentistConsistency
    expected_z = FrequentistConsistency.correlation_to_fisher_z(r_stat)
    assert test_result["agent_effect_size"] == pytest.approx(expected_z)


def test_agent_direction_and_significance():
    """Test agent direction and significance are correctly set."""
    np.random.seed(45)
    group1 = np.random.normal(0, 1, 20)
    group2 = np.random.normal(1.5, 1, 20)  # Clear positive effect
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    test_gt = {
        "reported_statistics": "t(38) = 3.0, p < .01",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }
    
    test_result = {
        "statistical_test_type": "t-test",
        "agent_reason": f"t={t_stat:.2f}, n1={len(group1)}, n2={len(group2)}",
    }
    
    add_statistical_replication_fields(
        test_result,
        test_gt,
        p_val_agent=p_val,
        test_stat_agent=t_stat,
        test_type="t-test",
        n_agent=len(group1),
        n2_agent=len(group2),
        n_human=20,
        n2_human=20,
        independent=True,
    )
    
    assert "agent_direction" in test_result
    assert test_result["agent_direction"] in [-1, 0, 1]
    assert "is_significant_agent" in test_result
    # With large effect, should be significant
    if p_val < 0.05:
        assert test_result["is_significant_agent"] is True
