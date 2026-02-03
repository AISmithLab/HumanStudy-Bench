"""
Unit tests for human data extraction: p-value parsing, calculation, and effect-size extraction.
"""

import math
import pytest
import numpy as np
from scipy import stats

from src.evaluation.stats_lib import (
    parse_p_value_from_reported,
    calculate_human_p_value,
    add_statistical_replication_fields,
    FrequentistConsistency,
)


# -----------------------------------------------------------------------------
# parse_p_value_from_reported
# -----------------------------------------------------------------------------


def test_parse_p_value_exact_sig():
    p_val, is_sig, conf = parse_p_value_from_reported("p = 0.03", 0.05)
    assert p_val == pytest.approx(0.03)
    assert is_sig is True
    assert conf == "high"


def test_parse_p_value_exact_nonsig():
    p_val, is_sig, conf = parse_p_value_from_reported("p = 0.5", 0.05)
    assert p_val == pytest.approx(0.5)
    assert is_sig is False
    assert conf == "high"


def test_parse_p_value_exact_no_space():
    p_val, is_sig, conf = parse_p_value_from_reported("p=0.001", 0.05)
    assert p_val == pytest.approx(0.001)
    assert is_sig is True
    assert conf == "high"


def test_parse_p_value_inequality_less():
    p_val, is_sig, conf = parse_p_value_from_reported("p < .05", 0.05)
    assert p_val == pytest.approx(0.025)  # threshold/2
    assert is_sig is True
    assert conf == "medium"


def test_parse_p_value_inequality_less_001():
    p_val, is_sig, conf = parse_p_value_from_reported("p < 0.001", 0.05)
    assert p_val == pytest.approx(0.0005)
    assert is_sig is True
    assert conf == "medium"


def test_parse_p_value_inequality_greater():
    p_val, is_sig, conf = parse_p_value_from_reported("p > .05", 0.05)
    assert p_val == pytest.approx(0.05)
    assert is_sig is False
    assert conf == "medium"


def test_parse_p_value_from_t_stat_large():
    p_val, is_sig, conf = parse_p_value_from_reported("t(79) = 2.66", 0.05)
    assert p_val == pytest.approx(0.025)
    assert is_sig is True
    assert conf == "low"


def test_parse_p_value_from_t_stat_small():
    p_val, is_sig, conf = parse_p_value_from_reported("t(30) = 1.0", 0.05)
    assert p_val == pytest.approx(0.10)
    assert is_sig is False
    assert conf == "low"


def test_parse_p_value_from_f_stat_large():
    p_val, is_sig, conf = parse_p_value_from_reported("F(1, 78) = 17.7", 0.05)
    assert p_val == pytest.approx(0.025)
    assert is_sig is True
    assert conf == "low"


def test_parse_p_value_from_f_stat_small():
    p_val, is_sig, conf = parse_p_value_from_reported("F(1, 78) = 2.0", 0.05)
    assert p_val == pytest.approx(0.10)
    assert is_sig is False
    assert conf == "low"


def test_parse_p_value_from_r_large():
    p_val, is_sig, conf = parse_p_value_from_reported("r = 0.5", 0.05)
    assert p_val == pytest.approx(0.025)
    assert is_sig is True
    assert conf == "low"


def test_parse_p_value_from_r_small():
    p_val, is_sig, conf = parse_p_value_from_reported("r = 0.1", 0.05)
    assert p_val == pytest.approx(0.10)
    assert is_sig is False
    assert conf == "low"


def test_parse_p_value_empty():
    p_val, is_sig, conf = parse_p_value_from_reported("", 0.05)
    assert p_val is None
    assert is_sig is False
    assert conf == "low"


def test_parse_p_value_no_match():
    p_val, is_sig, conf = parse_p_value_from_reported("Some text", 0.05)
    assert p_val is None
    assert is_sig is False
    assert conf == "low"


def test_parse_p_value_significance_level_001():
    p_val, is_sig, conf = parse_p_value_from_reported("p = 0.02", 0.01)
    assert p_val == pytest.approx(0.02)
    assert is_sig is False  # 0.02 > 0.01
    assert conf == "high"


# -----------------------------------------------------------------------------
# calculate_human_p_value
# -----------------------------------------------------------------------------


def test_calculate_human_p_value_ttest_two_sample():
    # Compare with scipy
    group1 = np.random.RandomState(42).normal(0, 1, 20)
    group2 = np.random.RandomState(43).normal(0.5, 1, 20)
    t_stat, p_scipy = stats.ttest_ind(group1, group2)
    
    p_calc = calculate_human_p_value("t-test", t_stat, 20, 20, None, None, "two-sided")
    assert p_calc == pytest.approx(p_scipy, abs=1e-10)


def test_calculate_human_p_value_ttest_one_sample():
    data = np.random.RandomState(44).normal(0.3, 1, 20)
    t_stat, p_scipy = stats.ttest_1samp(data, 0.0)
    
    p_calc = calculate_human_p_value("t-test", t_stat, 20, None, None, None, "two-sided")
    assert p_calc == pytest.approx(p_scipy, abs=1e-10)


def test_calculate_human_p_value_ttest_greater():
    t_stat = 2.0
    n = 20
    p_calc = calculate_human_p_value("t-test", t_stat, n, None, None, None, "greater")
    p_expected = 1 - stats.t.cdf(t_stat, n - 1)
    assert p_calc == pytest.approx(p_expected)


def test_calculate_human_p_value_correlation():
    r = 0.3
    n = 50
    p_calc = calculate_human_p_value("correlation", r, n, None, None, None, "two-sided")
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_expected = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    assert p_calc == pytest.approx(p_expected)


def test_calculate_human_p_value_binomial():
    k, n, p0 = 12, 20, 0.5
    p_calc = calculate_human_p_value("binomial", None, n, None, k, p0, "two-sided")
    binom_result = stats.binomtest(k, n, p0, alternative="two-sided")
    assert p_calc == pytest.approx(binom_result.pvalue)


def test_calculate_human_p_value_ftest():
    F = 4.0
    df1, df2 = 1, 30
    p_calc = calculate_human_p_value("f-test", F, df1, df2, None, None, "two-sided")
    p_expected = 1 - stats.f.cdf(F, df1, df2)
    assert p_calc == pytest.approx(p_expected)


def test_calculate_human_p_value_invalid_missing_stat():
    assert calculate_human_p_value("t-test", None, 20, None, None, None, "two-sided") is None


def test_calculate_human_p_value_invalid_missing_n():
    assert calculate_human_p_value("t-test", 2.0, None, None, None, None, "two-sided") is None


# -----------------------------------------------------------------------------
# Effect-size extraction / add_statistical_replication_fields (human side)
# -----------------------------------------------------------------------------


def test_t_to_cohens_d_independent():
    t, n1, n2 = 2.0, 20, 20
    d = FrequentistConsistency.t_to_cohens_d(t, n1, n2, independent=True)
    expected = t * math.sqrt((n1 + n2) / (n1 * n2))
    assert d == pytest.approx(expected)


def test_t_to_cohens_d_paired():
    t, n = 2.0, 20
    d = FrequentistConsistency.t_to_cohens_d(t, n, None, independent=False)
    expected = t / math.sqrt(n)
    assert d == pytest.approx(expected)


def test_correlation_to_fisher_z():
    r = 0.5
    z = FrequentistConsistency.correlation_to_fisher_z(r)
    expected = 0.5 * math.log((1 + r) / (1 - r))
    assert z == pytest.approx(expected)


def test_log_odds_ratio():
    a, b, c, d = 10.0, 5.0, 3.0, 12.0
    log_or = FrequentistConsistency.log_odds_ratio(a, b, c, d)
    expected = math.log((a * d) / (b * c))
    assert log_or == pytest.approx(expected)


def test_log_odds_ratio_with_zeros():
    # Should apply Haldane correction (add 0.5 to all cells)
    a, b, c, d = 0.0, 5.0, 3.0, 12.0
    log_or = FrequentistConsistency.log_odds_ratio(a, b, c, d)
    # With correction: (0.5 * 12.5) / (5.5 * 3.5)
    assert not math.isnan(log_or) and not math.isinf(log_or)


def test_log_odds_ratio_se():
    a, b, c, d = 10.0, 5.0, 3.0, 12.0
    se = FrequentistConsistency.log_odds_ratio_se(a, b, c, d)
    expected = math.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
    assert se == pytest.approx(expected)


def test_cohens_d_se_independent():
    d, n1, n2 = 0.5, 20, 20
    se = FrequentistConsistency.cohens_d_se(d, n1, n2)
    n_total = n1 + n2
    expected = math.sqrt((n_total / (n1 * n2)) + (d**2 / (2 * n_total)))
    assert se == pytest.approx(expected)


def test_correlation_se():
    r, n = 0.3, 50
    se = FrequentistConsistency.correlation_se(r, n)
    expected = 1.0 / math.sqrt(n - 3)
    assert se == pytest.approx(expected)


def test_add_statistical_replication_fields_ttest(test_gt_ttest, test_result_minimal):
    """Test add_statistical_replication_fields extracts human p-value and effect size for t-test."""
    test_result = test_result_minimal.copy()
    test_result["human_test_statistic"] = "2.66"
    
    add_statistical_replication_fields(
        test_result,
        test_gt_ttest,
        p_val_agent=0.03,
        test_stat_agent=2.5,
        test_type="t-test",
        n_agent=10,
        n2_agent=10,
        n_human=40,
        n2_human=40,
        independent=True,
    )
    
    assert "p_value_human" in test_result
    assert test_result["p_value_human"] is not None
    assert "is_significant_human" in test_result
    assert "human_effect_size" in test_result
    assert test_result["human_effect_size"] is not None
    assert "human_effect_d" in test_result
    assert test_result["human_effect_d"] is not None


def test_add_statistical_replication_fields_correlation(test_gt_correlation, test_result_correlation):
    """Test add_statistical_replication_fields for correlation."""
    test_result = test_result_correlation.copy()
    test_result["human_test_statistic"] = "0.42"
    
    add_statistical_replication_fields(
        test_result,
        test_gt_correlation,
        p_val_agent=0.01,
        test_stat_agent=0.55,
        test_type="correlation",
        n_agent=100,
        n_human=103,
    )
    
    assert "p_value_human" in test_result
    assert "human_effect_size" in test_result
    # For correlation, effect_size is Fisher's z
    assert test_result["human_effect_size"] is not None


def test_add_statistical_replication_fields_extracts_n_from_reported(test_gt_ttest):
    """Test that n_human is extracted from reported_statistics if not provided."""
    test_result = {
        "statistical_test_type": "t-test",
        "pi_human_source": "t=2.66",
    }
    
    add_statistical_replication_fields(
        test_result,
        test_gt_ttest,
        p_val_agent=0.03,
        test_stat_agent=2.5,
        test_type="t-test",
        n_agent=10,
        n2_agent=10,
        independent=True,
    )
    
    # Should extract n_human from "t(79)" -> df=79, n≈80 for two-sample
    assert "n_human_extracted" in test_result or "n_human" in test_result


def test_add_statistical_replication_fields_direction_match(test_gt_ttest, test_result_minimal):
    """Test direction matching between human and agent."""
    test_result = test_result_minimal.copy()
    test_result["human_test_statistic"] = "2.66"
    
    add_statistical_replication_fields(
        test_result,
        test_gt_ttest,
        p_val_agent=0.03,
        test_stat_agent=2.5,
        test_type="t-test",
        n_agent=10,
        n2_agent=10,
        n_human=40,
        n2_human=40,
        independent=True,
    )
    
    assert "human_direction" in test_result
    assert "agent_direction" in test_result
    assert "direction_match" in test_result
    assert isinstance(test_result["direction_match"], bool)
