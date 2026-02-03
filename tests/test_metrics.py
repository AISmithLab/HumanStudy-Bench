"""
Unit tests for metric calculations: calc_pas, aggregate_*, FrequentistConsistency.
"""

import math
import pytest
import numpy as np
from scipy import stats

from src.evaluation.stats_lib import (
    calc_pas,
    aggregate_finding_pas_raw,
    aggregate_study_pas,
    aggregate_pas_inverse_variance,
    FrequentistConsistency,
)


# -----------------------------------------------------------------------------
# calc_pas
# -----------------------------------------------------------------------------


def test_calc_pas_scalar_half():
    assert calc_pas(0.5, 0.5) == pytest.approx(0.5)


def test_calc_pas_scalar_one_one():
    # calc_pas clamps inputs to (1e-6, 1-1e-6), so (1.0, 1.0) becomes slightly less
    result = calc_pas(1.0, 1.0)
    assert result == pytest.approx(1.0, abs=1e-5)  # Allow small tolerance due to clamping


def test_calc_pas_scalar_zero_one():
    # calc_pas clamps inputs to (1e-6, 1-1e-6), so (0.0, 1.0) becomes (1e-6, 1-1e-6)
    # Result is approximately 2e-6, not exactly 0.0
    result = calc_pas(0.0, 1.0)
    assert result == pytest.approx(0.0, abs=1e-5)  # Allow small tolerance due to clamping


def test_calc_pas_scalar_symmetric():
    assert calc_pas(0.8, 0.8) == pytest.approx(0.68)


def test_calc_pas_three_way_dict():
    pi_h = {"pi_plus": 0.8, "pi_minus": 0.1, "pi_zero": 0.1}
    pi_a = {"pi_plus": 0.7, "pi_minus": 0.2, "pi_zero": 0.1}
    # PAS = dot product
    expected = 0.8 * 0.7 + 0.1 * 0.2 + 0.1 * 0.1
    assert calc_pas(pi_h, pi_a) == pytest.approx(expected)


def test_calc_pas_invalid_fallback():
    assert calc_pas("x", "y") == pytest.approx(0.5)


def test_calc_pas_clamp_near_zero_one():
    # Inputs near 0/1 are clamped; output should be valid
    b = calc_pas(1e-8, 1.0 - 1e-8)
    assert 0 <= b <= 1


# -----------------------------------------------------------------------------
# aggregate_finding_pas_raw
# -----------------------------------------------------------------------------


def test_aggregate_finding_pas_raw_single():
    tests = [{"pas": 0.7}]
    assert aggregate_finding_pas_raw(tests) == pytest.approx(0.7)


def test_aggregate_finding_pas_raw_single_score():
    tests = [{"score": 0.7}]
    assert aggregate_finding_pas_raw(tests) == pytest.approx(0.7)


def test_aggregate_finding_pas_raw_multiple_same_bas():
    tests = [{"pas": 0.6}, {"pas": 0.6}, {"pas": 0.6}]
    assert aggregate_finding_pas_raw(tests) == pytest.approx(0.6)


def test_aggregate_finding_pas_raw_mixed():
    tests = [{"pas": 0.6}, {"pas": 0.7}, {"pas": 0.8}]
    out = aggregate_finding_pas_raw(tests)
    assert 0.6 <= out <= 0.8


def test_aggregate_finding_pas_raw_extremes_clamp():
    tests = [{"pas": 0.0}, {"pas": 0.5}, {"pas": 1.0}]
    out = aggregate_finding_pas_raw(tests)
    assert 0 <= out <= 1 and not (math.isnan(out) or math.isinf(out))


def test_aggregate_finding_pas_raw_empty():
    assert aggregate_finding_pas_raw([]) == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# aggregate_study_pas
# -----------------------------------------------------------------------------


def _test_result(pas: float, finding_id: str, pi_human: float = 0.5):
    return {
        "finding_id": finding_id,
        "pas": pas,
        "pi_human": pi_human,
        "statistical_test_type": "t-test",
    }


def test_aggregate_study_pas_single_finding():
    tests = [
        _test_result(0.6, "F1"),
        _test_result(0.7, "F1"),
        _test_result(0.8, "F1"),
    ]
    pas_raw, pas_norm, breakdown = aggregate_study_pas(tests)
    assert "F1" in breakdown
    assert breakdown["F1"]["pas_raw"] >= 0 and breakdown["F1"]["pas_raw"] <= 1
    assert 0 <= pas_norm <= 1
    assert 0 <= pas_raw <= 1


def test_aggregate_study_pas_multiple_findings():
    tests = [
        _test_result(0.6, "F1"),
        _test_result(0.8, "F1"),
        _test_result(0.5, "F2"),
        _test_result(0.7, "F2"),
    ]
    pas_raw, pas_norm, breakdown = aggregate_study_pas(tests)
    assert "F1" in breakdown and "F2" in breakdown
    assert 0 <= pas_raw <= 1 and 0 <= pas_norm <= 1


def test_aggregate_study_pas_empty():
    pas_raw, pas_norm, breakdown = aggregate_study_pas([])
    assert pas_raw == pytest.approx(0.5)
    assert pas_norm == pytest.approx(0.0)
    assert breakdown == {}


# -----------------------------------------------------------------------------
# aggregate_pas_inverse_variance
# -----------------------------------------------------------------------------


def test_aggregate_pas_inverse_variance_valid_se():
    pas = [0.6, 0.7, 0.8]
    se = [0.1, 0.1, 0.1]
    agg, agg_se = aggregate_pas_inverse_variance(pas, se)
    # Equal weights -> mean
    assert agg == pytest.approx(0.7)
    # SE = 1 / sqrt(sum(1/se_k^2)) = 1/sqrt(300) = 1/sqrt(3)/10
    assert agg_se == pytest.approx(1.0 / math.sqrt(300))


def test_aggregate_pas_inverse_variance_zero_se_fallback():
    pas = [0.6, 0.7, 0.8]
    se = [0.0, 0.0, 0.0]
    agg, agg_se = aggregate_pas_inverse_variance(pas, se)
    assert agg == pytest.approx(0.7)
    assert agg_se == pytest.approx(0.0)


def test_aggregate_pas_inverse_variance_length_mismatch():
    pas = [0.6, 0.7]
    se = [0.1]
    agg, agg_se = aggregate_pas_inverse_variance(pas, se)
    assert agg == pytest.approx(0.0)
    assert agg_se == pytest.approx(0.0)


def test_aggregate_pas_inverse_variance_skips_nan():
    pas = [0.6, float("nan"), 0.8]
    se = [0.1, 0.1, 0.1]
    agg, agg_se = aggregate_pas_inverse_variance(pas, se)
    # Only 0.6 and 0.8 used
    assert agg == pytest.approx(0.7)
    assert agg_se > 0


# -----------------------------------------------------------------------------
# FrequentistConsistency
# -----------------------------------------------------------------------------


def test_calculate_z_diff_same_effect():
    z_diff, consistency = FrequentistConsistency.calculate_z_diff(0.0, 0.1, 0.0, 0.1)
    assert z_diff == pytest.approx(0.0)
    assert consistency == pytest.approx(1.0)


def test_calculate_z_diff_different_effect():
    z_diff, consistency = FrequentistConsistency.calculate_z_diff(1.0, 0.1, 0.0, 0.1)
    assert z_diff != 0
    assert consistency < 1.0 and consistency >= 0


def test_calculate_z_diff_se_zero_equal_effects():
    z_diff, consistency = FrequentistConsistency.calculate_z_diff(0.0, 0.0, 0.0, 0.0)
    assert z_diff == pytest.approx(0.0)
    assert consistency == pytest.approx(1.0)


def test_calculate_z_diff_se_zero_unequal_effects():
    z_diff, consistency = FrequentistConsistency.calculate_z_diff(1.0, 0.0, 0.0, 0.0)
    assert math.isinf(z_diff) and z_diff > 0
    assert consistency == pytest.approx(0.0)


def test_t_to_cohens_d_independent():
    # d = t * sqrt((n1+n2)/(n1*n2))
    t, n1, n2 = 2.0, 20, 20
    expected = t * math.sqrt((n1 + n2) / (n1 * n2))
    out = FrequentistConsistency.t_to_cohens_d(t, n1, n2, independent=True)
    assert out == pytest.approx(expected)


def test_t_to_cohens_d_paired():
    # d = t / sqrt(n)
    t, n = 2.0, 20
    expected = t / math.sqrt(n)
    out = FrequentistConsistency.t_to_cohens_d(t, n, None, independent=False)
    assert out == pytest.approx(expected)


def test_correlation_to_fisher_z_zero():
    assert FrequentistConsistency.correlation_to_fisher_z(0.0) == pytest.approx(0.0)


def test_correlation_to_fisher_z_half():
    z = FrequentistConsistency.correlation_to_fisher_z(0.5)
    assert z == pytest.approx(0.5 * math.log(3.0))  # atanh(0.5)


def test_cohens_d_se_independent():
    d, n1, n2 = 0.5, 20, 20
    se = FrequentistConsistency.cohens_d_se(d, n1, n2)
    assert se > 0 and not math.isinf(se)


def test_correlation_se():
    r, n = 0.3, 50
    se = FrequentistConsistency.correlation_se(r, n)
    assert se == pytest.approx(1.0 / math.sqrt(n - 3))


def test_log_odds_ratio():
    # 2x2 table; with 0.5 correction if any zero
    a, b, c, d = 10.0, 5.0, 3.0, 12.0
    log_or = FrequentistConsistency.log_odds_ratio(a, b, c, d)
    expected = math.log((a * d) / (b * c))
    assert log_or == pytest.approx(expected)


def test_log_odds_ratio_se():
    a, b, c, d = 10.0, 5.0, 3.0, 12.0
    se = FrequentistConsistency.log_odds_ratio_se(a, b, c, d)
    assert se > 0
