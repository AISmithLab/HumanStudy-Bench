"""
Shared pytest fixtures for extraction and metrics unit tests.
All in-memory; no data/ or results/ reads.
"""

import pytest


@pytest.fixture
def test_gt_ttest():
    """Minimal ground-truth test dict for t-test (human extraction / add_statistical_replication_fields)."""
    return {
        "test_name": "Example t-test",
        "reported_statistics": "t(79) = 2.66, p < .05",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }


@pytest.fixture
def test_gt_correlation():
    """Minimal ground-truth test dict for correlation."""
    return {
        "test_name": "Example correlation",
        "reported_statistics": "r(103) = 0.42, p < .001",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }


@pytest.fixture
def test_gt_exact_p():
    """Ground truth with exact p-value."""
    return {
        "test_name": "Exact p",
        "reported_statistics": "p = 0.03",
        "significance_level": 0.05,
        "expected_direction": "positive",
    }


@pytest.fixture
def test_result_minimal():
    """Minimal test_result for add_statistical_replication_fields (t-test)."""
    return {
        "finding_id": "F1",
        "test_name": "Example t-test",
        "statistical_test_type": "t-test",
        "pi_human_source": "t=2.66, n=80",
        "agent_reason": "t=2.5, n1=10, n2=10",
        "pas": 0.75,
    }


@pytest.fixture
def test_result_correlation():
    """Minimal test_result for correlation."""
    return {
        "finding_id": "F1",
        "test_name": "Example correlation",
        "statistical_test_type": "correlation",
        "pi_human_source": "r=0.42, n=103",
        "agent_reason": "r=0.55, n=100",
        "pas": 0.8,
    }


@pytest.fixture
def trial_info_002():
    """Minimal trial_info for study_002 get_required_q_numbers."""
    return {
        "items": [
            {"q_idx_estimate": "Q1.1", "q_idx_choice": None},
            {"q_idx_estimate": "Q1.2", "q_idx_choice": None},
            {"q_idx_estimate": "Q2.1", "q_idx_choice": None},
        ],
    }


@pytest.fixture
def trial_info_009():
    """Minimal trial_info for study_009 get_required_q_numbers (round_feedbacks)."""
    return {"round_feedbacks": [None, None, None, None]}  # 4 rounds -> Q1..Q4


@pytest.fixture
def trial_info_009_default():
    """trial_info with no round_feedbacks -> default Q1–Q4."""
    return {}


@pytest.fixture
def response_text_002():
    """Sample response_text for study_002 parse_agent_responses."""
    return "Q1.1=2340 Q1.2=7 Q2.1=29029"


@pytest.fixture
def response_text_002_colon():
    """Study 002 style with colon separator."""
    return "Q1: 36, Q2: 75"


@pytest.fixture
def response_text_006():
    """Sample response_text for study_006 (choice A/B)."""
    # Use commas or newlines to separate Q entries (matches study_006 pattern better)
    return "Q1=A, Q2=B, Q3=A"


@pytest.fixture
def response_text_009():
    """Sample response_text for study_009 (numeric Q1–Q4)."""
    return "Q1=50 Q2=25 Q3=30 Q4=40"


@pytest.fixture
def response_text_empty():
    """Empty or no-Q response."""
    return "Some prose with no Qk=value."


@pytest.fixture
def pas_se_lists():
    """For aggregate_pas_inverse_variance tests."""
    return {
        "pas": [0.6, 0.7, 0.8],
        "se": [0.1, 0.1, 0.1],
    }
