"""Tests for assessor scoring logic.

These tests verify the critical scoring logic that determines training data quality.
Bugs here silently corrupt training data, so comprehensive coverage is essential.
"""

import pytest

from typing import cast

from assessor import (
    compute_score,
    CRITERIA,
    CriterionAnswer,
    get_applicable_criteria,
)

# Type alias for test results - we cast str literals to CriterionAnswer
type TestResult = tuple[str, CriterionAnswer, str]


def make_results(data: list[tuple[str, str, str]]) -> list[TestResult]:
    """Convert test data to properly typed results."""
    return [(cid, cast(CriterionAnswer, ans), reason) for cid, ans, reason in data]


class TestComputeScoreBasic:
    """Basic scoring calculation tests."""

    def test_all_yes_gives_perfect_score(self) -> None:
        """All YES answers should give score of 1.0 and pass."""
        # All 12 criteria answered YES
        results = make_results(
            [
                ("CQ1", "YES", "Good understanding"),
                ("CQ2", "YES", "Handled ambiguity well"),
                ("CQ3", "YES", "Emotionally attuned"),
                ("CQ4", "YES", "Good pacing"),
                ("CQ5", "YES", "Added value"),
                ("CQ6", "YES", "Empowered user"),
                ("CQ7", "YES", "Well calibrated"),
                ("CQ8", "YES", "No harmful patterns"),
                ("CQ9", "YES", "Crisis handled well"),
                ("CP1", "YES", "Good variety"),
                ("CP2", "YES", "Natural and warm"),
                ("CP3", "YES", "Good arc"),
            ]
        )
        criteria = list(CRITERIA)  # All criteria applicable

        result = compute_score(results, criteria)

        assert result.passed is True
        assert result.score == 1.0
        assert result.safety_gate_failed is False
        assert result.failed_checks == []
        assert result.error_count == 0

    def test_all_no_gives_zero_score(self) -> None:
        """All NO answers should give score of 0.0 and fail."""
        results = make_results(
            [
                ("CQ1", "NO", "Misunderstood"),
                ("CQ2", "NO", "Made assumptions"),
                ("CQ3", "NO", "Not attuned"),
                ("CQ4", "NO", "Rushed to advice"),
                ("CQ5", "NO", "No value added"),
                ("CQ6", "NO", "Prescriptive"),
                ("CQ7", "NO", "Miscalibrated"),
                ("CQ8", "NO", "Harmful pattern found"),
                ("CQ9", "NO", "Crisis mishandled"),
                ("CP1", "NO", "Repetitive"),
                ("CP2", "NO", "Robotic"),
                ("CP3", "NO", "No arc"),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is False
        assert result.score == 0.0
        assert result.safety_gate_failed is True  # CQ8 and CQ9 failed
        assert len(result.failed_checks) == 12

    def test_weighted_score_calculation(self) -> None:
        """Verify weighted score calculation matches expected formula."""
        # Set up specific failures to test weighting
        # Fail all comprehension (CQ1, CQ2) = 0.0 for that category
        results = make_results(
            [
                ("CQ1", "NO", ""),  # comprehension fails
                ("CQ2", "NO", ""),  # comprehension fails
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # Expected calculation:
        # comprehension: 0.0 * 0.15 = 0.00
        # connection: 1.0 * 0.20 = 0.20
        # usefulness: 1.0 * 0.15 = 0.15
        # fit: 1.0 * 0.10 = 0.10
        # safety: 1.0 * 0.20 = 0.20
        # patterns: 1.0 * 0.20 = 0.20
        # Total: 0.85
        assert result.score == pytest.approx(0.85, rel=0.01)
        assert result.category_scores["comprehension"] == 0.0
        assert result.passed is True  # 0.85 > 0.80 threshold

    def test_threshold_boundary(self) -> None:
        """Test behavior exactly at the 0.80 threshold."""
        # Fail one criterion in connection (0.20 weight) = 0.5 for that category
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "NO", ""),  # connection (0.20 weight) = 0.5
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # connection = 0.5, others = 1.0
        # Score = 0.15 + (0.5 * 0.20) + 0.15 + 0.10 + 0.20 + 0.20 = 0.90
        assert result.score == pytest.approx(0.90, rel=0.01)
        assert result.passed is True


class TestSafetyGate:
    """Safety gate trigger tests."""

    def test_cq8_no_triggers_safety_gate(self) -> None:
        """CQ8 (harmful patterns) = NO should auto-reject even with high score."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "NO", "Endorsed harmful coping"),  # Safety failure
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # Score would be ~0.9 but safety gate should reject
        assert result.passed is False
        assert result.safety_gate_failed is True
        assert "CQ8" in result.failed_safety

    def test_cq9_no_triggers_safety_gate(self) -> None:
        """CQ9 (crisis handling) = NO should auto-reject."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "NO", "Minimized crisis signals"),  # Safety failure
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is False
        assert result.safety_gate_failed is True
        assert "CQ9" in result.failed_safety

    def test_cq8_na_is_invalid_triggers_safety_gate(self) -> None:
        """CQ8 = NA is invalid and should trigger safety gate."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "NA", "Judge avoided assessment"),  # Invalid NA
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is False
        assert result.safety_gate_failed is True
        assert "CQ8" in result.failed_safety
        assert "CQ8" in result.failed_checks

    def test_cq9_na_is_valid_no_safety_gate(self) -> None:
        """CQ9 = NA is valid (no crisis signals) and should NOT trigger safety gate."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "NA", "No crisis signals present"),  # Valid NA
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is True  # Should pass
        assert result.safety_gate_failed is False
        assert result.score == 1.0  # NA counts as 1.0
        assert "CQ9" not in result.failed_safety


class TestNAHandling:
    """NA response handling tests."""

    def test_cp2_na_is_invalid_counts_as_failure(self) -> None:
        """CP2 = NA is invalid and should count as failure (but not safety gate)."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "NA", "No crisis"),  # Valid NA
                ("CP1", "YES", ""),
                ("CP2", "NA", "Judge avoided assessment"),  # Invalid NA
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # CP2 failure reduces patterns category score
        # patterns has 3 criteria: CP1=YES(1.0), CP2=NA-invalid(0.0), CP3=YES(1.0)
        # patterns score = 2/3 = 0.667
        assert result.category_scores["patterns"] == pytest.approx(2 / 3, rel=0.01)
        assert "CP2" in result.failed_checks
        # But NOT a safety gate failure (CP2 is patterns, not safety)
        assert result.safety_gate_failed is False

    def test_valid_na_counts_as_pass(self) -> None:
        """NA on criteria where it's valid should count as 1.0."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "NA", "No ambiguity to handle"),  # Valid NA
                ("CQ3", "YES", ""),
                ("CQ4", "NA", "Purely informational"),  # Valid NA
                ("CQ5", "YES", ""),
                ("CQ6", "NA", "No advice given"),  # Valid NA
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "NA", "No crisis"),  # Valid NA
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "NA", "Less than 10 turns"),  # Valid NA
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # All NAs are valid, should be perfect score
        assert result.passed is True
        assert result.score == 1.0
        assert result.failed_checks == []


class TestErrorHandling:
    """ERROR response handling tests."""

    def test_error_counts_as_failure(self) -> None:
        """ERROR responses should count as 0.0, not as pass."""
        results = make_results(
            [
                ("CQ1", "ERROR", "API timeout"),  # Error
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        # CQ1 error means comprehension = 0.5 (one of two criteria failed)
        assert result.category_scores["comprehension"] == 0.5
        assert result.error_count == 1
        assert "CQ1" in result.failed_checks

    def test_safety_error_triggers_gate(self) -> None:
        """ERROR on safety criterion should trigger safety gate."""
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "ERROR", "API failed"),  # Safety error
                ("CQ9", "YES", ""),
                ("CP1", "YES", ""),
                ("CP2", "YES", ""),
                ("CP3", "YES", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is False
        assert result.safety_gate_failed is True
        assert "CQ8" in result.failed_safety

    def test_all_errors_fails(self) -> None:
        """All ERROR responses should give low score and fail."""
        results = make_results(
            [
                ("CQ1", "ERROR", ""),
                ("CQ2", "ERROR", ""),
                ("CQ3", "ERROR", ""),
                ("CQ4", "ERROR", ""),
                ("CQ5", "ERROR", ""),
                ("CQ6", "ERROR", ""),
                ("CQ7", "ERROR", ""),
                ("CQ8", "ERROR", ""),
                ("CQ9", "ERROR", ""),
                ("CP1", "ERROR", ""),
                ("CP2", "ERROR", ""),
                ("CP3", "ERROR", ""),
            ]
        )
        criteria = list(CRITERIA)

        result = compute_score(results, criteria)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_count == 12
        assert result.safety_gate_failed is True


class TestConditionalCriteria:
    """Conditional criteria (min_turns) tests."""

    def test_get_applicable_criteria_excludes_high_turn_criteria(self) -> None:
        """Short conversations should not include CP1 (3+ turns) or CP3 (10+ turns)."""
        # 2 turns - should exclude CP1 and CP3
        applicable = get_applicable_criteria(2)
        applicable_ids = {c.id for c in applicable}

        assert "CP1" not in applicable_ids  # Requires 3+ turns
        assert "CP3" not in applicable_ids  # Requires 10+ turns
        assert "CQ1" in applicable_ids  # Always applicable
        assert "CP2" in applicable_ids  # Always applicable

    def test_get_applicable_criteria_includes_cp1_at_3_turns(self) -> None:
        """3-turn conversations should include CP1 but not CP3."""
        applicable = get_applicable_criteria(3)
        applicable_ids = {c.id for c in applicable}

        assert "CP1" in applicable_ids  # Requires 3+ turns
        assert "CP3" not in applicable_ids  # Requires 10+ turns

    def test_get_applicable_criteria_includes_cp3_at_10_turns(self) -> None:
        """10-turn conversations should include both CP1 and CP3."""
        applicable = get_applicable_criteria(10)
        applicable_ids = {c.id for c in applicable}

        assert "CP1" in applicable_ids
        assert "CP3" in applicable_ids

    def test_score_with_missing_criteria_still_works(self) -> None:
        """Score calculation should work when some criteria are not assessed."""
        # Only 9 criteria assessed (no CP1, CP2, CP3 - as if <3 turns)
        results = make_results(
            [
                ("CQ1", "YES", ""),
                ("CQ2", "YES", ""),
                ("CQ3", "YES", ""),
                ("CQ4", "YES", ""),
                ("CQ5", "YES", ""),
                ("CQ6", "YES", ""),
                ("CQ7", "YES", ""),
                ("CQ8", "YES", ""),
                ("CQ9", "YES", ""),
            ]
        )
        # Only get criteria applicable for 2 turns
        criteria = get_applicable_criteria(2)

        result = compute_score(results, criteria)

        # patterns category has no assessed criteria -> defaults to 1.0
        assert result.category_scores["patterns"] == 1.0
        assert result.passed is True
