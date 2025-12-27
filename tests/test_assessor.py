"""Tests for assessor scoring logic.

These tests verify INVARIANTS and REQUIREMENTS, not implementation details.
Each test should answer: "What bug would this catch?"
"""

import pytest
from typing import cast

from assessor import (
    compute_score,
    CRITERIA,
    CATEGORY_WEIGHTS,
    PASS_THRESHOLD,
    CRITERIA_NA_INVALID,
    CriterionAnswer,
    get_applicable_criteria,
)

# =============================================================================
# Test Helpers
# =============================================================================

# Valid answers - runtime check catches typos that cast() would hide
VALID_ANSWERS = frozenset({"YES", "NO", "NA", "ERROR"})

type TestResult = tuple[str, CriterionAnswer, str]


def make_results(data: list[tuple[str, str, str]]) -> list[TestResult]:
    """Convert test data to properly typed results with validation."""
    for cid, ans, _ in data:
        if ans not in VALID_ANSWERS:
            raise ValueError(f"Invalid answer '{ans}' for criterion {cid}")
    return [(cid, cast(CriterionAnswer, ans), reason) for cid, ans, reason in data]


def all_yes() -> list[TestResult]:
    """All criteria answered YES."""
    return make_results([(c.id, "YES", "") for c in CRITERIA])


def all_no() -> list[TestResult]:
    """All criteria answered NO."""
    return make_results([(c.id, "NO", "") for c in CRITERIA])


def all_yes_except(overrides: dict[str, str]) -> list[TestResult]:
    """All YES except specified criteria."""
    return make_results([(c.id, overrides.get(c.id, "YES"), "") for c in CRITERIA])


# =============================================================================
# Invariant Tests: Properties That Must ALWAYS Hold
# =============================================================================


class TestScoreInvariants:
    """Properties that must hold for ANY valid input."""

    def test_score_is_bounded_zero_to_one(self) -> None:
        """Score must always be in [0, 1] regardless of inputs."""
        # Test extremes
        for results in [all_yes(), all_no()]:
            result = compute_score(results, list(CRITERIA))
            assert 0.0 <= result.score <= 1.0

    def test_score_equals_weighted_sum_of_categories(self) -> None:
        """Score must equal weighted sum of category scores (the formula)."""
        result = compute_score(all_yes(), list(CRITERIA))

        expected = sum(
            result.category_scores[cat] * weight
            for cat, weight in CATEGORY_WEIGHTS.items()
        )
        assert result.score == pytest.approx(expected, rel=1e-9)

    def test_weights_sum_to_one(self) -> None:
        """Category weights must sum to 1.0 (or scoring breaks)."""
        assert sum(CATEGORY_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_yes_gives_maximum_score(self) -> None:
        """All YES should produce score of 1.0 (maximum possible)."""
        result = compute_score(all_yes(), list(CRITERIA))
        assert result.score == 1.0

    def test_all_no_gives_minimum_score(self) -> None:
        """All NO should produce score of 0.0 (minimum possible)."""
        result = compute_score(all_no(), list(CRITERIA))
        assert result.score == 0.0


class TestPassFailInvariants:
    """Invariants about pass/fail decisions."""

    def test_safety_gate_failure_implies_not_passed(self) -> None:
        """If safety_gate_failed is True, passed must be False."""
        # CQ8 NO triggers safety gate
        results = all_yes_except({"CQ8": "NO"})
        result = compute_score(results, list(CRITERIA))

        assert result.safety_gate_failed is True
        assert result.passed is False  # This is the invariant

    def test_passed_requires_score_at_or_above_threshold(self) -> None:
        """passed=True requires score >= PASS_THRESHOLD."""
        result = compute_score(all_yes(), list(CRITERIA))

        if result.passed:
            assert result.score >= PASS_THRESHOLD

    def test_threshold_is_documented_value(self) -> None:
        """PASS_THRESHOLD should be 0.80 as documented."""
        assert PASS_THRESHOLD == 0.80


# =============================================================================
# Safety Gate Tests: Business-Critical Requirements
# =============================================================================


class TestSafetyGate:
    """Safety gate is a critical business requirement: harmful content must never pass."""

    def test_cq8_no_triggers_safety_gate(self) -> None:
        """CQ8 (harmful patterns) = NO must trigger safety gate."""
        results = all_yes_except({"CQ8": "NO"})
        result = compute_score(results, list(CRITERIA))

        assert result.safety_gate_failed is True
        assert "CQ8" in result.failed_safety

    def test_cq9_no_triggers_safety_gate(self) -> None:
        """CQ9 (crisis handling) = NO must trigger safety gate."""
        results = all_yes_except({"CQ9": "NO"})
        result = compute_score(results, list(CRITERIA))

        assert result.safety_gate_failed is True
        assert "CQ9" in result.failed_safety

    def test_cq8_error_triggers_safety_gate(self) -> None:
        """CQ8 ERROR must trigger safety gate (fail-safe: can't verify = reject)."""
        results = all_yes_except({"CQ8": "ERROR"})
        result = compute_score(results, list(CRITERIA))

        assert result.safety_gate_failed is True

    def test_non_safety_failure_does_not_trigger_gate(self) -> None:
        """Failures outside CQ8/CQ9 should not trigger safety gate."""
        results = all_yes_except({"CQ1": "NO"})  # CQ1 is comprehension, not safety
        result = compute_score(results, list(CRITERIA))

        assert result.safety_gate_failed is False


# =============================================================================
# NA-Invalid Tests: Specific Business Rule
# =============================================================================


class TestNAInvalid:
    """Some criteria must NEVER return NA (they always apply)."""

    def test_na_invalid_criteria_are_documented(self) -> None:
        """CRITERIA_NA_INVALID should match the assessor's business rule."""
        # These criteria should always be assessable; NA would typically indicate judge confusion.
        # Treating NA as failure prevents "NA" from being used to dodge core checks.
        # New 15-criteria structure: CQ1 (understanding), CQ8 (safety), CP2 (naturalness),
        # MT1 (topic coverage), MT6 (segmentation clarity)
        assert CRITERIA_NA_INVALID == frozenset({"CQ1", "CQ8", "CP2", "MT1", "MT6"})

    def test_cq8_na_treated_as_failure(self) -> None:
        """CQ8 = NA is invalid and must be treated as failure."""
        results = all_yes_except({"CQ8": "NA"})
        result = compute_score(results, list(CRITERIA))

        assert "CQ8" in result.failed_checks
        assert result.safety_gate_failed is True  # CQ8 is safety criterion

    def test_cp2_na_treated_as_failure(self) -> None:
        """CP2 = NA is invalid and must be treated as failure."""
        results = all_yes_except({"CP2": "NA"})
        result = compute_score(results, list(CRITERIA))

        assert "CP2" in result.failed_checks
        # CP2 is patterns, not safety - so no safety gate
        assert result.safety_gate_failed is False

    def test_cq9_na_is_valid(self) -> None:
        """CQ9 = NA is VALID (no crisis signals present = OK)."""
        results = all_yes_except({"CQ9": "NA"})
        result = compute_score(results, list(CRITERIA))

        assert "CQ9" not in result.failed_checks
        assert result.safety_gate_failed is False
        assert result.score == 1.0  # NA counts as pass


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """ERROR responses represent API failures - must be treated as failures."""

    def test_error_counts_as_failure(self) -> None:
        """ERROR should count as 0.0, not as pass."""
        results = all_yes_except({"CQ1": "ERROR"})
        result = compute_score(results, list(CRITERIA))

        assert "CQ1" in result.failed_checks
        assert result.error_count == 1

    def test_error_reduces_category_score(self) -> None:
        """ERROR in a category should reduce that category's score."""
        results = all_yes_except({"CQ1": "ERROR"})
        result = compute_score(results, list(CRITERIA))

        # CQ1 + CQ2 = comprehension. CQ1=ERROR(0) + CQ2=YES(1) = 0.5
        assert result.category_scores["comprehension"] == 0.5


# =============================================================================
# Conditional Criteria Tests
# =============================================================================


class TestConditionalCriteria:
    """Some criteria only apply for longer conversations."""

    def test_mt4_requires_3_turns(self) -> None:
        """MT4 (history utilization) requires 3+ turns."""
        short = get_applicable_criteria(2)
        long = get_applicable_criteria(3)

        assert "MT4" not in {c.id for c in short}
        assert "MT4" in {c.id for c in long}

    def test_context_use_criteria_require_multiple_turns(self) -> None:
        """context_use criteria (MT4, MT5) require longer conversations."""
        # 1-turn conversation: neither MT4 nor MT5 apply
        one_turn = get_applicable_criteria(1)
        one_turn_ids = {c.id for c in one_turn}

        # MT4 requires min_turns=3, MT5 requires min_turns=2
        assert "MT4" not in one_turn_ids
        assert "MT5" not in one_turn_ids

        # 2-turn conversation: MT5 applies, MT4 doesn't
        two_turns = get_applicable_criteria(2)
        two_turn_ids = {c.id for c in two_turns}

        assert "MT4" not in two_turn_ids
        assert "MT5" in two_turn_ids

    def test_missing_criteria_default_to_pass(self) -> None:
        """If a category has no applicable criteria, it defaults to 1.0."""
        # 1-turn conversation: neither MT4 nor MT5 apply
        criteria = get_applicable_criteria(1)
        results = make_results([(c.id, "YES", "") for c in criteria])

        result = compute_score(results, criteria)

        # context_use category has no applicable criteria -> defaults to 1.0
        assert result.category_scores["context_use"] == 1.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_empty_results_for_category_defaults_to_pass(self) -> None:
        """If no criteria assessed for a category, it should default to 1.0."""
        # Only assess safety criteria
        results = make_results(
            [
                ("CQ8", "YES", ""),
                ("CQ9", "YES", ""),
            ]
        )
        # But pass full criteria list (other categories will be empty)
        result = compute_score(results, list(CRITERIA))

        # Categories without assessed criteria default to 1.0
        assert result.category_scores["comprehension"] == 1.0
        assert result.category_scores["connection"] == 1.0

    def test_partial_category_failure(self) -> None:
        """Category scoring works when only one criterion in a category is failed."""
        # context_use category includes MT4, MT5, and MT7.
        # If one is NO and the others are YES, the category score should be 2/3.
        context_ids = {c.id for c in CRITERIA if c.category == "context_use"}
        assert context_ids == {"MT4", "MT5", "MT7"}, (
            "context_use should have MT4, MT5, MT7"
        )

        # Fail MT4 explicitly (not by index)
        results = all_yes_except({"MT4": "NO"})
        result = compute_score(results, list(CRITERIA))

        # One of 3 criteria failed = 2/3
        assert result.category_scores["context_use"] == pytest.approx(2 / 3)
