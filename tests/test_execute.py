"""Tests for auto_llm_predictor.nodes.execute.

Covers: execute_prep_code guard and check_prep_result routing.
"""

from __future__ import annotations

import logging


# ---------------------------------------------------------------------------
# execute_prep_code — missing path guard
# ---------------------------------------------------------------------------

class TestExecutePrepCodeGuard:
    """Tests for the missing prep_code_path guard in execute_prep_code."""

    def test_missing_prep_code_path(self, tmp_path):
        """execute_prep_code should return failure, not KeyError, when
        prep_code_path is missing."""
        from auto_llm_predictor.nodes.execute import execute_prep_code

        state = {"output_dir": str(tmp_path)}
        result = execute_prep_code(state)
        assert result["prep_succeeded"] is False
        assert "No prep script path" in result["prep_result"]

    def test_empty_prep_code_path(self, tmp_path):
        from auto_llm_predictor.nodes.execute import execute_prep_code

        state = {"output_dir": str(tmp_path), "prep_code_path": ""}
        result = execute_prep_code(state)
        assert result["prep_succeeded"] is False


# ---------------------------------------------------------------------------
# check_prep_result
# ---------------------------------------------------------------------------

class TestCheckPrepResult:
    """Tests for the check_prep_result routing function."""

    def test_success_routes_to_verify(self):
        from auto_llm_predictor.nodes.execute import check_prep_result

        state = {"prep_succeeded": True, "prep_attempts": 1}
        assert check_prep_result(state) == "verify_prepared_data"

    def test_retry_when_under_max(self):
        from auto_llm_predictor.nodes.execute import check_prep_result

        state = {"prep_succeeded": False, "prep_attempts": 2}
        assert check_prep_result(state) == "write_prep_code"

    def test_gives_up_at_max_attempts(self):
        from auto_llm_predictor.nodes.execute import check_prep_result

        state = {"prep_succeeded": False, "prep_attempts": 3}
        assert check_prep_result(state) == "verify_prepared_data"

    def test_logs_error_at_max_attempts(self, caplog):
        from auto_llm_predictor.nodes.execute import check_prep_result

        state = {"prep_succeeded": False, "prep_attempts": 3}
        with caplog.at_level(logging.ERROR, logger="auto_llm_predictor.nodes.execute"):
            check_prep_result(state)
        assert any("Max prep attempts" in r.message for r in caplog.records)
