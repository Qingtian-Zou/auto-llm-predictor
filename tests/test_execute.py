"""Tests for auto_llm_predictor.nodes.execute.

Covers: execute_prep_code guard and check_prep_result routing, plus the
CLI-arg passing and transformers.pkl validation.
"""

from __future__ import annotations

import json
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

    def test_retry_routes_to_debug(self):
        from auto_llm_predictor.nodes.execute import check_prep_result

        state = {"prep_succeeded": False, "prep_attempts": 2}
        assert check_prep_result(state) == "debug_prep_failure"

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


# ---------------------------------------------------------------------------
# execute_prep_code — CLI args + transformers.pkl validation
# ---------------------------------------------------------------------------


class TestExecutePrepCodeCliInvocation:
    """Verify that execute_prep_code passes --input-csv / --test-csv /
    --output-dir to the generated script and validates transformers.pkl."""

    def _make_script_capturing_args(self, tmp_path, *, write_outputs: bool):
        """Create a fake prep script that records its argv and optionally
        writes the expected output files."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        data_dir = tmp_path / "data"
        argv_log = tmp_path / "argv.json"

        write_block = ""
        if write_outputs:
            write_block = (
                "from pathlib import Path\n"
                "import argparse, json, pickle\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--input-csv', required=True)\n"
                "p.add_argument('--test-csv', default='')\n"
                "p.add_argument('--output-dir', required=True)\n"
                "p.add_argument('--predict-only', action='store_true')\n"
                "ns = p.parse_args()\n"
                "out = Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'all_data.json').write_text(json.dumps([\n"
                "    {'instruction': 'I', 'input': 'a', 'output': 'A'},\n"
                "]))\n"
                "(out / 'dataset_info.json').write_text(json.dumps({\n"
                "    'train': {'file_name': 'all_data.json'},\n"
                "    'test':  {'file_name': 'test_data.json'},\n"
                "}))\n"
                "(out / 'test_data.json').write_text(json.dumps([\n"
                "    {'instruction': 'I', 'input': 'b', 'output': 'B'},\n"
                "]))\n"
                "(out / 'transformers.pkl').write_bytes(pickle.dumps({}))\n"
            )

        script = scripts_dir / "prepare_data.py"
        script.write_text(
            "import sys, json\n"
            f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv))\n"
            + write_block
        )
        return script, data_dir, argv_log

    def test_passes_input_test_output_args(self, tmp_path):
        from auto_llm_predictor.nodes.execute import execute_prep_code

        script, data_dir, argv_log = self._make_script_capturing_args(
            tmp_path, write_outputs=True,
        )

        state = {
            "output_dir": str(tmp_path),
            "prep_code_path": str(script),
            "csv_path": "/tmp/in.csv",
            "test_csv_path": "/tmp/test.csv",
            "task_type": "binary",
        }
        result = execute_prep_code(state)

        argv = json.loads(argv_log.read_text())
        assert "--input-csv" in argv
        assert argv[argv.index("--input-csv") + 1] == "/tmp/in.csv"
        assert "--test-csv" in argv
        assert argv[argv.index("--test-csv") + 1] == "/tmp/test.csv"
        assert "--output-dir" in argv
        assert argv[argv.index("--output-dir") + 1] == str(data_dir)
        # Sanity: the script ran and the node reports success.
        assert result["prep_succeeded"] is True

    def test_omits_test_csv_arg_when_not_set(self, tmp_path):
        from auto_llm_predictor.nodes.execute import execute_prep_code

        script, _, argv_log = self._make_script_capturing_args(
            tmp_path, write_outputs=True,
        )
        # Strip test_data.json since this scenario has no separate test CSV.
        # (The node will still expect it because it currently only relies on
        # state["test_csv_path"] to decide whether test_data.json is required.)

        state = {
            "output_dir": str(tmp_path),
            "prep_code_path": str(script),
            "csv_path": "/tmp/in.csv",
            "test_csv_path": "",
            "task_type": "binary",
        }
        execute_prep_code(state)

        argv = json.loads(argv_log.read_text())
        assert "--test-csv" not in argv

    def test_fails_when_transformers_pkl_missing(self, tmp_path):
        from auto_llm_predictor.nodes.execute import execute_prep_code

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        data_dir = tmp_path / "data"
        script = scripts_dir / "prepare_data.py"
        # Writes outputs but NOT transformers.pkl.
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--input-csv', required=True)\n"
            "p.add_argument('--test-csv', default='')\n"
            "p.add_argument('--output-dir', required=True)\n"
            "p.add_argument('--predict-only', action='store_true')\n"
            "ns = p.parse_args()\n"
            "out = Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'all_data.json').write_text(json.dumps([\n"
            "    {'instruction': 'I', 'input': 'a', 'output': 'A'},\n"
            "]))\n"
            "(out / 'dataset_info.json').write_text('{}')\n"
            "(out / 'test_data.json').write_text(json.dumps([\n"
            "    {'instruction': 'I', 'input': 'b', 'output': 'B'},\n"
            "]))\n"
        )

        state = {
            "output_dir": str(tmp_path),
            "prep_code_path": str(script),
            "csv_path": "/tmp/in.csv",
            "test_csv_path": "/tmp/test.csv",
            "task_type": "binary",
        }
        result = execute_prep_code(state)

        assert result["prep_succeeded"] is False
        assert "transformers.pkl" in result["prep_result"]
