"""Tests for auto_llm_predictor.utils.

Covers: profile_csv, run_script, load_jsonl, find_latest_checkpoint,
set_resume_in_yaml, and unused-import checks.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# profile_csv
# ---------------------------------------------------------------------------

class TestProfileCSV:
    """Tests for utils.profile_csv."""

    def test_normal_csv(self, tmp_path):
        """profile_csv returns a meaningful summary for a valid CSV."""
        from auto_llm_predictor.utils import profile_csv

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,yes\n3,4,no\n5,6,yes\n")
        result = profile_csv(str(csv))
        assert "data.csv" in result
        assert "3 rows" in result
        assert "3 columns" in result

    def test_malformed_csv(self, tmp_path):
        """profile_csv returns an error message instead of crashing on a bad file."""
        from auto_llm_predictor.utils import profile_csv

        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        result = profile_csv(str(bad))
        assert "ERROR" in result or "error" in result.lower()

    def test_empty_csv(self, tmp_path):
        """profile_csv handles an empty file gracefully."""
        from auto_llm_predictor.utils import profile_csv

        empty = tmp_path / "empty.csv"
        empty.write_text("")
        result = profile_csv(str(empty))
        # Should return something (error or degenerate summary), not crash
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operator_precedence_fix(self, tmp_path):
        """Categorical column detection uses correct precedence.

        A numeric column with <=10 unique values should be included,
        and an object column with >20 unique values should NOT be included
        in the value-counts section.
        """
        from auto_llm_predictor.utils import profile_csv

        # Build a CSV with a numeric low-cardinality column (should show up)
        # and a string high-cardinality column (should NOT show up for value counts)
        rows = ["id,score\n"]
        for i in range(30):
            rows.append(f"name_{i},{i % 5}\n")
        csv = tmp_path / "prec.csv"
        csv.write_text("".join(rows))

        result = profile_csv(str(csv))
        assert "Value counts for 'score'" in result
        # 'id' has 30 unique string values — should not show value counts
        assert "Value counts for 'id'" not in result


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------

class TestRunScript:
    """Tests for utils.run_script."""

    def test_successful_script(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "ok.py"
        script.write_text("print('hello')\n")
        success, output = run_script(str(script))
        assert success is True
        assert "hello" in output

    def test_failing_script(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "fail.py"
        script.write_text("raise ValueError('boom')\n")
        success, output = run_script(str(script))
        assert success is False
        assert "boom" in output

    def test_timeout(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)\n")
        success, output = run_script(str(script), timeout=1)
        assert success is False
        assert "timed out" in output.lower()


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJSONL:
    def test_normal(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        result = load_jsonl(str(f))
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_empty_file(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_jsonl(str(f)) == []

    def test_blank_lines(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "blanks.jsonl"
        f.write_text('{"a": 1}\n\n\n{"b": 2}\n\n')
        result = load_jsonl(str(f))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    """Tests for utils.find_latest_checkpoint."""

    def test_no_checkpoints(self, tmp_path):
        from auto_llm_predictor.utils import find_latest_checkpoint

        sft = tmp_path / "sft"
        sft.mkdir()
        assert find_latest_checkpoint(str(sft)) is None

    def test_nonexistent_dir(self, tmp_path):
        from auto_llm_predictor.utils import find_latest_checkpoint

        assert find_latest_checkpoint(str(tmp_path / "nonexistent")) is None

    def test_single_checkpoint(self, tmp_path):
        from auto_llm_predictor.utils import find_latest_checkpoint

        sft = tmp_path / "sft"
        (sft / "checkpoint-500").mkdir(parents=True)
        result = find_latest_checkpoint(str(sft))
        assert result is not None
        assert "checkpoint-500" in result

    def test_multiple_checkpoints_returns_latest(self, tmp_path):
        from auto_llm_predictor.utils import find_latest_checkpoint

        sft = tmp_path / "sft"
        for step in [500, 1000, 1500]:
            (sft / f"checkpoint-{step}").mkdir(parents=True)
        result = find_latest_checkpoint(str(sft))
        assert "checkpoint-1500" in result

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        from auto_llm_predictor.utils import find_latest_checkpoint

        sft = tmp_path / "sft"
        sft.mkdir()
        (sft / "logs").mkdir()
        (sft / "trainer_state.json").touch()
        assert find_latest_checkpoint(str(sft)) is None


# ---------------------------------------------------------------------------
# set_resume_in_yaml
# ---------------------------------------------------------------------------

class TestSetResumeInYaml:
    """Tests for utils.set_resume_in_yaml."""

    def test_sets_resume_true(self, tmp_path):
        import yaml
        from auto_llm_predictor.utils import set_resume_in_yaml

        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text("model_name_or_path: test\noutput_dir: /tmp/out\n")

        set_resume_in_yaml(str(yaml_path), resume=True)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config["resume_from_checkpoint"] is True
        assert config["model_name_or_path"] == "test"

    def test_removes_resume_false(self, tmp_path):
        import yaml
        from auto_llm_predictor.utils import set_resume_in_yaml

        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(
            "model_name_or_path: test\nresume_from_checkpoint: true\n"
        )

        set_resume_in_yaml(str(yaml_path), resume=False)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert "resume_from_checkpoint" not in config

    def test_idempotent_set(self, tmp_path):
        import yaml
        from auto_llm_predictor.utils import set_resume_in_yaml

        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(
            "model_name_or_path: test\nresume_from_checkpoint: true\n"
        )

        set_resume_in_yaml(str(yaml_path), resume=True)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config["resume_from_checkpoint"] is True


# ---------------------------------------------------------------------------
# Unused imports
# ---------------------------------------------------------------------------

class TestUnusedImports:
    """Verify unused imports were removed."""

    def test_no_textwrap_import(self):
        import inspect
        import auto_llm_predictor.utils as utils_mod

        source = inspect.getsource(utils_mod)
        # textwrap should no longer be imported (it was unused)
        lines = [l.strip() for l in source.splitlines() if l.strip().startswith("import textwrap")]
        assert lines == [], f"textwrap is still imported: {lines}"


# ---------------------------------------------------------------------------
# is_local_model
# ---------------------------------------------------------------------------

class TestIsLocalModel:
    """Tests for utils.is_local_model."""

    def test_existing_directory_returns_true(self, tmp_path):
        from auto_llm_predictor.utils import is_local_model

        assert is_local_model(str(tmp_path)) is True

    def test_huggingface_id_returns_false(self):
        from auto_llm_predictor.utils import is_local_model

        assert is_local_model("mistralai/Mistral-7B-Instruct-v0.3") is False

    def test_nonexistent_path_returns_false(self):
        from auto_llm_predictor.utils import is_local_model

        assert is_local_model("/nonexistent/path/to/model") is False


# ---------------------------------------------------------------------------
# validate_local_model
# ---------------------------------------------------------------------------

class TestValidateLocalModel:
    """Tests for utils.validate_local_model."""

    def test_valid_model_dir(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")
        (tmp_path / "tokenizer.json").write_text("{}")
        assert validate_local_model(str(tmp_path)) == []

    def test_missing_config_json(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        (tmp_path / "model.safetensors").write_text("")
        (tmp_path / "tokenizer.json").write_text("{}")
        errors = validate_local_model(str(tmp_path))
        assert len(errors) == 1
        assert "config.json" in errors[0]

    def test_missing_weights(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        errors = validate_local_model(str(tmp_path))
        assert len(errors) == 1
        assert "weights" in errors[0].lower()

    def test_missing_tokenizer(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")
        errors = validate_local_model(str(tmp_path))
        assert len(errors) == 1
        assert "tokenizer" in errors[0].lower()

    def test_all_missing(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        errors = validate_local_model(str(tmp_path))
        assert len(errors) == 3

    def test_sharded_weights_accepted(self, tmp_path):
        from auto_llm_predictor.utils import validate_local_model

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors.index.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        assert validate_local_model(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# detect_template_from_config
# ---------------------------------------------------------------------------

class TestDetectTemplateFromConfig:
    """Tests for utils.detect_template_from_config."""

    def test_llama_model_type(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        assert detect_template_from_config(str(tmp_path)) == "llama3"

    def test_qwen2_model_type(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
        assert detect_template_from_config(str(tmp_path)) == "qwen"

    def test_mistral_model_type(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        (tmp_path / "config.json").write_text('{"model_type": "mistral"}')
        assert detect_template_from_config(str(tmp_path)) == "mistral"

    def test_unknown_model_type(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        (tmp_path / "config.json").write_text('{"model_type": "unknown_arch"}')
        assert detect_template_from_config(str(tmp_path)) == "default"

    def test_missing_config_json(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        assert detect_template_from_config(str(tmp_path)) == "default"

    def test_malformed_config_json(self, tmp_path):
        from auto_llm_predictor.utils import detect_template_from_config

        (tmp_path / "config.json").write_text("not valid json {{")
        assert detect_template_from_config(str(tmp_path)) == "default"
