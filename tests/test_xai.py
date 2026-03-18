"""Tests for auto_llm_predictor.xai (standalone XAI mode).

Covers: run_standalone_xai — happy path, error cases, and hardware defaults.
All tests run without GPU or LLM API via monkeypatching.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _patch_xai_internals(monkeypatch, *, shap_result=None, tl_result=None,
                         attn_result=None, captured=None):
    """Monkeypatch all GPU-dependent XAI helpers in nodes.explain."""
    if captured is None:
        captured = {}

    def fake_merge(base_model, adapter_path, training_config):
        captured["training_config"] = training_config
        return "fake_model", "fake_tokenizer"

    def fake_shap(model, tokenizer, samples, xai_dir, log_callback=None):
        captured["shap_samples"] = samples
        return shap_result

    def fake_tl(model, tokenizer, base_model, samples, xai_dir, log_callback=None):
        return tl_result

    def fake_attn(model, tokenizer, samples, log_callback=None):
        return attn_result

    monkeypatch.setattr("auto_llm_predictor.nodes.explain._merge_and_load", fake_merge)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_shap", fake_shap)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_transformer_lens", fake_tl)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_attention", fake_attn)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._release_model", lambda m: None)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._cleanup_gpu", lambda: None)
    monkeypatch.setattr("auto_llm_predictor.nodes.explain._save_heatmap",
                        lambda results, path: None)

    return captured


def _setup_training_output(tmp_path, *, test_data=None, adapter=True):
    """Create a minimal training output directory structure."""
    from auto_llm_predictor.checkpoint import save_state

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    test_data_path = output_dir / "data" / "test.json"
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    if test_data is not None:
        test_data_path.write_text(json.dumps(test_data))

    adapter_path = run_dir / "sft"
    if adapter:
        adapter_path.mkdir()

    save_state({
        "base_model": "test-model",
        "adapter_path": str(adapter_path),
        "training_config": {"precision": "bf16"},
        "test_data_path": str(test_data_path),
    }, str(output_dir))

    return output_dir, run_dir


def _make_shap_result(samples):
    """Build a fake SHAP result from sample list."""
    return {
        "method": "shap",
        "num_samples": len(samples),
        "sample_explanations": [
            {
                "sample_index": i,
                "input_preview": s["input"][:50],
                "true_label": s["output"],
                "token_scores": [{"token": "test", "score": 0.5}],
            }
            for i, s in enumerate(samples)
        ],
    }


# ---------------------------------------------------------------------------
# run_standalone_xai
# ---------------------------------------------------------------------------

class TestRunStandaloneXai:
    """Tests for xai.run_standalone_xai."""

    def test_happy_path(self, tmp_path, monkeypatch):
        test_data = [
            {"instruction": "Classify", "input": "age: 30", "output": "A"},
            {"instruction": "Classify", "input": "age: 40", "output": "B"},
        ]
        output_dir, run_dir = _setup_training_output(tmp_path, test_data=test_data)

        captured = _patch_xai_internals(
            monkeypatch,
            shap_result={
                "method": "shap",
                "num_samples": 2,
                "sample_explanations": [
                    {"sample_index": 0, "input_preview": "age: 30",
                     "true_label": "A",
                     "token_scores": [{"token": "age", "score": 0.8}]},
                    {"sample_index": 1, "input_preview": "age: 40",
                     "true_label": "B",
                     "token_scores": [{"token": "age", "score": 0.7}]},
                ],
            },
            captured={},
        )

        from auto_llm_predictor.xai import run_standalone_xai

        result = run_standalone_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
        )

        # Verify report saved
        assert result["xai_report_path"]
        assert Path(result["xai_report_path"]).exists()
        report = json.loads(Path(result["xai_report_path"]).read_text())
        assert report["methods_succeeded"] == ["shap"]
        assert report["num_samples"] == 2

        # Verify return structure
        assert result["methods_succeeded"] == ["shap"]
        assert len(result["xai_results"]) == 1
        assert result["num_samples"] == 2

    def test_missing_test_data_raises(self, tmp_path, monkeypatch):
        output_dir, run_dir = _setup_training_output(
            tmp_path, test_data=None,
        )
        _patch_xai_internals(monkeypatch)

        from auto_llm_predictor.xai import run_standalone_xai

        with pytest.raises(FileNotFoundError, match="[Tt]est data"):
            run_standalone_xai(
                output_dir=str(output_dir),
                run_dir=str(run_dir),
            )

    def test_missing_adapter_raises(self, tmp_path, monkeypatch):
        test_data = [{"instruction": "X", "input": "a", "output": "b"}]
        output_dir, run_dir = _setup_training_output(
            tmp_path, test_data=test_data, adapter=False,
        )
        _patch_xai_internals(monkeypatch)

        from auto_llm_predictor.xai import run_standalone_xai

        with pytest.raises(FileNotFoundError, match="[Aa]dapter"):
            run_standalone_xai(
                output_dir=str(output_dir),
                run_dir=str(run_dir),
            )

    def test_empty_test_data_raises(self, tmp_path, monkeypatch):
        output_dir, run_dir = _setup_training_output(tmp_path, test_data=[])
        _patch_xai_internals(monkeypatch)

        from auto_llm_predictor.xai import run_standalone_xai

        with pytest.raises(ValueError, match="[Ee]mpty"):
            run_standalone_xai(
                output_dir=str(output_dir),
                run_dir=str(run_dir),
            )

    def test_max_samples_cap(self, tmp_path, monkeypatch):
        test_data = [
            {"instruction": "Classify", "input": f"val: {i}", "output": str(i)}
            for i in range(100)
        ]
        output_dir, run_dir = _setup_training_output(tmp_path, test_data=test_data)

        captured = _patch_xai_internals(
            monkeypatch,
            shap_result=None,
            attn_result={
                "method": "attention",
                "num_samples": 10,
                "sample_explanations": [],
            },
            captured={},
        )

        from auto_llm_predictor.xai import run_standalone_xai

        result = run_standalone_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
            max_samples=10,
        )

        assert result["num_samples"] == 10

    def test_all_methods_fail(self, tmp_path, monkeypatch):
        test_data = [{"instruction": "X", "input": "a", "output": "b"}]
        output_dir, run_dir = _setup_training_output(tmp_path, test_data=test_data)

        _patch_xai_internals(monkeypatch)

        from auto_llm_predictor.xai import run_standalone_xai

        result = run_standalone_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
        )

        assert result["xai_report_path"] == ""
        assert result["xai_results"] == []
        assert result["methods_succeeded"] == []

    def test_hardware_defaults_applied(self, tmp_path, monkeypatch):
        test_data = [{"instruction": "X", "input": "a", "output": "b"}]
        output_dir, run_dir = _setup_training_output(tmp_path, test_data=test_data)

        captured = _patch_xai_internals(
            monkeypatch,
            shap_result={"method": "shap", "num_samples": 1, "sample_explanations": []},
            captured={},
        )

        from auto_llm_predictor.xai import run_standalone_xai

        run_standalone_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
        )

        # Default precision is fp16 and quantization is 8-bit
        assert captured["training_config"]["precision"] == "fp16"
        assert captured["training_config"]["quantization_bit"] == 8

    def test_fallback_when_state_paths_stale(self, tmp_path, monkeypatch):
        """When adapter_path/test_data_path in state are stale, fall back to
        user-provided run_dir/output_dir."""
        from auto_llm_predictor.checkpoint import save_state

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "sft").mkdir()

        test_data_path = output_dir / "data" / "test.json"
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        test_data_path.write_text(
            json.dumps([{"instruction": "X", "input": "a", "output": "b"}]),
        )

        # Save state with deliberately wrong absolute paths
        save_state({
            "base_model": "test-model",
            "adapter_path": "/nonexistent/path/sft",
            "training_config": {"precision": "bf16"},
            "test_data_path": "/nonexistent/path/test.json",
        }, str(output_dir))

        _patch_xai_internals(
            monkeypatch,
            shap_result={"method": "shap", "num_samples": 1, "sample_explanations": []},
        )

        from auto_llm_predictor.xai import run_standalone_xai

        # Should succeed using fallback paths from run_dir and output_dir
        result = run_standalone_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
        )
        assert result["methods_succeeded"] == ["shap"]
