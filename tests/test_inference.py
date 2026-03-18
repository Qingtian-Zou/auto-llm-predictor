"""Tests for auto_llm_predictor.inference.

Covers: generate_inference_yaml, _format_features_like_example, and get_feature_names.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# generate_inference_yaml
# ---------------------------------------------------------------------------

class TestGenerateInferenceYaml:
    """Tests for inference.generate_inference_yaml."""

    def test_produces_valid_yaml(self, tmp_path):
        from auto_llm_predictor.inference import generate_inference_yaml
        import yaml

        yaml_path = generate_inference_yaml(
            base_model="test-model",
            adapter_path=str(tmp_path / "adapter"),
            data_dir=str(tmp_path / "data"),
            dataset_name="infer",
            template="default",
            cutoff_len=4096,
            output_dir=str(tmp_path / "output"),
            precision="bf16",
            flash_attn="auto",
        )

        assert Path(yaml_path).exists()
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["model_name_or_path"] == "test-model"
        assert config["do_predict"] is True
        assert config["finetuning_type"] == "lora"
        assert config["cutoff_len"] == 4096
        assert config["template"] == "default"
        assert config["dataset"] == "infer"

    def test_includes_quantization_when_set(self, tmp_path):
        from auto_llm_predictor.inference import generate_inference_yaml
        import yaml

        yaml_path = generate_inference_yaml(
            base_model="test-model",
            adapter_path=str(tmp_path / "adapter"),
            data_dir=str(tmp_path / "data"),
            dataset_name="infer",
            template="default",
            cutoff_len=2048,
            output_dir=str(tmp_path / "output"),
            quantization_bit=4,
        )

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["quantization_bit"] == 4

    def test_output_dir_created(self, tmp_path):
        from auto_llm_predictor.inference import generate_inference_yaml

        out = tmp_path / "nonexistent" / "output"
        generate_inference_yaml(
            base_model="m",
            adapter_path="a",
            data_dir="d",
            dataset_name="x",
            template="t",
            cutoff_len=512,
            output_dir=str(out),
        )

        assert (out / "configs" / "infer_predict.yaml").exists()


# ---------------------------------------------------------------------------
# _format_features_like_example
# ---------------------------------------------------------------------------

class TestFormatFeaturesLikeExample:
    """Tests for inference._format_features_like_example."""

    def _fmt(self, example, features):
        from auto_llm_predictor.inference import _format_features_like_example
        return _format_features_like_example(example, features)

    def test_newline_separated(self):
        example = "age: 25\nbmi: 22.5\nsmoker: No"
        result = self._fmt(example, {"age": "30", "bmi": "25"})
        assert "age: 30" in result
        assert "bmi: 25" in result
        assert "\n" in result

    def test_comma_separated(self):
        example = "age: 25, bmi: 22.5, smoker: No"
        result = self._fmt(example, {"age": "30", "bmi": "25"})
        assert "age: 30" in result
        assert ", " in result

    def test_equals_format(self):
        example = "age=25, bmi=22.5"
        result = self._fmt(example, {"age": "30"})
        assert "age=30" in result

    def test_default_format(self):
        example = "some plain text"
        result = self._fmt(example, {"age": "30"})
        assert "age: 30" in result


# ---------------------------------------------------------------------------
# get_feature_names
# ---------------------------------------------------------------------------

class TestGetFeatureNames:
    """Tests for inference.get_feature_names."""

    def test_returns_features_from_state(self, tmp_path):
        from auto_llm_predictor.inference import get_feature_names
        from auto_llm_predictor.checkpoint import save_state

        state = {
            "selected_features": ["age", "bmi", "smoker"],
            "base_model": "test",
        }
        save_state(state, str(tmp_path))

        features = get_feature_names(str(tmp_path))
        assert features == ["age", "bmi", "smoker"]

    def test_returns_empty_when_no_features(self, tmp_path):
        from auto_llm_predictor.inference import get_feature_names
        from auto_llm_predictor.checkpoint import save_state

        save_state({"base_model": "test"}, str(tmp_path))
        features = get_feature_names(str(tmp_path))
        assert features == []


# ---------------------------------------------------------------------------
# _run_single_xai — prediction passed as SHAP target
# ---------------------------------------------------------------------------

class TestRunSingleXaiPrediction:
    """Verify that the predicted label is forwarded as the sample output."""

    def test_prediction_passed_as_output(self, tmp_path, monkeypatch):
        captured = {}

        def fake_shap(model, tokenizer, samples, xai_dir, log_callback=None):
            captured["shap_samples"] = samples
            return None

        def fake_tl(model, tokenizer, base_model, samples, xai_dir, log_callback=None):
            return None

        def fake_attn(model, tokenizer, samples, log_callback=None):
            captured["attn_samples"] = samples
            return {"method": "attention", "sample_explanations": []}

        monkeypatch.setattr(
            "auto_llm_predictor.nodes.explain._run_shap", fake_shap,
        )
        monkeypatch.setattr(
            "auto_llm_predictor.nodes.explain._run_transformer_lens", fake_tl,
        )
        monkeypatch.setattr(
            "auto_llm_predictor.nodes.explain._run_attention", fake_attn,
        )
        monkeypatch.setattr(
            "auto_llm_predictor.nodes.explain._cleanup_gpu", lambda: None,
        )

        from auto_llm_predictor.inference import _run_single_xai

        _run_single_xai(
            model=None,
            tokenizer=None,
            base_model="test",
            prompt="Predict class",
            state={},
            run_dir=str(tmp_path),
            prediction="class_A",
        )

        assert captured["shap_samples"][0]["output"] == "class_A"
        assert captured["attn_samples"][0]["output"] == "class_A"


# ---------------------------------------------------------------------------
# _build_xai_samples — pairing inference data with predictions
# ---------------------------------------------------------------------------

import json


class TestBuildXaiSamples:
    """Tests for inference._build_xai_samples."""

    def _write_fixtures(self, tmp_path, data, predictions):
        data_path = tmp_path / "all_data.json"
        data_path.write_text(json.dumps(data))

        preds_path = tmp_path / "generated_predictions.jsonl"
        with open(preds_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        return str(data_path), str(preds_path)

    def test_basic_pairing(self, tmp_path):
        from auto_llm_predictor.inference import _build_xai_samples

        data = [
            {"instruction": "Classify", "input": "age: 30", "output": ""},
            {"instruction": "Classify", "input": "age: 40", "output": ""},
        ]
        preds = [
            {"predict": "class_A", "label": ""},
            {"predict": "class_B", "label": ""},
        ]
        data_path, preds_path = self._write_fixtures(tmp_path, data, preds)

        samples = _build_xai_samples(data_path, preds_path)

        assert len(samples) == 2
        assert samples[0]["output"] == "class_A"
        assert samples[0]["instruction"] == "Classify"
        assert samples[0]["input"] == "age: 30"
        assert samples[1]["output"] == "class_B"

    def test_cap_at_max_samples(self, tmp_path):
        from auto_llm_predictor.inference import _build_xai_samples

        data = [{"instruction": "X", "input": str(i), "output": ""} for i in range(100)]
        preds = [{"predict": f"pred_{i}"} for i in range(100)]
        data_path, preds_path = self._write_fixtures(tmp_path, data, preds)

        samples = _build_xai_samples(data_path, preds_path, max_samples=10)
        assert len(samples) == 10

    def test_mismatched_sizes(self, tmp_path):
        from auto_llm_predictor.inference import _build_xai_samples

        data = [
            {"instruction": "X", "input": "a", "output": ""},
            {"instruction": "X", "input": "b", "output": ""},
            {"instruction": "X", "input": "c", "output": ""},
        ]
        preds = [{"predict": "p1"}]
        data_path, preds_path = self._write_fixtures(tmp_path, data, preds)

        samples = _build_xai_samples(data_path, preds_path)
        assert len(samples) == 1
        assert samples[0]["output"] == "p1"

    def test_empty_predictions(self, tmp_path):
        from auto_llm_predictor.inference import _build_xai_samples

        data = [{"instruction": "X", "input": "a", "output": ""}]
        data_path = tmp_path / "all_data.json"
        data_path.write_text(json.dumps(data))
        preds_path = tmp_path / "generated_predictions.jsonl"
        preds_path.write_text("")

        samples = _build_xai_samples(str(data_path), str(preds_path))
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# run_batch_xai — end-to-end with monkeypatched XAI methods
# ---------------------------------------------------------------------------

class TestRunBatchXai:
    """Tests for inference.run_batch_xai."""

    def test_runs_xai_and_saves_report(self, tmp_path, monkeypatch):
        from auto_llm_predictor.checkpoint import save_state

        # Set up pipeline state
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        save_state({
            "base_model": "test-model",
            "adapter_path": str(tmp_path / "adapter"),
            "training_config": {"precision": "bf16"},
        }, str(output_dir))

        # Create adapter directory
        (tmp_path / "adapter").mkdir()

        # Create inference data + predictions
        infer_output = tmp_path / "infer"
        data_dir = infer_output / "data"
        data_dir.mkdir(parents=True)

        data = [
            {"instruction": "Classify", "input": "age: 30", "output": ""},
            {"instruction": "Classify", "input": "age: 40", "output": ""},
        ]
        (data_dir / "all_data.json").write_text(json.dumps(data))

        preds_path = infer_output / "generated_predictions.jsonl"
        with open(preds_path, "w") as f:
            f.write(json.dumps({"predict": "class_A"}) + "\n")
            f.write(json.dumps({"predict": "class_B"}) + "\n")

        # Monkeypatch XAI internals
        captured = {}

        def fake_merge(base_model, adapter_path, training_config):
            captured["training_config"] = training_config
            return "fake_model", "fake_tokenizer"

        def fake_shap(model, tokenizer, samples, xai_dir, log_callback=None):
            captured["shap_samples"] = samples
            return {
                "method": "shap",
                "num_samples": len(samples),
                "sample_explanations": [
                    {"sample_index": i, "input_preview": s["input"][:50],
                     "true_label": s["output"],
                     "token_scores": [{"token": "test", "score": 0.5}]}
                    for i, s in enumerate(samples)
                ],
            }

        def fake_tl(model, tokenizer, base_model, samples, xai_dir, log_callback=None):
            return None

        monkeypatch.setattr("auto_llm_predictor.nodes.explain._merge_and_load", fake_merge)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_shap", fake_shap)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_transformer_lens", fake_tl)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_attention",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._release_model", lambda m: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._cleanup_gpu", lambda: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._save_heatmap",
                            lambda results, path: None)

        from auto_llm_predictor.inference import run_batch_xai

        result = run_batch_xai(
            output_dir=str(output_dir),
            run_dir=str(tmp_path),
            infer_output=str(infer_output),
            predictions_path=str(preds_path),
        )

        # Verify XAI samples have predictions as output
        assert captured["shap_samples"][0]["output"] == "class_A"
        assert captured["shap_samples"][1]["output"] == "class_B"

        # Verify hardware defaults applied
        assert captured["training_config"]["precision"] == "fp16"
        assert captured["training_config"]["quantization_bit"] == 8

        # Verify report saved
        assert result["xai_report_path"]
        assert Path(result["xai_report_path"]).exists()
        report = json.loads(Path(result["xai_report_path"]).read_text())
        assert report["methods_succeeded"] == ["shap"]
        assert report["num_samples"] == 2

        # Verify return structure
        assert result["methods_succeeded"] == ["shap"]
        assert len(result["xai_results"]) == 1

    def test_returns_empty_when_all_methods_fail(self, tmp_path, monkeypatch):
        from auto_llm_predictor.checkpoint import save_state

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        save_state({
            "base_model": "test-model",
            "adapter_path": str(tmp_path / "adapter"),
            "training_config": {},
        }, str(output_dir))
        (tmp_path / "adapter").mkdir()

        infer_output = tmp_path / "infer"
        data_dir = infer_output / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "all_data.json").write_text(
            json.dumps([{"instruction": "X", "input": "a", "output": ""}]),
        )
        preds_path = infer_output / "preds.jsonl"
        preds_path.write_text(json.dumps({"predict": "p"}) + "\n")

        monkeypatch.setattr("auto_llm_predictor.nodes.explain._merge_and_load",
                            lambda *a: ("m", "t"))
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_shap",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_transformer_lens",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_attention",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._release_model", lambda m: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._cleanup_gpu", lambda: None)

        from auto_llm_predictor.inference import run_batch_xai

        result = run_batch_xai(
            output_dir=str(output_dir),
            run_dir=str(tmp_path),
            infer_output=str(infer_output),
            predictions_path=str(preds_path),
        )

        assert result["xai_report_path"] == ""
        assert result["xai_results"] == []
        assert result["methods_succeeded"] == []

    def test_fallback_adapter_from_run_dir(self, tmp_path, monkeypatch):
        """When adapter_path in state is stale, fall back to run_dir/sft."""
        from auto_llm_predictor.checkpoint import save_state

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "sft").mkdir()

        save_state({
            "base_model": "test-model",
            "adapter_path": "/nonexistent/path/sft",
            "training_config": {},
        }, str(output_dir))

        infer_output = tmp_path / "infer"
        data_dir = infer_output / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "all_data.json").write_text(
            json.dumps([{"instruction": "X", "input": "a", "output": ""}]),
        )
        preds_path = infer_output / "preds.jsonl"
        preds_path.write_text(json.dumps({"predict": "p"}) + "\n")

        monkeypatch.setattr("auto_llm_predictor.nodes.explain._merge_and_load",
                            lambda *a: ("m", "t"))
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_shap",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_transformer_lens",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._run_attention",
                            lambda *a, **kw: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._release_model", lambda m: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._cleanup_gpu", lambda: None)
        monkeypatch.setattr("auto_llm_predictor.nodes.explain._save_heatmap",
                            lambda results, path: None)

        from auto_llm_predictor.inference import run_batch_xai

        # Should not raise — falls back to run_dir/sft
        result = run_batch_xai(
            output_dir=str(output_dir),
            run_dir=str(run_dir),
            infer_output=str(infer_output),
            predictions_path=str(preds_path),
        )
        assert result["xai_results"] == []  # all methods returned None
