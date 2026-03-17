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
