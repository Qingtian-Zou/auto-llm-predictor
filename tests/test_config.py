"""Tests for auto_llm_predictor.nodes.config.

Covers: _guess_template with both HuggingFace IDs and local model paths.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# _guess_template
# ---------------------------------------------------------------------------

class TestGuessTemplate:
    """Tests for config._guess_template."""

    def test_huggingface_llama(self):
        from auto_llm_predictor.nodes.config import _guess_template

        assert _guess_template("meta-llama/Llama-3-8B-Instruct") == "llama3"

    def test_huggingface_qwen(self):
        from auto_llm_predictor.nodes.config import _guess_template

        assert _guess_template("Qwen/Qwen2.5-7B-Instruct") == "qwen"

    def test_huggingface_mistral(self):
        from auto_llm_predictor.nodes.config import _guess_template

        assert _guess_template("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"

    def test_unknown_huggingface(self):
        from auto_llm_predictor.nodes.config import _guess_template

        assert _guess_template("some-org/totally-custom-model") == "default"

    def test_local_path_with_config_json(self, tmp_path):
        from auto_llm_predictor.nodes.config import _guess_template

        (tmp_path / "config.json").write_text('{"model_type": "gemma2"}')
        assert _guess_template(str(tmp_path)) == "gemma"

    def test_local_path_without_config_json(self, tmp_path):
        from auto_llm_predictor.nodes.config import _guess_template

        assert _guess_template(str(tmp_path)) == "default"

    def test_local_path_unknown_model_type(self, tmp_path):
        from auto_llm_predictor.nodes.config import _guess_template

        (tmp_path / "config.json").write_text('{"model_type": "custom_arch"}')
        assert _guess_template(str(tmp_path)) == "default"
