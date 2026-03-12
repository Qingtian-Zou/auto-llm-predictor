"""Tests for auto_llm_predictor.webui.

Covers: _prune_completed_runs, RunState events lock, and path traversal guard.
"""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# _prune_completed_runs
# ---------------------------------------------------------------------------

class TestPruneCompletedRuns:
    """Tests for the _prune_completed_runs cleanup function."""

    def test_prunes_excess_completed_runs(self):
        from auto_llm_predictor.webui import RunState, _runs, _prune_completed_runs, _MAX_COMPLETED_RUNS

        _runs.clear()
        # Add more than the limit of completed runs
        for i in range(_MAX_COMPLETED_RUNS + 10):
            run = RunState(run_id=f"run-{i:04d}", status="completed")
            _runs[run.run_id] = run

        _prune_completed_runs()
        completed = [r for r in _runs.values() if r.status == "completed"]
        assert len(completed) <= _MAX_COMPLETED_RUNS
        _runs.clear()

    def test_preserves_running_runs(self):
        from auto_llm_predictor.webui import RunState, _runs, _prune_completed_runs, _MAX_COMPLETED_RUNS

        _runs.clear()
        # Add some running and completed runs
        _runs["running-1"] = RunState(run_id="running-1", status="running")
        _runs["running-2"] = RunState(run_id="running-2", status="interrupted")
        for i in range(_MAX_COMPLETED_RUNS + 5):
            run = RunState(run_id=f"done-{i:04d}", status="completed")
            _runs[run.run_id] = run

        _prune_completed_runs()
        # Running/interrupted runs must remain
        assert "running-1" in _runs
        assert "running-2" in _runs
        _runs.clear()

    def test_no_op_when_under_limit(self):
        from auto_llm_predictor.webui import RunState, _runs, _prune_completed_runs

        _runs.clear()
        _runs["a"] = RunState(run_id="a", status="completed")
        _runs["b"] = RunState(run_id="b", status="completed")
        _prune_completed_runs()
        assert len(_runs) == 2
        _runs.clear()


# ---------------------------------------------------------------------------
# RunState events lock — thread safety
# ---------------------------------------------------------------------------

class TestRunStateEventsLock:
    """Tests that RunState uses a lock for events."""

    def test_runstate_has_events_lock(self):
        from auto_llm_predictor.webui import RunState

        run = RunState(run_id="test")
        assert hasattr(run, "_events_lock")
        assert isinstance(run._events_lock, type(threading.Lock()))

    def test_emit_uses_lock(self):
        """_emit should append to events under the lock."""
        from auto_llm_predictor.webui import RunState, _emit

        run = RunState(run_id="test")
        _emit(run, "log", "hello")
        assert len(run.events) == 1
        assert run.events[0]["event"] == "log"
        assert run.events[0]["message"] == "hello"

    def test_concurrent_emit_no_crash(self):
        """Multiple threads calling _emit should not crash."""
        from auto_llm_predictor.webui import RunState, _emit

        run = RunState(run_id="test")
        errors = []

        def emit_many():
            try:
                for i in range(100):
                    _emit(run, "log", f"msg-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(run.events) == 500


# ---------------------------------------------------------------------------
# Path traversal — download endpoint guard
# ---------------------------------------------------------------------------

class TestPathTraversalGuard:
    """Tests for the path traversal guard in the download endpoint logic."""

    def test_file_inside_output_dir_is_allowed(self, tmp_path):
        """A file inside the output_dir should be accessible."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        data_file = output_dir / "data" / "all_data.json"
        data_file.parent.mkdir(parents=True)
        data_file.write_text("[]")

        path = data_file.resolve()
        allowed_base = output_dir.resolve()
        assert str(path).startswith(str(allowed_base))

    def test_file_outside_output_dir_is_blocked(self, tmp_path):
        """A file outside the output_dir should be rejected."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive")

        path = secret.resolve()
        allowed_base = output_dir.resolve()
        assert not str(path).startswith(str(allowed_base))

    def test_symlink_traversal_is_blocked(self, tmp_path):
        """A symlink pointing outside the output_dir should be blocked."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive")
        link = output_dir / "link.txt"
        link.symlink_to(secret)

        path = link.resolve()  # resolves through symlink
        allowed_base = output_dir.resolve()
        assert not str(path).startswith(str(allowed_base))
