/* ================================================================
   Auto LLM Predictor — Web UI Client
   Handles form submission, SSE streaming, and review interactions.
   ================================================================ */

(function () {
    "use strict";

    // ── DOM refs ────────────────────────────────────────────────
    const form = document.getElementById("run-form");
    const runBtn = document.getElementById("run-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    const fileInput = document.getElementById("csv-file");
    const fileDrop = document.getElementById("file-drop");
    const fileName = document.getElementById("file-name");
    const logEl = document.getElementById("log");
    const reviewPanel = document.getElementById("review-panel");
    const reviewSummary = document.getElementById("review-summary");
    const reviewForm = document.getElementById("review-form");
    const reviewInput = document.getElementById("review-input");
    const approveBtn = document.getElementById("approve-btn");
    const resultsPanel = document.getElementById("results-panel");
    const resultsContent = document.getElementById("results-content");
    const stepsEl = document.getElementById("pipeline-steps");
    const exportBtn = document.getElementById("export-btn");
    const exportMenu = document.getElementById("export-menu");

    let currentRunId = null;
    let evtSource = null;

    // ── Helpers ─────────────────────────────────────────────────
    function timestamp() {
        const d = new Date();
        return d.toLocaleTimeString("en-GB", { hour12: false });
    }

    function appendLog(msg, cls = "") {
        const line = document.createElement("div");
        line.className = "log__line" + (cls ? ` log__line--${cls}` : "");
        line.innerHTML = `<span class="log__time">${timestamp()}</span>${escapeHtml(msg)}`;
        logEl.appendChild(line);
        logEl.scrollTop = logEl.scrollHeight;
    }

    function escapeHtml(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    function setStepState(stepName, state) {
        const el = stepsEl.querySelector(`[data-step="${stepName}"]`);
        if (!el) return;
        el.classList.remove("step--active", "step--done", "step--error");
        if (state) el.classList.add(`step--${state}`);
    }

    function clearAllSteps() {
        stepsEl.querySelectorAll(".step").forEach(el => {
            el.classList.remove("step--active", "step--done", "step--error");
        });
    }

    // ── File drop behaviour ─────────────────────────────────────
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            fileName.textContent = fileInput.files[0].name;
            fileName.classList.add("file-drop__name--show");
        }
    });
    fileDrop.addEventListener("dragover", e => {
        e.preventDefault();
        fileDrop.classList.add("file-drop--active");
    });
    fileDrop.addEventListener("dragleave", () => {
        fileDrop.classList.remove("file-drop--active");
    });
    fileDrop.addEventListener("drop", e => {
        e.preventDefault();
        fileDrop.classList.remove("file-drop--active");
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            fileName.textContent = e.dataTransfer.files[0].name;
            fileName.classList.add("file-drop__name--show");
        }
    });

    // ── Test CSV file drop behaviour ────────────────────────────
    const testFileInput = document.getElementById("test-csv-file");
    const testFileDrop = document.getElementById("test-file-drop");
    const testFileName = document.getElementById("test-file-name");

    testFileInput.addEventListener("change", () => {
        if (testFileInput.files.length) {
            testFileName.textContent = testFileInput.files[0].name;
            testFileName.classList.add("file-drop__name--show");
        }
    });
    testFileDrop.addEventListener("dragover", e => {
        e.preventDefault();
        testFileDrop.classList.add("file-drop--active");
    });
    testFileDrop.addEventListener("dragleave", () => {
        testFileDrop.classList.remove("file-drop--active");
    });
    testFileDrop.addEventListener("drop", e => {
        e.preventDefault();
        testFileDrop.classList.remove("file-drop--active");
        if (e.dataTransfer.files.length) {
            testFileInput.files = e.dataTransfer.files;
            testFileName.textContent = e.dataTransfer.files[0].name;
            testFileName.classList.add("file-drop__name--show");
        }
    });

    // ── Form submission ─────────────────────────────────────────
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="spinner"></span> Starting…';
        cancelBtn.classList.remove("btn--hidden");
        cancelBtn.disabled = false;
        cancelBtn.innerHTML = svgSquare + " Cancel";
        logEl.innerHTML = "";
        exportBtn.disabled = true;
        exportMenu.classList.add("dropdown__menu--hidden");
        reviewPanel.classList.add("review-panel--hidden");
        resultsPanel.classList.add("results-panel--hidden");
        clearAllSteps();

        const fd = new FormData(form);

        try {
            const res = await fetch("/api/run", { method: "POST", body: fd });
            const data = await res.json();

            if (!res.ok) {
                appendLog(data.error || "Failed to start pipeline", "error");
                runBtn.disabled = false;
                runBtn.innerHTML = svgPlay + " Start Pipeline";
                return;
            }

            currentRunId = data.run_id;
            exportBtn.disabled = false;
            appendLog(`Pipeline started  [run_id: ${data.run_id}]`, "status");
            appendLog(`Output dir: ${data.output_dir}`);
            runBtn.innerHTML = '<span class="spinner"></span> Running…';

            connectSSE(data.run_id);
        } catch (err) {
            appendLog("Network error: " + err.message, "error");
            resetBtn();
        }
    });

    const svgPlay = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    const svgSquare = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>`;

    cancelBtn.addEventListener("click", async () => {
        if (!currentRunId) return;
        if (!confirm("Are you sure you want to cancel the pipeline?")) return;
        cancelBtn.disabled = true;
        cancelBtn.innerHTML = '<span class="spinner"></span> Cancelling…';
        try {
            await fetch(`/api/cancel/${currentRunId}`, { method: "POST" });
        } catch (err) {
            appendLog("Failed to cancel: " + err.message, "error");
            cancelBtn.disabled = false;
            cancelBtn.innerHTML = svgSquare + " Cancel";
        }
    });

    // ── Export functionality ──────────────────────────────────────
    exportBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!currentRunId) return;

        const isHidden = exportMenu.classList.contains("dropdown__menu--hidden");
        if (isHidden) {
            exportMenu.innerHTML = '<div class="dropdown__item dropdown__item--disabled">Loading...</div>';
            exportMenu.classList.remove("dropdown__menu--hidden");

            try {
                const res = await fetch(`/api/artifacts/${currentRunId}`);
                if (!res.ok) throw new Error("Failed to fetch artifacts");
                const data = await res.json();

                exportMenu.innerHTML = "";
                if (data.artifacts && data.artifacts.length > 0) {
                    data.artifacts.forEach(artifact => {
                        const a = document.createElement("a");
                        a.href = `/api/download/${currentRunId}?key=${artifact.key}`;
                        a.className = "dropdown__item";
                        a.download = ""; // Hint to browser it's a download
                        a.textContent = artifact.label;
                        exportMenu.appendChild(a);
                    });
                } else {
                    exportMenu.innerHTML = '<div class="dropdown__item dropdown__item--disabled">No artifacts available yet</div>';
                }
            } catch (err) {
                exportMenu.innerHTML = `<div class="dropdown__item dropdown__item--disabled">Error: ${err.message}</div>`;
            }
        } else {
            exportMenu.classList.add("dropdown__menu--hidden");
        }
    });

    document.addEventListener("click", (e) => {
        if (!exportMenu.contains(e.target) && e.target !== exportBtn) {
            exportMenu.classList.add("dropdown__menu--hidden");
        }
    });

    // ── SSE connection ──────────────────────────────────────────
    function connectSSE(runId) {
        if (evtSource) evtSource.close();
        evtSource = new EventSource(`/api/events/${runId}`);

        evtSource.onmessage = (e) => {
            let evt;
            try { evt = JSON.parse(e.data); } catch { return; }
            handleEvent(evt);
        };

        evtSource.onerror = () => {
            // EventSource will auto-reconnect; if stale, just close
            if (evtSource.readyState === EventSource.CLOSED) {
                appendLog("SSE connection closed.", "status");
            }
        };
    }

    // ── Ordered pipeline steps for accurate tracking ────────────
    const PIPELINE_STEPS = [
        "explore_data", "select_features", "plan_preparation",
        "review_prep_plan", "write_prep_code", "execute_prep_code",
        "verify_prepared_data", "review_prep_data", "write_balance_code",
        "execute_balance_code", "review_balanced_data", "split_data",
        "generate_lmf_config", "review_lmf_config", "run_finetuning",
        "run_prediction", "run_evaluation",
    ];

    let activeStepIndex = -1;

    function handleEvent(evt) {
        switch (evt.event) {
            case "status":
                appendLog(evt.message, "status");
                // Try to detect node name from message
                if (evt.node) {
                    markStepActive(evt.node);
                }
                break;

            case "node_start":
                appendLog(`▸ ${evt.node}`, "node");
                markStepActive(evt.node);
                break;

            case "node_complete":
                appendLog(`✓ ${evt.node} done`, "complete");
                setStepState(evt.node, "done");
                break;

            case "interrupt":
                appendLog("⏸ Review checkpoint reached", "status");
                showReview(evt);
                break;

            case "complete":
                appendLog("✅ " + evt.message, "complete");
                // Mark all remaining steps as done
                PIPELINE_STEPS.forEach(s => {
                    const el = stepsEl.querySelector(`[data-step="${s}"]`);
                    if (el && el.classList.contains("step--active")) {
                        setStepState(s, "done");
                    }
                });
                showResults(evt);
                resetBtn();
                if (evtSource) evtSource.close();
                break;

            case "error":
                appendLog("✗ Error: " + evt.message, "error");
                if (evt.message && evt.message.includes("cancel")) {
                    clearAllSteps();
                    reviewPanel.classList.add("review-panel--hidden");
                }
                resetBtn();
                if (evtSource) evtSource.close();
                break;

            case "heartbeat":
                break;

            case "log":
                if (evt.message) appendLog(evt.message, "log");
                break;

            default:
                if (evt.message) appendLog(evt.message);
        }
    }

    function markStepActive(node) {
        // Mark previous active as done, mark new as active
        const idx = PIPELINE_STEPS.indexOf(node);
        if (idx === -1) return;

        if (idx <= activeStepIndex) {
            // Pipeline backtracked (re-executing after review feedback).
            // Reset all steps from this node onward to pending.
            for (let i = idx; i <= activeStepIndex; i++) {
                setStepState(PIPELINE_STEPS[i], null);
            }
        } else if (activeStepIndex >= 0 && activeStepIndex < PIPELINE_STEPS.length) {
            setStepState(PIPELINE_STEPS[activeStepIndex], "done");
        }

        setStepState(node, "active");
        activeStepIndex = idx;
    }

    // ── Review panel ────────────────────────────────────────────
    const planEditor = document.getElementById("plan-editor");
    const savePlanBtn = document.getElementById("save-plan-btn");
    const reviewLabel = document.getElementById("review-panel-label");
    let isPlanEditMode = false;

    function showReview(evt) {
        const summary = evt.summary || "";
        const node = evt.node || "";

        if (node === "review_prep_plan" && evt.prep_plan) {
            // Plan editing mode
            isPlanEditMode = true;
            reviewLabel.textContent = "Edit Preparation Plan";
            reviewSummary.style.display = "none";
            planEditor.classList.remove("review-panel__editor--hidden");
            // Pretty-print the JSON plan
            try {
                const parsed = JSON.parse(evt.prep_plan);
                planEditor.value = JSON.stringify(parsed, null, 2);
            } catch {
                planEditor.value = evt.prep_plan;
            }
            // Show Save Plan, hide Send Feedback
            savePlanBtn.classList.remove("btn--hidden");
            savePlanBtn.innerHTML = "💾 Save Plan";
            document.getElementById("send-btn").classList.add("btn--hidden");
            reviewInput.style.display = "none";
            planEditor.focus();
        } else if (node === "review_lmf_config" && evt.lmf_train_yaml) {
            // LlamaFactory Config editing mode
            isPlanEditMode = true;
            reviewLabel.textContent = "Edit LlamaFactory Config";
            reviewSummary.style.display = "none";
            planEditor.classList.remove("review-panel__editor--hidden");
            planEditor.value = evt.lmf_train_yaml;
            // Show Save Config, hide Send Feedback
            savePlanBtn.classList.remove("btn--hidden");
            savePlanBtn.innerHTML = "💾 Save Config";
            document.getElementById("send-btn").classList.add("btn--hidden");
            reviewInput.style.display = "none";
            planEditor.focus();
        } else {
            // Normal feedback mode
            isPlanEditMode = false;
            reviewLabel.textContent = "Review Checkpoint";
            reviewSummary.style.display = "";
            reviewSummary.textContent = summary;
            planEditor.classList.add("review-panel__editor--hidden");
            savePlanBtn.classList.add("btn--hidden");
            document.getElementById("send-btn").classList.remove("btn--hidden");
            reviewInput.style.display = "";
            reviewInput.value = "";
            reviewInput.focus();
        }
        reviewPanel.classList.remove("review-panel--hidden");
    }

    function hideReview() {
        reviewPanel.classList.add("review-panel--hidden");
        // Reset to default state
        isPlanEditMode = false;
        reviewSummary.style.display = "";
        planEditor.classList.add("review-panel__editor--hidden");
        savePlanBtn.classList.add("btn--hidden");
        document.getElementById("send-btn").classList.remove("btn--hidden");
        reviewInput.style.display = "";
    }

    async function submitReview(response) {
        if (!currentRunId) return;
        hideReview();
        appendLog(`→ Sent: ${response.length > 120 ? response.slice(0, 120) + "…" : response}`, "status");

        const fd = new FormData();
        fd.append("response", response);

        try {
            await fetch(`/api/review/${currentRunId}`, { method: "POST", body: fd });
        } catch (err) {
            appendLog("Failed to submit review: " + err.message, "error");
        }
    }

    approveBtn.addEventListener("click", () => submitReview("approve"));
    savePlanBtn.addEventListener("click", () => {
        const editedPlan = planEditor.value.trim();
        if (reviewLabel.textContent.includes("Plan")) {
            // Validate JSON
            try {
                JSON.parse(editedPlan);
            } catch {
                alert("The plan must be valid JSON. Please fix any syntax errors.");
                return;
            }
        }
        submitReview(editedPlan);
    });
    reviewForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const val = reviewInput.value.trim() || "approve";
        submitReview(val);
    });

    // ── Results panel ───────────────────────────────────────────
    function showResults(evt) {
        if (!evt.results || evt.results.length === 0) {
            resultsContent.innerHTML = `<p>Pipeline completed. Output: <code>${escapeHtml(evt.run_dir || "—")}</code></p>`;
        } else {
            let html = "";
            for (const r of evt.results) {
                html += `<div class="result-card">`;
                html += `<div class="result-card__split">${escapeHtml(r.split)}</div>`;
                if (r.accuracy != null)
                    html += metric("Accuracy", (r.accuracy * 100).toFixed(2) + "%");
                if (r.f1 != null)
                    html += metric("F1", r.f1.toFixed(4));
                if (r.macro_f1 != null)
                    html += metric("Macro F1", r.macro_f1.toFixed(4));
                if (r.weighted_f1 != null)
                    html += metric("Weighted F1", r.weighted_f1.toFixed(4));
                if (r.valid_predictions != null && r.total_samples != null)
                    html += metric("Valid Predictions", `${r.valid_predictions} / ${r.total_samples}`);
                html += `</div>`;
            }
            if (evt.run_dir) {
                html += `<p style="margin-top:10px;color:var(--text-muted)">Run dir: <code>${escapeHtml(evt.run_dir)}</code></p>`;
            }
            resultsContent.innerHTML = html;
        }
        resultsPanel.classList.remove("results-panel--hidden");
    }

    function metric(label, value) {
        return `<div class="result-card__metric"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>`;
    }

    // ── Reset ───────────────────────────────────────────────────
    function resetBtn() {
        runBtn.disabled = false;
        runBtn.innerHTML = svgPlay + " Start Pipeline";
        cancelBtn.classList.add("btn--hidden");
    }

    // ── Auto-reconnect on page load ────────────────────────────
    (async function checkActiveRun() {
        try {
            const res = await fetch("/api/runs/active");
            const data = await res.json();
            if (data.run_id) {
                currentRunId = data.run_id;
                appendLog(`Reconnected to active run [${data.run_id}] (${data.status})`, "status");
                runBtn.disabled = true;
                runBtn.innerHTML = '<span class="spinner"></span> Running…';
                cancelBtn.classList.remove("btn--hidden");
                cancelBtn.disabled = false;
                cancelBtn.innerHTML = svgSquare + " Cancel";
                exportBtn.disabled = false;
                connectSSE(data.run_id);
            }
        } catch (err) {
            // No active run or server unreachable — stay on config page
        }
    })();
})();
