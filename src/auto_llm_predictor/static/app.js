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
    function setupFileDrop(input, dropZone, nameEl) {
        input.addEventListener("change", () => {
            if (input.files.length) {
                nameEl.textContent = input.files[0].name;
                nameEl.classList.add("file-drop__name--show");
            }
        });
        dropZone.addEventListener("dragover", e => {
            e.preventDefault();
            dropZone.classList.add("file-drop--active");
        });
        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("file-drop--active");
        });
        dropZone.addEventListener("drop", e => {
            e.preventDefault();
            dropZone.classList.remove("file-drop--active");
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                nameEl.textContent = e.dataTransfer.files[0].name;
                nameEl.classList.add("file-drop__name--show");
            }
        });
    }

    setupFileDrop(fileInput, fileDrop, fileName);
    setupFileDrop(
        document.getElementById("test-csv-file"),
        document.getElementById("test-file-drop"),
        document.getElementById("test-file-name"),
    );

    // ── LLM provider toggle ─────────────────────────────────────
    const providerSelect = document.getElementById("llm-provider");
    const apiBaseInput = document.getElementById("agent-api-base");
    const apiKeyInput = document.getElementById("agent-api-key");
    const apiKeyLabel = apiKeyInput ? apiKeyInput.previousElementSibling : null;

    if (providerSelect) {
        providerSelect.addEventListener("change", () => {
            const isOllama = providerSelect.value === "ollama";
            if (apiBaseInput) {
                apiBaseInput.placeholder = isOllama ? "http://localhost:11434" : "from .env";
            }
            if (apiKeyInput) {
                apiKeyInput.style.display = isOllama ? "none" : "";
            }
            if (apiKeyLabel) {
                apiKeyLabel.style.display = isOllama ? "none" : "";
            }
        });
    }

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
                exportMenu.innerHTML = `<div class="dropdown__item dropdown__item--disabled">Error: ${escapeHtml(err.message)}</div>`;
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
    let sseRetries = 0;
    const SSE_MAX_RETRIES = 5;

    function connectSSE(runId) {
        if (evtSource) evtSource.close();
        sseRetries = 0;
        evtSource = new EventSource(`/api/events/${runId}`);

        evtSource.onmessage = (e) => {
            sseRetries = 0; // reset on successful message
            let evt;
            try { evt = JSON.parse(e.data); } catch { return; }
            handleEvent(evt);
        };

        evtSource.onerror = () => {
            if (evtSource.readyState === EventSource.CLOSED) {
                appendLog("SSE connection closed.", "status");
            } else {
                sseRetries++;
                if (sseRetries >= SSE_MAX_RETRIES) {
                    evtSource.close();
                    appendLog("SSE connection lost after " + SSE_MAX_RETRIES + " retries.", "error");
                }
            }
        };
    }

    // ── Ordered pipeline steps for accurate tracking ────────────
    const PIPELINE_STEPS = [
        "explore_data", "select_features", "plan_preparation",
        "review_prep_plan", "write_prep_code", "execute_prep_code",
        "verify_prepared_data", "review_prep_data", "write_balance_code",
        "execute_balance_code", "review_balanced_data", "split_data",
        "determine_cutoff_len", "generate_lmf_config", "review_lmf_config",
        "run_finetuning", "run_prediction", "run_evaluation",
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

            case "eval_results":
                appendLog("📊 Evaluation complete — results available", "complete");
                showResults(evt);
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

    function enterEditorMode(label, content, saveLabel) {
        isPlanEditMode = true;
        reviewLabel.textContent = label;
        reviewSummary.style.display = "none";
        planEditor.classList.remove("review-panel__editor--hidden");
        planEditor.value = content;
        savePlanBtn.classList.remove("btn--hidden");
        savePlanBtn.innerHTML = saveLabel;
        document.getElementById("send-btn").classList.add("btn--hidden");
        reviewInput.style.display = "none";
        planEditor.focus();
    }

    function showReview(evt) {
        const summary = evt.summary || "";
        const node = evt.node || "";

        if (node === "review_prep_plan" && evt.prep_plan) {
            let content = evt.prep_plan;
            try {
                content = JSON.stringify(JSON.parse(content), null, 2);
            } catch { /* keep raw */ }
            enterEditorMode("Edit Preparation Plan", content, "💾 Save Plan");
        } else if (node === "review_lmf_config" && evt.lmf_train_yaml) {
            enterEditorMode("Edit LlamaFactory Config", evt.lmf_train_yaml, "💾 Save Config");
        } else {
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

    // =========================================================
    // INFERENCE TAB
    // =========================================================

    // ── Tab switching ─────────────────────────────────────────
    const tabBar = document.getElementById("tab-bar");
    if (tabBar) {
        tabBar.addEventListener("click", function (e) {
            const btn = e.target.closest(".tab");
            if (!btn) return;
            const tabName = btn.dataset.tab;

            // Update tab buttons
            tabBar.querySelectorAll(".tab").forEach(function (t) {
                t.classList.remove("tab--active");
            });
            btn.classList.add("tab--active");

            // Update tab content
            document.querySelectorAll(".tab-content").forEach(function (tc) {
                tc.classList.remove("tab-content--active");
            });
            const target = document.getElementById("tab-" + tabName);
            if (target) target.classList.add("tab-content--active");
        });
    }

    // ── Inference mode toggle (batch / single) ───────────────
    const modeBatchBtn = document.getElementById("mode-batch-btn");
    const modeSingleBtn = document.getElementById("mode-single-btn");
    const inferBatchPanel = document.getElementById("infer-batch-panel");
    const inferSinglePanel = document.getElementById("infer-single-panel");

    function setInferMode(mode) {
        if (mode === "batch") {
            modeBatchBtn.classList.add("infer-mode-btn--active");
            modeSingleBtn.classList.remove("infer-mode-btn--active");
            inferBatchPanel.classList.remove("infer-panel--hidden");
            inferSinglePanel.classList.add("infer-panel--hidden");
        } else {
            modeSingleBtn.classList.add("infer-mode-btn--active");
            modeBatchBtn.classList.remove("infer-mode-btn--active");
            inferSinglePanel.classList.remove("infer-panel--hidden");
            inferBatchPanel.classList.add("infer-panel--hidden");
        }
    }

    if (modeBatchBtn) modeBatchBtn.addEventListener("click", function () { setInferMode("batch"); });
    if (modeSingleBtn) modeSingleBtn.addEventListener("click", function () { setInferMode("single"); });

    // ── Inference file drop  ─────────────────────────────────
    const inferFileDrop = document.getElementById("infer-file-drop");
    const inferCsvFile = document.getElementById("infer-csv-file");
    const inferFileName = document.getElementById("infer-file-name");

    if (inferFileDrop && inferCsvFile) {
        inferFileDrop.addEventListener("click", function () { inferCsvFile.click(); });
        inferCsvFile.addEventListener("change", function () {
            if (inferCsvFile.files.length) {
                inferFileName.textContent = inferCsvFile.files[0].name;
                inferFileDrop.classList.add("file-drop--has-file");
            }
        });
        inferFileDrop.addEventListener("dragover", function (e) { e.preventDefault(); inferFileDrop.classList.add("file-drop--drag"); });
        inferFileDrop.addEventListener("dragleave", function () { inferFileDrop.classList.remove("file-drop--drag"); });
        inferFileDrop.addEventListener("drop", function (e) {
            e.preventDefault();
            inferFileDrop.classList.remove("file-drop--drag");
            if (e.dataTransfer.files.length) {
                inferCsvFile.files = e.dataTransfer.files;
                inferFileName.textContent = e.dataTransfer.files[0].name;
                inferFileDrop.classList.add("file-drop--has-file");
            }
        });
    }

    // ── Inference log helper ─────────────────────────────────
    const inferLog = document.getElementById("infer-log");

    function appendInferLog(msg, cls) {
        if (!inferLog) return;
        const line = document.createElement("div");
        line.className = "log__line" + (cls ? " log__line--" + cls : "");
        line.textContent = msg;
        inferLog.appendChild(line);
        inferLog.scrollTop = inferLog.scrollHeight;
    }

    // ── Batch inference ──────────────────────────────────────
    const inferBatchForm = document.getElementById("infer-batch-form");
    const inferBatchBtn = document.getElementById("infer-batch-btn");
    const inferResultsPanel = document.getElementById("infer-results-panel");
    const inferResultsContent = document.getElementById("infer-results-content");

    if (inferBatchForm) {
        inferBatchForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const outputDir = document.getElementById("infer-output-dir").value.trim();
            const runDir = document.getElementById("infer-run-dir").value.trim();

            if (!outputDir || !runDir) {
                alert("Please fill in both Training Output Directory and Run Directory.");
                return;
            }

            if (!inferCsvFile.files.length) {
                alert("Please select a CSV file for inference.");
                return;
            }

            // Clear previous
            inferLog.innerHTML = "";
            inferResultsPanel.classList.add("results-panel--hidden");
            inferBatchBtn.disabled = true;
            inferBatchBtn.innerHTML = '<span class="spinner"></span> Running…';

            const fd = new FormData();
            fd.append("csv_file", inferCsvFile.files[0]);
            fd.append("output_dir", outputDir);
            fd.append("run_dir", runDir);
            fd.append("precision", document.getElementById("infer-precision").value);
            fd.append("flash_attn", document.getElementById("infer-flash-attn").value);
            const qbitBatch = document.getElementById("infer-quantization-bit").value;
            if (qbitBatch) fd.append("quantization_bit", qbitBatch);
            const batchXai = document.getElementById("infer-batch-xai");
            if (batchXai && batchXai.checked) fd.append("xai", "true");

            try {
                const res = await fetch("/api/infer/batch", { method: "POST", body: fd });
                const data = await res.json();

                if (!res.ok) {
                    appendInferLog("Error: " + (data.error || "Unknown error"), "error");
                    inferBatchBtn.disabled = false;
                    inferBatchBtn.textContent = "Run Batch Inference";
                    return;
                }

                const runId = data.run_id;
                appendInferLog("Batch inference started (run " + runId + ")", "status");

                // Connect SSE for batch inference
                const evtSource = new EventSource("/api/events/" + runId);
                evtSource.onmessage = function (ev) {
                    const evt = JSON.parse(ev.data);
                    if (evt.event === "log") {
                        appendInferLog(evt.message || "");
                    } else if (evt.event === "status") {
                        appendInferLog(evt.message || "", "status");
                    } else if (evt.event === "complete") {
                        appendInferLog("✓ " + (evt.message || "Complete!"), "status");
                        inferResultsPanel.classList.remove("results-panel--hidden");
                        let batchHtml =
                            '<p><strong>Samples:</strong> ' + (evt.num_samples || 0) + '</p>' +
                            '<p><strong>Output:</strong> ' + (evt.infer_output || '') + '</p>' +
                            '<a href="/api/infer/download/' + runId + '" class="btn btn--sm btn--secondary" download>⬇ Download Predictions</a>';
                        if (evt.xai_results && evt.xai_results.length) {
                            batchHtml += '<hr style="margin:16px 0">';
                            batchHtml += '<h4>XAI Results</h4>';
                            batchHtml += renderXaiResults(evt.xai_results);
                            if (evt.xai_report_path) {
                                batchHtml += '<p style="margin-top:8px"><strong>XAI Report:</strong> ' +
                                    escapeHtml(evt.xai_report_path) + '</p>';
                            }
                        }
                        inferResultsContent.innerHTML = batchHtml;
                        inferBatchBtn.disabled = false;
                        inferBatchBtn.textContent = "Run Batch Inference";
                        evtSource.close();
                    } else if (evt.event === "error") {
                        appendInferLog("✗ Error: " + (evt.message || ""), "error");
                        inferBatchBtn.disabled = false;
                        inferBatchBtn.textContent = "Run Batch Inference";
                        evtSource.close();
                    }
                };
                evtSource.onerror = function () {
                    appendInferLog("SSE connection lost", "error");
                    inferBatchBtn.disabled = false;
                    inferBatchBtn.textContent = "Run Batch Inference";
                    evtSource.close();
                };
            } catch (err) {
                appendInferLog("Network error: " + err.message, "error");
                inferBatchBtn.disabled = false;
                inferBatchBtn.textContent = "Run Batch Inference";
            }
        });
    }

    // ── Single inference — load features ─────────────────────
    const loadFeaturesBtn = document.getElementById("load-features-btn");
    const featureInputsDiv = document.getElementById("feature-inputs");
    const inferSingleBtn = document.getElementById("infer-single-btn");
    const singleResultPanel = document.getElementById("single-result-panel");
    const singleResultContent = document.getElementById("single-result-content");

    // Auto-fill DOM refs
    const csvAutofillSection = document.getElementById("csv-autofill-section");
    const singleCsvDrop = document.getElementById("single-csv-drop");
    const singleCsvFile = document.getElementById("single-csv-file");
    const singleCsvName = document.getElementById("single-csv-name");
    const csvRowSelector = document.getElementById("csv-row-selector");
    const csvRowIndex = document.getElementById("csv-row-index");
    const csvRowCount = document.getElementById("csv-row-count");
    const csvRowLabel = document.getElementById("csv-row-label");
    const csvMissingNotice = document.getElementById("csv-missing-notice");
    const csvFeaturesList = document.getElementById("csv-features-list");
    const loadTrainingCsvBtn = document.getElementById("load-training-csv-btn");
    const trainingCsvStatus = document.getElementById("training-csv-status");
    const loadProcessedBtn = document.getElementById("load-processed-btn");
    const processedDataset = document.getElementById("processed-dataset");
    const processedStatus = document.getElementById("processed-status");

    let loadedFeatures = [];
    let parsedCsvRows = [];
    let currentTrueLabel = null;

    // ── Data source toggle ────────────────────────────────────
    const dataSrcBtns = document.querySelectorAll(".data-src-btn");
    const srcPanels = {
        "upload": document.getElementById("src-upload-panel"),
        "training-csv": document.getElementById("src-training-panel"),
        "processed": document.getElementById("src-processed-panel"),
    };

    function setDataSource(source) {
        for (const btn of dataSrcBtns) {
            btn.classList.toggle("data-src-btn--active", btn.dataset.source === source);
        }
        for (const [key, panel] of Object.entries(srcPanels)) {
            if (panel) panel.classList.toggle("src-panel--hidden", key !== source);
        }
    }

    for (const btn of dataSrcBtns) {
        btn.addEventListener("click", function () { setDataSource(btn.dataset.source); });
    }

    // ── Helpers: show rows in the shared row selector ─────────
    function resetRowSelector() {
        parsedCsvRows = [];
        currentTrueLabel = null;
        if (csvRowSelector) csvRowSelector.classList.add("csv-row-selector--hidden");
        if (csvRowLabel) csvRowLabel.classList.add("csv-row-label--hidden");
        if (csvMissingNotice) csvMissingNotice.classList.add("csv-missing-notice--hidden");
        if (csvFeaturesList) csvFeaturesList.classList.add("csv-features-list--hidden");
    }

    function showRowSelector(data) {
        parsedCsvRows = data.rows || [];
        if (!parsedCsvRows.length) return;

        csvRowIndex.max = parsedCsvRows.length - 1;
        csvRowIndex.value = 0;
        csvRowCount.textContent = "of " + parsedCsvRows.length + " rows" +
            (data.truncated ? " (truncated)" : "");
        csvRowSelector.classList.remove("csv-row-selector--hidden");

        // Missing features notice
        if (data.missing_features && data.missing_features.length > 0) {
            csvMissingNotice.textContent = "Features not in source: " + data.missing_features.join(", ");
            csvMissingNotice.classList.remove("csv-missing-notice--hidden");
        } else {
            csvMissingNotice.classList.add("csv-missing-notice--hidden");
        }

        // Selected features list
        if (data.selected_features && data.selected_features.length > 0) {
            csvFeaturesList.textContent = "Selected features: " + data.selected_features.join(", ");
            csvFeaturesList.classList.remove("csv-features-list--hidden");
        } else {
            csvFeaturesList.classList.add("csv-features-list--hidden");
        }

        fillFeaturesFromRow(0);
    }

    function fillFeaturesFromRow(idx) {
        if (idx < 0 || idx >= parsedCsvRows.length) return;
        var rowData = parsedCsvRows[idx];

        var inputs = featureInputsDiv.querySelectorAll(".feature-value");
        for (var i = 0; i < inputs.length; i++) {
            var feature = inputs[i].dataset.feature;
            if (feature in rowData) {
                inputs[i].value = rowData[feature];
            }
        }

        // Track true label for display next to prediction
        currentTrueLabel = ("__label__" in rowData) ? rowData["__label__"] : null;

        // Show label for rows with known output
        if ("__label__" in rowData && csvRowLabel) {
            csvRowLabel.textContent = "True label: " + rowData["__label__"];
            csvRowLabel.classList.remove("csv-row-label--hidden");
        } else if (csvRowLabel) {
            csvRowLabel.classList.add("csv-row-label--hidden");
        }
    }

    if (csvRowIndex) {
        csvRowIndex.addEventListener("input", function () {
            var idx = parseInt(csvRowIndex.value, 10);
            if (!isNaN(idx) && idx >= 0 && idx < parsedCsvRows.length) {
                fillFeaturesFromRow(idx);
            }
        });
    }

    // ── Upload CSV file drop ──────────────────────────────────
    if (singleCsvDrop && singleCsvFile) {
        singleCsvDrop.addEventListener("click", function (e) {
            if (e.target === singleCsvFile) return;
            singleCsvFile.click();
        });
        singleCsvDrop.addEventListener("dragover", function (e) {
            e.preventDefault();
            singleCsvDrop.classList.add("file-drop--drag");
        });
        singleCsvDrop.addEventListener("dragleave", function () {
            singleCsvDrop.classList.remove("file-drop--drag");
        });
        singleCsvDrop.addEventListener("drop", function (e) {
            e.preventDefault();
            singleCsvDrop.classList.remove("file-drop--drag");
            if (e.dataTransfer.files.length) {
                singleCsvFile.files = e.dataTransfer.files;
                handleSingleCsvUpload();
            }
        });
        singleCsvFile.addEventListener("change", function () {
            if (singleCsvFile.files.length) handleSingleCsvUpload();
        });
    }

    async function handleSingleCsvUpload() {
        var file = singleCsvFile.files[0];
        if (!file) return;

        var outputDir = document.getElementById("infer-output-dir").value.trim();
        if (!outputDir) {
            alert("Please enter the Training Output Directory first.");
            return;
        }

        singleCsvName.textContent = file.name + " (parsing…)";
        singleCsvName.classList.add("file-drop__name--show");
        resetRowSelector();

        var fd = new FormData();
        fd.append("csv_file", file);
        fd.append("output_dir", outputDir);

        try {
            var res = await fetch("/api/infer/parse-csv", { method: "POST", body: fd });
            var data = await res.json();

            if (!res.ok) {
                alert("CSV parse error: " + (data.error || "Unknown"));
                singleCsvName.textContent = file.name + " (error)";
                return;
            }

            singleCsvName.textContent = file.name;
            if (!data.rows || !data.rows.length) {
                singleCsvName.textContent = file.name + " (no matching rows)";
                return;
            }

            showRowSelector(data);
        } catch (err) {
            alert("Network error: " + err.message);
            singleCsvName.textContent = file.name + " (error)";
        }
    }

    // ── Training CSV load ─────────────────────────────────────
    if (loadTrainingCsvBtn) {
        loadTrainingCsvBtn.addEventListener("click", async function () {
            var outputDir = document.getElementById("infer-output-dir").value.trim();
            if (!outputDir) { alert("Please enter the Training Output Directory first."); return; }

            loadTrainingCsvBtn.disabled = true;
            trainingCsvStatus.textContent = "Loading…";
            resetRowSelector();

            try {
                var res = await fetch("/api/infer/training-rows?output_dir=" + encodeURIComponent(outputDir) + "&source=csv");
                var data = await res.json();

                if (!res.ok) {
                    alert("Error: " + (data.error || "Unknown"));
                    trainingCsvStatus.textContent = "Error";
                    return;
                }

                trainingCsvStatus.textContent = data.total_rows + " rows loaded";
                if (data.rows && data.rows.length) {
                    showRowSelector(data);
                } else {
                    trainingCsvStatus.textContent = "No matching rows";
                }
            } catch (err) {
                alert("Network error: " + err.message);
                trainingCsvStatus.textContent = "Error";
            } finally {
                loadTrainingCsvBtn.disabled = false;
            }
        });
    }

    // ── Processed data load ───────────────────────────────────
    var processedDatasetsLoaded = false;

    async function loadProcessedData(dataset) {
        var outputDir = document.getElementById("infer-output-dir").value.trim();
        if (!outputDir) { alert("Please enter the Training Output Directory first."); return; }

        if (loadProcessedBtn) loadProcessedBtn.disabled = true;
        if (processedStatus) processedStatus.textContent = "Loading…";
        resetRowSelector();

        var url = "/api/infer/training-rows?output_dir=" + encodeURIComponent(outputDir) + "&source=jsonl";
        if (dataset) url += "&dataset=" + encodeURIComponent(dataset);

        try {
            var res = await fetch(url);
            var data = await res.json();

            if (!res.ok) {
                alert("Error: " + (data.error || "Unknown"));
                if (processedStatus) processedStatus.textContent = "Error";
                return;
            }

            // Populate dataset dropdown if available
            if (data.available_datasets && processedDataset) {
                processedDataset.innerHTML = "";
                for (var i = 0; i < data.available_datasets.length; i++) {
                    var ds = data.available_datasets[i];
                    var opt = document.createElement("option");
                    opt.value = ds.name;
                    opt.textContent = ds.name + " (" + ds.count + " rows)";
                    if (ds.name === data.dataset) opt.selected = true;
                    processedDataset.appendChild(opt);
                }
                processedDatasetsLoaded = true;
            }

            if (processedStatus) processedStatus.textContent = data.total_rows + " rows loaded";
            if (data.rows && data.rows.length) {
                showRowSelector(data);
            } else {
                if (processedStatus) processedStatus.textContent = "No rows";
            }
        } catch (err) {
            alert("Network error: " + err.message);
            if (processedStatus) processedStatus.textContent = "Error";
        } finally {
            if (loadProcessedBtn) loadProcessedBtn.disabled = false;
        }
    }

    if (loadProcessedBtn) {
        loadProcessedBtn.addEventListener("click", function () {
            var ds = processedDataset ? processedDataset.value : "";
            loadProcessedData(ds);
        });
    }

    if (processedDataset) {
        processedDataset.addEventListener("change", function () {
            if (processedDatasetsLoaded && processedDataset.value) {
                loadProcessedData(processedDataset.value);
            }
        });
    }

    // ── Load features ─────────────────────────────────────────
    if (loadFeaturesBtn) {
        loadFeaturesBtn.addEventListener("click", async function () {
            const outputDir = document.getElementById("infer-output-dir").value.trim();
            if (!outputDir) {
                alert("Please enter the Training Output Directory first.");
                return;
            }

            loadFeaturesBtn.disabled = true;
            loadFeaturesBtn.textContent = "Loading…";

            try {
                const res = await fetch("/api/infer/features?output_dir=" + encodeURIComponent(outputDir));
                const data = await res.json();

                if (!res.ok) {
                    alert("Error: " + (data.error || "Unknown"));
                    return;
                }

                loadedFeatures = data.features || [];
                if (!loadedFeatures.length) {
                    featureInputsDiv.innerHTML = '<p class="form__hint">No features found in pipeline state.</p>';
                    return;
                }

                // Build feature input fields
                let html = "";
                for (const feat of loadedFeatures) {
                    html += '<div class="feature-input-row">';
                    html += '<label class="form__label">' + escapeHtml(feat) + '</label>';
                    html += '<input type="text" class="form__input feature-value" data-feature="' + escapeHtml(feat) + '" placeholder="Enter value">';
                    html += '</div>';
                }
                featureInputsDiv.innerHTML = html;
                inferSingleBtn.disabled = false;

                // Show auto-fill section and reset previous state
                if (csvAutofillSection) {
                    csvAutofillSection.classList.remove("csv-autofill-section--hidden");
                }
                resetRowSelector();
                processedDatasetsLoaded = false;
                if (processedDataset) processedDataset.innerHTML = '<option value="">-- select dataset --</option>';
                if (singleCsvName) { singleCsvName.textContent = ""; singleCsvName.classList.remove("file-drop__name--show"); }
                if (trainingCsvStatus) trainingCsvStatus.textContent = "";
                if (processedStatus) processedStatus.textContent = "";
            } catch (err) {
                alert("Network error: " + err.message);
            } finally {
                loadFeaturesBtn.disabled = false;
                loadFeaturesBtn.textContent = "↻ Load Features";
            }
        });
    }

    // ── Single inference — render result helpers ───────────────
    function renderPredictionOnly(data) {
        let html = '<div class="single-prediction">';
        html += '<p class="prediction-label">Prediction</p>';
        html += '<p class="prediction-value">' + escapeHtml(data.prediction || "—") + '</p>';

        if (data.true_label != null && data.true_label !== "") {
            var match = String(data.prediction).trim() === String(data.true_label).trim();
            html += '<p class="prediction-true-label">';
            html += 'True label: <strong>' + escapeHtml(String(data.true_label)) + '</strong> ';
            html += match
                ? '<span class="label-match label-match--correct">Match</span>'
                : '<span class="label-match label-match--mismatch">Mismatch</span>';
            html += '</p>';
        }

        if (data.target_mapping && Object.keys(data.target_mapping).length) {
            html += '<p class="prediction-mapping"><small>Target mapping: ' +
                JSON.stringify(data.target_mapping) + '</small></p>';
        }

        html += '</div>';
        return html;
    }

    function renderXaiResults(xaiResults) {
        if (!xaiResults || !xaiResults.length) return "";
        let html = "";
        for (const xr of xaiResults) {
            const method = (xr.method || "unknown").toUpperCase();
            html += '<div class="xai-result">';
            html += '<h4 class="xai-method">XAI — ' + method + '</h4>';

            // Prefer embedded SHAP HTML visualisation when available
            if (xr.html) {
                // Inject a small resize script so the iframe fits its content
                const resizeScript = '<script>new ResizeObserver(()=>{' +
                    'frameElement.style.height=document.documentElement.scrollHeight+"px"' +
                    '}).observe(document.documentElement)<\/script>';
                const framedHtml = xr.html + resizeScript;
                html += '<iframe class="xai-shap-frame" srcdoc="' +
                    framedHtml.replace(/&/g, '&amp;').replace(/"/g, '&quot;') +
                    '" sandbox="allow-scripts allow-same-origin" frameborder="0"></iframe>';
            } else {
                const explanations = xr.sample_explanations || [];
                if (explanations.length && explanations[0].token_scores) {
                    html += '<table class="xai-tokens"><thead><tr><th>Token</th><th>Score</th></tr></thead><tbody>';
                    const topScores = explanations[0].token_scores.slice(0, 15);
                    const maxScore = Math.max(...topScores.map(ts => Math.abs(ts.score)), 1e-9);
                    for (const ts of topScores) {
                        const barWidth = Math.round(Math.abs(ts.score) / maxScore * 100);
                        html += '<tr><td class="xai-token">' + escapeHtml(ts.token) + '</td>';
                        html += '<td class="xai-score"><div class="xai-bar" style="width:' + barWidth + '%"></div>';
                        html += '<span>' + ts.score.toFixed(6) + '</span></td></tr>';
                    }
                    html += '</tbody></table>';
                }
            }
            html += '</div>';
        }
        return html;
    }

    function renderSingleResult(data) {
        return renderPredictionOnly(data) + renderXaiResults(data.xai_results);
    }

    // ── Single inference — predict ───────────────────────────
    if (inferSingleBtn) {
        inferSingleBtn.addEventListener("click", async function () {
            const outputDir = document.getElementById("infer-output-dir").value.trim();
            const runDir = document.getElementById("infer-run-dir").value.trim();

            if (!outputDir || !runDir) {
                alert("Please fill in both directories.");
                return;
            }

            // Collect feature values
            const features = {};
            const inputs = featureInputsDiv.querySelectorAll(".feature-value");
            for (const inp of inputs) {
                features[inp.dataset.feature] = inp.value.trim();
            }

            const xaiEnabled = document.getElementById("infer-xai").checked;

            inferSingleBtn.disabled = true;
            inferSingleBtn.innerHTML = '<span class="spinner"></span> Predicting…';
            singleResultPanel.classList.add("results-panel--hidden");
            appendInferLog("Running single inference…", "status");

            // Capture true label at prediction time (may be null)
            const trueLabel = currentTrueLabel;

            const fd = new FormData();
            fd.append("output_dir", outputDir);
            fd.append("run_dir", runDir);
            fd.append("features_json", JSON.stringify(features));
            fd.append("xai", xaiEnabled ? "true" : "false");
            fd.append("precision", document.getElementById("infer-precision").value);
            const qbitSingle = document.getElementById("infer-quantization-bit").value;
            if (qbitSingle) fd.append("quantization_bit", qbitSingle);

            try {
                const res = await fetch("/api/infer/single", { method: "POST", body: fd });
                const data = await res.json();

                if (!res.ok) {
                    appendInferLog("Error: " + (data.error || "Unknown"), "error");
                    inferSingleBtn.disabled = false;
                    inferSingleBtn.innerHTML = '🔍 Predict';
                    return;
                }

                // XAI mode: background thread with SSE streaming
                if (data.run_id) {
                    const runId = data.run_id;
                    appendInferLog("Single inference with XAI started (run " + runId + ")", "status");

                    const evtSource = new EventSource("/api/events/" + runId);
                    evtSource.onmessage = function (ev) {
                        const evt = JSON.parse(ev.data);
                        if (evt.event === "log") {
                            appendInferLog(evt.message || "");
                        } else if (evt.event === "status") {
                            appendInferLog(evt.message || "", "status");
                        } else if (evt.event === "single_prediction") {
                            // Prediction is ready — show it immediately
                            appendInferLog("✓ Prediction: " + (evt.prediction || ""), "status");
                            evt.true_label = trueLabel;
                            singleResultContent.innerHTML = renderPredictionOnly(evt);
                            singleResultPanel.classList.remove("results-panel--hidden");
                        } else if (evt.event === "single_xai") {
                            // XAI results arrived — append them below prediction
                            appendInferLog("✓ XAI analysis complete", "status");
                            singleResultContent.innerHTML += renderXaiResults(evt.xai_results);
                        } else if (evt.event === "single_complete") {
                            // All done — if prediction wasn't shown yet (e.g. replayed events), render everything
                            if (singleResultPanel.classList.contains("results-panel--hidden")) {
                                appendInferLog("✓ Prediction: " + (evt.prediction || ""), "status");
                                evt.true_label = trueLabel;
                                singleResultContent.innerHTML = renderSingleResult(evt);
                                singleResultPanel.classList.remove("results-panel--hidden");
                            }
                            inferSingleBtn.disabled = false;
                            inferSingleBtn.innerHTML = '🔍 Predict';
                            evtSource.close();
                        } else if (evt.event === "error") {
                            appendInferLog("Error: " + (evt.message || "Unknown"), "error");
                            inferSingleBtn.disabled = false;
                            inferSingleBtn.innerHTML = '🔍 Predict';
                            evtSource.close();
                        }
                    };
                    evtSource.onerror = function () {
                        appendInferLog("SSE connection lost", "error");
                        inferSingleBtn.disabled = false;
                        inferSingleBtn.innerHTML = '🔍 Predict';
                        evtSource.close();
                    };
                    return;  // Don't reset button — SSE handlers will do it
                }

                // Non-XAI mode: synchronous result
                appendInferLog("✓ Prediction: " + data.prediction, "status");
                data.true_label = trueLabel;
                singleResultContent.innerHTML = renderSingleResult(data);
                singleResultPanel.classList.remove("results-panel--hidden");
            } catch (err) {
                appendInferLog("Network error: " + err.message, "error");
            } finally {
                if (!xaiEnabled) {
                    inferSingleBtn.disabled = false;
                    inferSingleBtn.innerHTML = '🔍 Predict';
                }
            }
        });
    }

    // ══════════════════════════════════════════════════════════
    // STANDALONE XAI TAB
    // ══════════════════════════════════════════════════════════

    const xaiForm = document.getElementById("xai-form");
    const xaiRunBtn = document.getElementById("xai-run-btn");
    const xaiLog = document.getElementById("xai-log");
    const xaiResultsPanel = document.getElementById("xai-results-panel");
    const xaiResultsContent = document.getElementById("xai-results-content");

    function appendXaiLog(msg, cls) {
        if (!xaiLog) return;
        const line = document.createElement("div");
        line.className = "log__line" + (cls ? " log__line--" + cls : "");
        line.textContent = msg;
        xaiLog.appendChild(line);
        xaiLog.scrollTop = xaiLog.scrollHeight;
    }

    let xaiEvtSource = null;
    let xaiSseRetries = 0;
    const XAI_SSE_MAX_RETRIES = 5;

    function resetXaiBtn() {
        if (!xaiRunBtn) return;
        xaiRunBtn.disabled = false;
        xaiRunBtn.textContent = "Run XAI Analysis";
    }

    function connectXaiSSE(runId) {
        if (xaiEvtSource) xaiEvtSource.close();
        xaiSseRetries = 0;
        xaiEvtSource = new EventSource("/api/events/" + runId);

        xaiEvtSource.onmessage = function (ev) {
            xaiSseRetries = 0;
            let evt;
            try { evt = JSON.parse(ev.data); } catch { return; }

            if (evt.event === "log") {
                appendXaiLog(evt.message || "");
            } else if (evt.event === "status") {
                appendXaiLog(evt.message || "", "status");
            } else if (evt.event === "xai_complete") {
                appendXaiLog("✓ " + (evt.message || "Complete!"), "status");
                xaiResultsPanel.classList.remove("results-panel--hidden");
                let html =
                    '<p><strong>Samples:</strong> ' + (evt.num_samples || 0) + '</p>' +
                    '<p><strong>Methods:</strong> ' + (evt.methods_succeeded || []).join(", ") + '</p>';
                if (evt.xai_results && evt.xai_results.length) {
                    html += '<hr style="margin:16px 0">';
                    html += renderXaiResults(evt.xai_results);
                }
                if (evt.xai_report_path) {
                    html += '<p style="margin-top:8px"><strong>Report:</strong> ' +
                        escapeHtml(evt.xai_report_path) + '</p>';
                    html += '<a href="/api/xai/download/' + runId +
                        '" class="btn btn--sm btn--secondary" download>⬇ Download XAI Report</a>';
                }
                xaiResultsContent.innerHTML = html;
                resetXaiBtn();
                xaiEvtSource.close();
            } else if (evt.event === "error") {
                appendXaiLog("✗ Error: " + (evt.message || ""), "error");
                resetXaiBtn();
                xaiEvtSource.close();
            }
        };

        xaiEvtSource.onerror = function () {
            if (xaiEvtSource.readyState === EventSource.CLOSED) {
                appendXaiLog("SSE connection closed.", "status");
            } else {
                xaiSseRetries++;
                if (xaiSseRetries >= XAI_SSE_MAX_RETRIES) {
                    xaiEvtSource.close();
                    appendXaiLog("SSE connection lost after " + XAI_SSE_MAX_RETRIES + " retries.", "error");
                    resetXaiBtn();
                }
            }
        };
    }

    if (xaiForm) {
        xaiForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const outputDir = document.getElementById("xai-output-dir").value.trim();
            const runDir = document.getElementById("xai-run-dir").value.trim();

            if (!outputDir || !runDir) {
                alert("Please fill in both Training Output Directory and Run Directory.");
                return;
            }

            // Clear previous
            xaiLog.innerHTML = "";
            xaiResultsPanel.classList.add("results-panel--hidden");
            xaiRunBtn.disabled = true;
            xaiRunBtn.innerHTML = '<span class="spinner"></span> Running…';

            const fd = new FormData();
            fd.append("output_dir", outputDir);
            fd.append("run_dir", runDir);
            fd.append("max_samples", document.getElementById("xai-max-samples").value);
            fd.append("precision", document.getElementById("xai-precision").value);
            const qbit = document.getElementById("xai-quantization-bit").value;
            if (qbit) fd.append("quantization_bit", qbit);

            try {
                const res = await fetch("/api/xai/run", { method: "POST", body: fd });
                const data = await res.json();

                if (!res.ok) {
                    appendXaiLog("Error: " + (data.error || "Unknown error"), "error");
                    resetXaiBtn();
                    return;
                }

                const runId = data.run_id;
                appendXaiLog("XAI analysis started (run " + runId + ")", "status");
                connectXaiSSE(runId);
            } catch (err) {
                appendXaiLog("Network error: " + err.message, "error");
                resetXaiBtn();
            }
        });
    }

    // ── XAI: Auto-reconnect on page load ─────────────────────
    (async function checkActiveXaiRun() {
        try {
            const res = await fetch("/api/xai/active");
            const data = await res.json();
            if (data.run_id) {
                appendXaiLog("Reconnected to active XAI run [" + data.run_id + "] (" + data.status + ")", "status");
                if (xaiRunBtn) {
                    xaiRunBtn.disabled = true;
                    xaiRunBtn.innerHTML = '<span class="spinner"></span> Running…';
                }
                connectXaiSSE(data.run_id);
            }
        } catch (err) {
            // No active XAI run or server unreachable — stay idle
        }
    })();
})();
