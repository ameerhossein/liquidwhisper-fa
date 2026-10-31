/**
 * Persian ASR training console.
 * Live telemetry over WebSocket with HTTP polling fallback, loss curve,
 * epoch benchmark ladder, canary word diffs, and the training log.
 */

const DEFAULT_EPOCHS = 2;
const numberFormat = new Intl.NumberFormat("en-US");

const el = {
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
  gpuModel: document.getElementById("gpuModel"),
  vramText: document.getElementById("vramText"),
  vramFill: document.getElementById("vramFill"),
  runTitle: document.getElementById("runTitle"),
  runMeta: document.getElementById("runMeta"),
  progressFill: document.getElementById("progressFill"),
  progressTicks: document.getElementById("progressTicks"),
  statLoss: document.getElementById("statLoss"),
  statLR: document.getElementById("statLR"),
  statSpeed: document.getElementById("statSpeed"),
  statElapsed: document.getElementById("statElapsed"),
  btnStart: document.getElementById("btnStart"),
  btnStop: document.getElementById("btnStop"),
  btnFresh: document.getElementById("btnFresh"),
  btnSmoke: document.getElementById("btnSmoke"),
  btnResume: document.getElementById("btnResume"),
  btnMerge: document.getElementById("btnMerge"),
  btnRecover: document.getElementById("btnRecover"),
  hfRecoverDialog: document.getElementById("hfRecoverDialog"),
  inputHfRepo: document.getElementById("inputHfRepo"),
  inputHfRunId: document.getElementById("inputHfRunId"),
  btnHfList: document.getElementById("btnHfList"),
  hfRunList: document.getElementById("hfRunList"),
  btnHfRestore: document.getElementById("btnHfRestore"),
  btnHfRestoreResume: document.getElementById("btnHfRestoreResume"),
  hfRestoreStatus: document.getElementById("hfRestoreStatus"),
  hfRestoreLog: document.getElementById("hfRestoreLog"),
  hfRestoreLogWrap: document.getElementById("hfRestoreLogWrap"),
  btnHfClose: document.getElementById("btnHfClose"),
  btnHfOpenLocal: document.getElementById("btnHfOpenLocal"),
  hfLocalRunNotice: document.getElementById("hfLocalRunNotice"),
  hfSyncBadge: document.getElementById("hfSyncBadge"),
  runSelect: document.getElementById("runSelect"),
  runPicker: document.getElementById("runPicker"),
  runHead: document.getElementById("runHead"),
  runIdentity: document.getElementById("runIdentity"),
  runStatus: document.getElementById("runStatus"),
  runSource: document.getElementById("runSource"),
  runCreated: document.getElementById("runCreated"),
  stageStepper: document.getElementById("stageStepper"),
  stepSaved: document.getElementById("stepSaved"),
  storageWarning: document.getElementById("storageWarning"),
  mergeResult: document.getElementById("mergeResult"),
  mergeResultText: document.getElementById("mergeResultText"),
  mergeLogWrap: document.getElementById("mergeLogWrap"),
  mergeLog: document.getElementById("mergeLog"),
  smokeResult: document.getElementById("smokeResult"),
  smokeStatus: document.getElementById("smokeStatus"),
  smokeLogWrap: document.getElementById("smokeLogWrap"),
  smokeLog: document.getElementById("smokeLog"),
  inputEpochs: document.getElementById("inputEpochs"),
  inputTelephonyMode: document.getElementById("inputTelephonyMode"),
  inputBatchSize: document.getElementById("inputBatchSize"),
  inputGradAccum: document.getElementById("inputGradAccum"),
  inputGradCheckpoint: document.getElementById("inputGradCheckpoint"),
  inputEvalSteps: document.getElementById("inputEvalSteps"),
  inputLearningRate: document.getElementById("inputLearningRate"),
  benchmarkTbody: document.getElementById("benchmarkTbody"),
  benchmarkSubtitle: document.getElementById("benchmarkSubtitle"),
  benchmarkSummary: document.getElementById("benchmarkSummary"),
  canaryContainer: document.getElementById("canaryContainer"),
  consoleBox: document.getElementById("consoleBox"),
  svgChart: document.getElementById("svgChart"),
  chartEmpty: document.getElementById("chartEmpty"),
  syncedAt: document.getElementById("syncedAt"),
};

let ws = null;
let wsActive = false;
let lossHistory = [];
let renderedTicks = 0;
let smokeRunning = false;
let smokePollTimer = null;
let fullRunRunning = false;
let formRestored = false;
let selectedRunId = "latest";
let viewingHistorical = false;
let runViewToken = 0;
let lastRenderedRunKey = null;
let lastRunListSignature = null;
let resumePending = false;
let mergePending = false;
let mergePollTimer = null;
let mergePollGeneration = 0;
let mergePendingPolls = 0;
let mergeError = "";
let mergeViewSignature = null;

const HF_RESTORE_STORAGE_KEY = "asr.hfRestore";
const HF_RESTORE_ACTIVE_STATES = new Set([
  "running",
  "resolving",
  "listing",
  "ledger",
  "checkpoint",
  "verify",
]);

let hfConfig = { enabled: false, repo_id: "", runs_root: "" };
let remoteHfRuns = [];
let hfRestorePending = false;
let hfRestoreTimer = null;
let hfRestoreGeneration = 0;
let hfRestoreResumeRequested = false;
let hfRestoreSelectedRunId = "";
let hfLocalRunId = "";

const STAGE_STATUS_LABELS = {
  pending: "Pending",
  running: "Running",
  completed: "Completed",
  stopped: "Stopped",
  failed: "Failed",
  skipped: "Skipped",
  interrupted: "Interrupted",
};

function humanizeStatus(value) {
  const text = String(value ?? "").trim().replace(/[_-]+/g, " ");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Unknown";
}

// Formatters

function formatDuration(secs) {
  if (secs == null || isNaN(secs)) return "--";
  const total = Math.floor(secs);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function formatLR(lr) {
  return lr ? Number(lr).toExponential(2) : "--";
}

function formatPct(value) {
  return value == null ? "--" : `${Number(value).toFixed(2)}%`;
}

function formatDelta(value) {
  if (value == null || isNaN(value)) return `<span class="muted">—</span>`;
  if (Math.abs(value) < 0.005) return `<span class="delta flat">0.00 pp</span>`;
  const sign = value < 0 ? "−" : "+";
  const cls = value < 0 ? "good" : "bad";
  return `<span class="delta ${cls}">${sign}${Math.abs(value).toFixed(2)} pp</span>`;
}

function formatPp(value) {
  if (value == null || isNaN(value)) return "";
  if (Math.abs(value) < 0.005) return "0.00 pp";
  const sign = value < 0 ? "−" : "+";
  return `${sign}${Math.abs(value).toFixed(2)} pp`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Run provenance

function restoreRunConfig(config) {
  if (formRestored || !config || typeof config !== "object") return;
  formRestored = true;

  const values = formValuesFromConfig(config);
  const fields = [
    [el.inputEpochs, values.epochs],
    [el.inputTelephonyMode, values.telephony_mode],
    [el.inputBatchSize, values.batch_size],
    [el.inputGradAccum, values.grad_accum],
    [el.inputEvalSteps, values.eval_steps],
    [el.inputLearningRate, values.learning_rate],
  ];
  for (const [input, value] of fields) {
    if (input && value != null) input.value = value;
  }
  if (values.gradient_checkpointing != null && el.inputGradCheckpoint) {
    el.inputGradCheckpoint.value = values.gradient_checkpointing;
  }
}

function formatCreatedAt(createdAt) {
  const seconds = Number(createdAt);
  if (createdAt == null || !Number.isFinite(seconds)) return "Created —";
  const date = new Date(seconds * 1000);
  if (isNaN(date.getTime())) return "Created —";
  const text = date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `Created ${text}`;
}

function updateStorageWarning(storage) {
  if (!el.storageWarning) return;
  const show = !!(storage && storage.persistent === false);
  el.storageWarning.hidden = !show;
  if (!show) {
    el.storageWarning.textContent = "";
    return;
  }
  const runsDir = storage.runs_dir || "unknown";
  el.storageWarning.textContent =
    `Runs are not on persistent storage (${runsDir}). ` +
    "Set ASR_RUNS_DIR to a /workspace path.";
}

function renderStepper(run) {
  if (!el.stageStepper) return;
  const stages = stageMap(run);
  for (const stage of el.stageStepper.querySelectorAll(".stage")) {
    const name = stage.dataset.stage;
    const current = stages.get(name) || { status: "pending" };
    const rawStatus = String(current.status || "pending").trim().toLowerCase();
    const known = Object.prototype.hasOwnProperty.call(STAGE_STATUS_LABELS, rawStatus);
    const status = known ? rawStatus : "unknown";
    const detailText = known ? STAGE_STATUS_LABELS[rawStatus] : humanizeStatus(current.status);

    for (const cls of Array.from(stage.classList)) {
      if (cls.startsWith("is-")) stage.classList.remove(cls);
    }
    stage.classList.add(`is-${status}`);
    stage.dataset.status = status;

    const label = stage.querySelector(".stage-label");
    if (label) label.textContent = stageLabel(name);
    const detail = stage.querySelector(".stage-detail");
    if (detail) detail.textContent = detailText;
    stage.title = current.detail ? String(current.detail) : "";
  }
}

function renderRun(run, steps, storage, adapterExists = false) {
  if (!run || !el.runHead) {
    hideRunUi();
    return;
  }

  const status = String(run.status || "unknown");
  const knownStatus = ["starting", "running", "completed", "stopped", "failed", "interrupted"];
  el.runHead.hidden = false;
  el.runIdentity.textContent = run.run_id || "—";
  el.runStatus.textContent = runStatusLabel(run);
  el.runStatus.className = `run-status run-status-${knownStatus.includes(status) ? status : "unknown"}`;
  el.runCreated.textContent = formatCreatedAt(run.created_at);
  el.runSource.hidden = !(run.source === "fake");
  el.runSource.textContent = "Dev simulation";
  renderStepper(run);
  el.stepSaved.textContent = savedStepsLabel(steps);
  updateStorageWarning(storage);
  renderHfBadge(run);

  const viewState = runViewState(run, adapterExists);
  el.btnResume.hidden = !viewState.resumeVisible;
  el.btnMerge.hidden = !viewState.mergeVisible;
  syncRunActionButtons();
  noteRenderedRun(run);
}

function hideRunUi() {
  if (el.runHead) el.runHead.hidden = true;
  if (el.btnResume) el.btnResume.hidden = true;
  if (el.btnMerge) el.btnMerge.hidden = true;
  if (el.hfSyncBadge) {
    el.hfSyncBadge.hidden = true;
    el.hfSyncBadge.textContent = "";
  }
  hideMergeResult();
}

function noteRenderedRun(run) {
  const key = `${run && run.run_id ? run.run_id : ""}|${run && run.status ? run.status : ""}`;
  const previous = lastRenderedRunKey;
  lastRenderedRunKey = key;
  if (viewingHistorical) return;
  // A first run can appear while the page is already open in legacy mode.
  const pickerWasHidden = Boolean(el.runPicker && el.runPicker.hidden);
  if (previous ? previous !== key : pickerWasHidden) refreshRunList();
}

function setRunPickerAvailable(available) {
  if (el.runPicker) el.runPicker.hidden = !available;
  if (!el.runSelect) return;
  el.runSelect.disabled = !available;
  if (!available) el.runSelect.value = "latest";
}

function runOptionLabel(run) {
  const id = String(run.run_id || "");
  const status = run.status ? humanizeStatus(run.status) : "Unknown";
  return `${id} · ${status}`;
}

// Fill the history picker with Latest plus the newest ledger runs. The picker
// stays hidden in legacy/local-demo mode, where no ledger run exists.
async function refreshRunList() {
  if (!el.runSelect) return;
  // Capture the in-flight selection: a user change while this fetch is in
  // flight must win over the stale response below.
  const selectionAtStart = el.runSelect.value || "latest";
  const selectedIdAtStart = selectedRunId;
  try {
    const res = await fetch("/api/runs?limit=20");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    // Bail on a stale response instead of overwriting the user's picker.
    if (
      !shouldApplyPickerRefresh(selectionAtStart, el.runSelect.value || "latest") ||
      !shouldApplyPickerRefresh(selectedIdAtStart, selectedRunId)
    ) {
      return;
    }
    const runs = Array.isArray(data && data.runs) ? data.runs : [];
    const next = resolvePickerSelection(selectionAtStart, runs);

    // Rebuilding the options every poll would close an open native dropdown.
    const signature = runs
      .map((run) => `${run && run.run_id ? run.run_id : ""}:${run && run.status ? run.status : ""}`)
      .join("|");
    if (signature !== lastRunListSignature) {
      lastRunListSignature = signature;
      const fragment = document.createDocumentFragment();
      const latestOption = document.createElement("option");
      latestOption.value = "latest";
      latestOption.textContent = "Latest";
      fragment.append(latestOption);
      for (const run of runs) {
        if (!run || !run.run_id) continue;
        const option = document.createElement("option");
        option.value = String(run.run_id);
        option.textContent = runOptionLabel(run);
        fragment.append(option);
      }
      el.runSelect.replaceChildren(fragment);
    }
    el.runSelect.value = next;
    setRunPickerAvailable(runs.length > 0);

    if (next !== selectionAtStart) {
      selectedRunId = "latest";
      viewingHistorical = false;
      loadRunState();
    }
  } catch (e) {
    // A transient /api/runs failure must not discard a historical view: keep
    // the rendered run and repair only the picker on the next poll. Only the
    // very first legacy failure (no options yet) hides the picker.
    if (lastRunListSignature === null && el.runSelect && el.runSelect.options.length <= 1) {
      setRunPickerAvailable(false);
    }
    console.error("Failed to refresh run list", e);
  }
}

async function selectRun(runId) {
  const target = runId == null || runId === "" ? "latest" : String(runId);
  selectedRunId = target;
  viewingHistorical = target !== "latest";
  formRestored = false;
  mergePending = false;
  mergePendingPolls = 0;
  mergeError = "";
  mergeViewSignature = null;
  stopMergePolling();
  syncRunActionButtons();
  if (el.runSelect && el.runSelect.value !== target) el.runSelect.value = target;

  const token = ++runViewToken;
  const url =
    target === "latest" ? "/api/runs/latest" : `/api/runs/${encodeURIComponent(target)}`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    if (token !== runViewToken) return;
    if (!payload || !payload.run) {
      // Legacy/local demo without a run ledger: keep the pre-run cockpit but
      // still surface a non-persistent runs-root warning.
      updateStorageWarning(payload && payload.storage);
      selectedRunId = "latest";
      hideRunUi();
      return;
    }

    selectedRunId = String(payload.run.run_id || "latest");
    renderRun(
      { ...payload.run, hf_sync: payload.hf_sync },
      payload.steps,
      payload.storage,
      payload.adapter_exists,
    );
    restoreRunConfig(payload.run.config);
    updateTelemetry(payload.telemetry || {}, viewingHistorical);
    if (Array.isArray(payload.benchmarks)) updateBenchmarks(payload.benchmarks);
    if (Array.isArray(payload.loss_history)) updateLossHistory(payload.loss_history);
    renderMergeStatus(payload);
    ensureMergePolling(payload);
    await fetchLogs();
  } catch (e) {
    if (token !== runViewToken) return;
    console.error("Failed to load run", target, e);
    if (target !== "latest") {
      // A stale history entry falls back to Latest rather than a dead view.
      await selectRun("latest");
      refreshRunList();
    } else {
      selectedRunId = "latest";
      hideRunUi();
    }
  }
}

function applyLiveRun(data) {
  if (!data || !data.run || viewingHistorical) return;
  selectedRunId = data.run.run_id ? String(data.run.run_id) : "latest";
  renderRun(
    { ...data.run, hf_sync: data.hf_sync },
    data.steps,
    data.storage,
    data.adapter_exists,
  );
  restoreRunConfig(data.run.config);
  renderMergeStatus(data);
}

async function loadRunState() {
  if (viewingHistorical) return;
  const token = ++runViewToken;
  try {
    const res = await fetch("/api/runs/latest");
    if (token !== runViewToken) return;
    if (!res.ok) {
      hideRunUi();
      return;
    }
    const payload = await res.json();
    if (token !== runViewToken) return;
    if (!payload || !payload.run) {
      // Legacy/local demo without a run ledger: keep the pre-run cockpit but
      // still surface a non-persistent runs-root warning.
      updateStorageWarning(payload && payload.storage);
      hideRunUi();
      return;
    }

    selectedRunId = payload.run.run_id ? String(payload.run.run_id) : "latest";
    renderRun(
      { ...payload.run, hf_sync: payload.hf_sync },
      payload.steps,
      payload.storage,
      payload.adapter_exists,
    );
    restoreRunConfig(payload.run.config);
    if (Array.isArray(payload.benchmarks)) updateBenchmarks(payload.benchmarks);
    if (Array.isArray(payload.loss_history)) updateLossHistory(payload.loss_history);
    renderMergeStatus(payload);
    ensureMergePolling(payload);
  } catch (e) {
    if (token === runViewToken) hideRunUi();
  }
}

// Run actions

function syncRunActionButtons() {
  if (el.btnResume) {
    el.btnResume.disabled = resumePending;
    el.btnResume.textContent = resumePending ? "Resuming…" : "Resume run";
  }
  if (el.btnMerge) {
    el.btnMerge.disabled = mergePending;
    el.btnMerge.textContent = mergePending ? "Merging…" : "Merge adapter";
  }
}

function hideMergeResult() {
  mergeViewSignature = null;
  if (!el.mergeResult) return;
  el.mergeResult.hidden = true;
  el.mergeResult.className = "merge-result";
  if (el.mergeResultText) el.mergeResultText.textContent = "";
  if (el.mergeLogWrap) el.mergeLogWrap.hidden = true;
  if (el.mergeLog) el.mergeLog.textContent = "";
}

function setMergeResult(kind, text, logLines = []) {
  if (!el.mergeResult || !el.mergeResultText) return;
  el.mergeResult.hidden = false;
  el.mergeResult.className = `merge-result ${kind}`;
  el.mergeResultText.textContent = text;
  const hasLog = logLines.length > 0;
  if (el.mergeLogWrap) el.mergeLogWrap.hidden = !hasLog;
  if (el.mergeLog) el.mergeLog.textContent = hasLog ? logLines.join("\n") : "";
  if (hasLog && el.mergeLogWrap) el.mergeLogWrap.open = kind === "fail";
}

// renderMergeStatus accepts a /api/runs/{id} payload and reflects only the
// merge stage; the run's completed training status is never rewritten here.
function renderMergeStatus(payload) {
  if (!el.mergeResult) return;
  const run = payload && payload.run;
  const stage = run ? stageMap(run).get("merge") : null;
  const status = stage ? String(stage.status || "pending") : "pending";
  let kind = "idle";
  let text = "";
  let logs = [];

  if (status === "running" || (status === "pending" && mergePending)) {
    kind = "running";
    text = "Merging adapter…";
  } else if (status === "completed") {
    kind = "pass";
    text = stage.detail ? `Merge complete — ${stage.detail}` : "Merge complete.";
  } else if (["failed", "stopped", "interrupted"].includes(status)) {
    kind = "fail";
    const label = status === "failed" ? "Merge failed" : `Merge ${humanizeStatus(status).toLowerCase()}`;
    text = stage.detail ? `${label}: ${stage.detail}` : `${label}.`;
    logs = Array.isArray(stage.log_tail) ? stage.log_tail.map(String) : [];
  } else if (status === "pending" && mergeError) {
    kind = "fail";
    text = mergeError;
  }

  if (kind === "pass" || kind === "fail") {
    mergePending = false;
    syncRunActionButtons();
  }

  const signature = `${kind}\u0000${text}\u0000${logs.join("\n")}`;
  if (signature === mergeViewSignature) return;
  if (kind === "idle") {
    if (el.mergeResult.hidden) {
      mergeViewSignature = signature;
    } else {
      hideMergeResult();
    }
    return;
  }
  mergeViewSignature = signature;
  setMergeResult(kind, text, logs);
}

async function errorDetail(res) {
  const text = await res.text();
  if (!text) return `HTTP ${res.status}`;
  let detail = text;
  try {
    const err = JSON.parse(text);
    detail = err.detail || err.error || text;
  } catch {
    // Non-JSON body: keep the raw text.
  }
  if (typeof detail === "string" && detail.length > 500) detail = `${detail.slice(0, 500)}…`;
  return detail || `HTTP ${res.status}`;
}

function stopMergePolling() {
  mergePollGeneration += 1;
  if (mergePollTimer) {
    clearTimeout(mergePollTimer);
    mergePollTimer = null;
  }
}

function scheduleMergePoll(runId, delay = 2000, generation = mergePollGeneration) {
  if (mergePollTimer) clearTimeout(mergePollTimer);
  mergePollTimer = setTimeout(() => pollMergeStatus(runId, generation), delay);
}

function ensureMergePolling(payload) {
  const run = payload && payload.run;
  if (!run || !run.run_id) return;
  const stage = stageMap(run).get("merge");
  if (!stage || stage.status !== "running") return;
  mergePending = true;
  syncRunActionButtons();
  if (mergePollTimer) return;
  scheduleMergePoll(String(run.run_id), 2000);
}

async function pollMergeStatus(runId, generation) {
  mergePollTimer = null;
  if (!runId || generation !== mergePollGeneration || selectedRunId !== runId) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    // The user may have switched runs while this request was in flight.
    if (generation !== mergePollGeneration || selectedRunId !== runId) return;
    renderRun(
      payload.run ? { ...payload.run, hf_sync: payload.hf_sync } : payload.run,
      payload.steps,
      payload.storage,
      payload.adapter_exists,
    );
    if (Array.isArray(payload.benchmarks)) updateBenchmarks(payload.benchmarks);
    if (Array.isArray(payload.loss_history)) updateLossHistory(payload.loss_history);
    renderMergeStatus(payload);

    const stage = payload.run ? stageMap(payload.run).get("merge") : null;
    const status = stage ? String(stage.status || "pending") : "pending";
    if (status === "running") {
      mergePendingPolls = 0;
      scheduleMergePoll(runId, 2000, generation);
      return;
    }
    if (status === "pending" && mergePendingPolls < 3) {
      // The POST has returned but the ledger may not have caught up yet.
      mergePendingPolls += 1;
      scheduleMergePoll(runId, 2000, generation);
      return;
    }
    mergePending = false;
    if (status === "pending") {
      mergeError = "Merge did not start. Check the training log.";
      mergeViewSignature = null;
      renderMergeStatus(payload);
    }
    syncRunActionButtons();
    refreshRunList();
  } catch (e) {
    if (generation !== mergePollGeneration || selectedRunId !== runId) return;
    mergePending = false;
    mergeError = `Merge status unavailable: ${e}`;
    mergeViewSignature = null;
    renderMergeStatus({});
    syncRunActionButtons();
    refreshRunList();
  }
}

async function resumeRun() {
  const runId = selectedRunId !== "latest" ? selectedRunId : null;
  if (!runId || resumePending) return;
  resumePending = true;
  syncRunActionButtons();
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!res.ok) {
      alert(`Resume failed: ${await errorDetail(res)}`);
      return;
    }
    await refreshRunList();
    await selectRun("latest");
  } catch (e) {
    alert(`Resume failed: ${e}`);
  } finally {
    resumePending = false;
    syncRunActionButtons();
  }
}

async function mergeRun() {
  const runId = selectedRunId !== "latest" ? selectedRunId : null;
  if (!runId || mergePending) return;
  const generation = mergePollGeneration;
  mergePending = true;
  mergePendingPolls = 0;
  mergeError = "";
  mergeViewSignature = null;
  syncRunActionButtons();
  renderMergeStatus({ run: { stages: [{ name: "merge", status: "running" }] } });
  const stillSelected = () => generation === mergePollGeneration && selectedRunId === runId;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!res.ok) {
      const detail = await errorDetail(res);
      if (stillSelected()) {
        mergePending = false;
        mergeError = `Could not start merge: ${detail}`;
        mergeViewSignature = null;
        renderMergeStatus({});
        syncRunActionButtons();
      }
      refreshRunList();
      return;
    }
    if (stillSelected()) scheduleMergePoll(runId, 2000, generation);
    refreshRunList();
  } catch (e) {
    if (!stillSelected()) return;
    mergePending = false;
    mergeError = `Could not start merge: ${e}`;
    mergeViewSignature = null;
    renderMergeStatus({});
    syncRunActionButtons();
  }
}

// Hugging Face recovery

async function loadHfConfig() {
  if (!el.btnRecover) return;
  try {
    const res = await fetch("/api/hf/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    hfConfig = {
      enabled: data && data.enabled !== false,
      repo_id: String((data && data.repo_id) || ""),
      runs_root: String((data && data.runs_root) || ""),
    };
    // The Recover button remains visible with no local runs; only an explicit
    // enabled:false removes it, because HF support is genuinely unavailable.
    el.btnRecover.hidden = Boolean(data && data.enabled === false);
    if (el.inputHfRepo && !el.inputHfRepo.value.trim() && hfConfig.repo_id) {
      el.inputHfRepo.value = hfConfig.repo_id;
    }
  } catch (e) {
    // Keep the button usable: a config failure must not remove the only way
    // back to a restore that may still be running.
    console.error("Failed to load HF config", e);
  }
}

function openRecoverDialog(preserveStatus = false) {
  if (!el.hfRecoverDialog) return;
  if (el.inputHfRepo && !el.inputHfRepo.value.trim() && hfConfig.repo_id) {
    el.inputHfRepo.value = hfConfig.repo_id;
  }
  if (!preserveStatus && !hfRestorePending) {
    setHfRestoreStatus("");
    hideHfRestoreLog();
    hideHfLocalRunNotice();
  }
  if (!el.hfRecoverDialog.open) {
    if (typeof el.hfRecoverDialog.showModal === "function") {
      try {
        el.hfRecoverDialog.showModal();
      } catch (e) {
        el.hfRecoverDialog.setAttribute("open", "");
      }
    } else {
      el.hfRecoverDialog.setAttribute("open", "");
    }
  }
  if (el.inputHfRunId) el.inputHfRunId.focus();
  syncHfRestoreButtons();
}

function closeRecoverDialog() {
  if (!el.hfRecoverDialog) return;
  if (el.hfRecoverDialog.open) el.hfRecoverDialog.close();
  else el.hfRecoverDialog.removeAttribute("open");
}

function setHfRestoreStatus(text, isError = false) {
  if (!el.hfRestoreStatus) return;
  el.hfRestoreStatus.textContent = String(text || "");
  el.hfRestoreStatus.className = `hf-restore-status${isError ? " is-error" : ""}`;
}

function hideHfRestoreLog() {
  if (el.hfRestoreLogWrap) el.hfRestoreLogWrap.hidden = true;
  if (el.hfRestoreLog) el.hfRestoreLog.textContent = "";
}

function showHfLocalRunNotice(runId) {
  hfLocalRunId = String(runId || "");
  if (el.hfLocalRunNotice) el.hfLocalRunNotice.hidden = !hfLocalRunId;
  if (el.btnHfOpenLocal) {
    el.btnHfOpenLocal.textContent = hfLocalRunId
      ? `Open local run ${hfLocalRunId}`
      : "Open the existing local run";
  }
}

function hideHfLocalRunNotice() {
  hfLocalRunId = "";
  if (el.hfLocalRunNotice) el.hfLocalRunNotice.hidden = true;
}

function selectedHfRemoteRun() {
  const runId = el.inputHfRunId ? String(el.inputHfRunId.value || "").trim() : "";
  if (!runId) return null;
  return remoteHfRuns.find((run) => run && String(run.run_id || "") === runId) || null;
}

function hfRestoreSelection() {
  const runId = el.inputHfRunId ? String(el.inputHfRunId.value || "").trim() : "";
  const run = runId ? selectedHfRemoteRun() : null;
  // A manually typed id that is not in the listed page still gets Restore
  // only and Resume; the server validates it and reports a missing checkpoint.
  const state = run
    ? recoverActionState(run)
    : { canRestore: Boolean(runId), canResume: Boolean(runId), reason: "" };
  return { runId, run, state };
}

function syncHfRestoreButtons() {
  const { runId, state } = hfRestoreSelection();
  if (el.btnRecover) el.btnRecover.disabled = hfRestorePending;
  if (el.btnHfRestore) {
    el.btnHfRestore.disabled = hfRestorePending || !runId || !state.canRestore;
    el.btnHfRestore.title = !runId ? "Choose a remote run first." : state.reason || "";
  }
  if (el.btnHfRestoreResume) {
    el.btnHfRestoreResume.disabled = hfRestorePending || !runId || !state.canResume;
    el.btnHfRestoreResume.title = state.canResume
      ? "Restore the run and continue training from its last checkpoint."
      : state.reason || "No complete checkpoint is available.";
  }
  if (el.btnHfList) el.btnHfList.disabled = hfRestorePending;
}

function renderHfBadge(run) {
  if (!el.hfSyncBadge) return;
  const sync = run && run.hf_sync;
  if (!sync || typeof sync !== "object" || Object.keys(sync).length === 0) {
    el.hfSyncBadge.hidden = true;
    el.hfSyncBadge.textContent = "";
    el.hfSyncBadge.className = "hf-sync-badge";
    return;
  }
  const label = hfSyncLabel(sync);
  el.hfSyncBadge.textContent = label;
  el.hfSyncBadge.hidden = !label;
  const status = String(sync.status || "idle").trim().toLowerCase();
  el.hfSyncBadge.className = `hf-sync-badge${label ? ` is-${status}` : ""}`;
}

function renderRemoteRuns(runs) {
  remoteHfRuns = Array.isArray(runs)
    ? runs.filter((run) => run && run.run_id)
    : [];
  if (!el.hfRunList) return;
  if (remoteHfRuns.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty-cell";
    cell.textContent = "No remote runs found.";
    row.append(cell);
    el.hfRunList.replaceChildren(row);
    syncHfRestoreButtons();
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const run of remoteHfRuns) {
    const runId = String(run.run_id || "");
    const hasCheckpoint = truthyFlagValue(run.has_checkpoint);
    const row = document.createElement("tr");
    row.className = "hf-run-row";
    row.tabIndex = 0;
    row.dataset.runId = runId;

    const runCell = document.createElement("td");
    runCell.className = "hf-run-id";
    runCell.textContent = runId;
    const statusCell = document.createElement("td");
    statusCell.className = "hf-run-status";
    statusCell.textContent = String(run.status || "unknown");
    const stepCell = document.createElement("td");
    stepCell.className = "num";
    const step = Number(run.last_step);
    stepCell.textContent =
      run.last_step != null && Number.isFinite(step) ? numberFormat.format(step) : "—";
    const checkpointCell = document.createElement("td");
    checkpointCell.className = hasCheckpoint ? "hf-has-checkpoint" : "muted";
    checkpointCell.textContent = hasCheckpoint ? "Yes" : "No";
    row.append(runCell, statusCell, stepCell, checkpointCell);

    const select = () => {
      if (el.inputHfRunId) el.inputHfRunId.value = runId;
      const state = recoverActionState(run);
      setHfRestoreStatus(state.reason || `Selected ${runId}.`);
      syncHfRestoreButtons();
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    fragment.append(row);
  }
  el.hfRunList.replaceChildren(fragment);
  syncHfRestoreButtons();
}

async function loadRemoteRuns() {
  const repo = el.inputHfRepo ? String(el.inputHfRepo.value || "").trim() : "";
  if (el.btnHfList) el.btnHfList.disabled = true;
  setHfRestoreStatus("Listing remote runs…");
  try {
    const url = repo ? `/api/hf/runs?repo=${encodeURIComponent(repo)}` : "/api/hf/runs";
    const res = await fetch(url);
    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();
    renderRemoteRuns(data && Array.isArray(data.runs) ? data.runs : []);
    const count = remoteHfRuns.length;
    setHfRestoreStatus(
      count
        ? `Found ${numberFormat.format(count)} remote run${count === 1 ? "" : "s"}.`
        : "No remote runs found.",
    );
  } catch (e) {
    renderRemoteRuns([]);
    setHfRestoreStatus(
      `Could not list remote runs: ${e && e.message ? e.message : e}`,
      true,
    );
  } finally {
    if (el.btnHfList) el.btnHfList.disabled = hfRestorePending;
  }
}

function saveHfRestoreState(runId, repo, resume) {
  try {
    localStorage.setItem(
      HF_RESTORE_STORAGE_KEY,
      JSON.stringify({ runId, repo, resume }),
    );
  } catch (e) {
    // Storage can be unavailable in a private window; polling still works.
  }
}

function readHfRestoreState() {
  try {
    const raw = localStorage.getItem(HF_RESTORE_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function clearHfRestoreState() {
  try {
    localStorage.removeItem(HF_RESTORE_STORAGE_KEY);
  } catch (e) {
    // Nothing to clear.
  }
}

function scheduleHfRestorePoll(delay = 2000, generation = hfRestoreGeneration) {
  if (hfRestoreTimer) clearTimeout(hfRestoreTimer);
  hfRestoreTimer = setTimeout(() => pollRestore(generation), delay);
}

async function startRestore(resume) {
  if (hfRestorePending) return;
  const { runId, run, state } = hfRestoreSelection();
  if (!runId || !state.canRestore) {
    setHfRestoreStatus(state.reason || "Choose a remote run to restore.", true);
    return;
  }
  if (resume && run && !state.canResume) {
    setHfRestoreStatus(state.reason || "This run cannot be resumed.", true);
    return;
  }

  const repo = el.inputHfRepo ? String(el.inputHfRepo.value || "").trim() : "";
  hfRestorePending = true;
  hfRestoreResumeRequested = Boolean(resume);
  hfRestoreSelectedRunId = runId;
  hfRestoreGeneration += 1;
  const generation = hfRestoreGeneration;
  hideHfLocalRunNotice();
  hideHfRestoreLog();
  setHfRestoreStatus(
    resume
      ? `Restoring ${runId} and preparing to resume…`
      : `Restoring ${runId}…`,
  );
  syncHfRestoreButtons();
  saveHfRestoreState(runId, repo, Boolean(resume));

  try {
    const body = { run_id: runId, include_merged: false };
    if (repo) body.repo = repo;
    const res = await fetch("/api/hf/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await errorDetail(res);
      hfRestorePending = false;
      hfRestoreResumeRequested = false;
      clearHfRestoreState();
      if (res.status === 409) {
        setHfRestoreStatus(`That run already exists locally. ${detail}`, true);
        showHfLocalRunNotice(runId);
      } else {
        setHfRestoreStatus(`Could not start restore: ${detail}`, true);
      }
      syncHfRestoreButtons();
      return;
    }
    const data = await res.json().catch(() => ({}));
    if (data && data.run_id) hfRestoreSelectedRunId = String(data.run_id);
    scheduleHfRestorePoll(2000, generation);
  } catch (e) {
    hfRestorePending = false;
    hfRestoreResumeRequested = false;
    clearHfRestoreState();
    setHfRestoreStatus(
      `Could not start restore: ${e && e.message ? e.message : e}`,
      true,
    );
    syncHfRestoreButtons();
  }
}

const HF_RESTORE_STATE_LABELS = {
  idle: "Idle.",
  running: "Restoring…",
  resolving: "Resolving the remote run…",
  listing: "Listing remote files…",
  ledger: "Restoring the run ledger…",
  checkpoint: "Downloading the checkpoint…",
  verify: "Verifying the restored run…",
  completed: "Restore complete.",
  failed: "Restore failed.",
  interrupted: "Restore interrupted.",
};

function renderHfRestoreStatus(data) {
  const state = String((data && data.state) || "idle").toLowerCase();
  const label = HF_RESTORE_STATE_LABELS[state] || `Restoring (${state})…`;
  const message = data && data.message ? String(data.message) : "";
  const isError = state === "failed" || state === "interrupted";
  setHfRestoreStatus(message ? `${label} — ${message}` : label, isError);

  const lines =
    data && Array.isArray(data.log_tail) ? data.log_tail.map(String) : [];
  if (el.hfRestoreLogWrap) el.hfRestoreLogWrap.hidden = lines.length === 0;
  if (el.hfRestoreLog) el.hfRestoreLog.textContent = lines.join("\n");
  if (isError && lines.length && el.hfRestoreLogWrap) {
    el.hfRestoreLogWrap.open = true;
  }

  const runId = data && data.run_id ? String(data.run_id) : "";
  if (runId) hfRestoreSelectedRunId = runId;
  if (runId && el.inputHfRunId && !el.inputHfRunId.value.trim()) {
    el.inputHfRunId.value = runId;
  }
}

async function pollRestore(generation = hfRestoreGeneration) {
  hfRestoreTimer = null;
  if (!hfRestorePending || generation !== hfRestoreGeneration) return;
  try {
    const res = await fetch("/api/hf/restore/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!hfRestorePending || generation !== hfRestoreGeneration) return;

    const state = String((data && data.state) || "idle").toLowerCase();
    renderHfRestoreStatus(data);

    if (HF_RESTORE_ACTIVE_STATES.has(state)) {
      scheduleHfRestorePoll(2000, generation);
      return;
    }

    if (state === "completed") {
      const restoredId = String(
        (data && data.run_id) || hfRestoreSelectedRunId || "",
      );
      const resumeAfterRestore = hfRestoreResumeRequested;
      hfRestorePending = false;
      hfRestoreResumeRequested = false;
      if (hfRestoreTimer) {
        clearTimeout(hfRestoreTimer);
        hfRestoreTimer = null;
      }
      clearHfRestoreState();
      syncHfRestoreButtons();
      closeRecoverDialog();
      await refreshRunList();
      await selectRun(restoredId || "latest");
      if (resumeAfterRestore && restoredId) await resumeRun();
      return;
    }

    if (state === "failed" || state === "interrupted") {
      hfRestorePending = false;
      hfRestoreResumeRequested = false;
      clearHfRestoreState();
      syncHfRestoreButtons();
      const text = state === "failed" ? "Restore failed" : "Restore interrupted";
      const message = (data && data.message) || "see the restore log.";
      setHfRestoreStatus(`${text}: ${message}`, true);
      if (
        el.hfRestoreLogWrap &&
        data &&
        Array.isArray(data.log_tail) &&
        data.log_tail.length > 0
      ) {
        el.hfRestoreLogWrap.open = true;
      }
      refreshRunList();
      return;
    }

    // An unexpected idle status means the server never recorded the restore.
    hfRestorePending = false;
    hfRestoreResumeRequested = false;
    clearHfRestoreState();
    syncHfRestoreButtons();
    setHfRestoreStatus("Restore status is idle; the restore did not start.", true);
  } catch (e) {
    if (!hfRestorePending || generation !== hfRestoreGeneration) return;
    setHfRestoreStatus(
      `Restore status unavailable: ${e && e.message ? e.message : e}. Retrying…`,
      true,
    );
    scheduleHfRestorePoll(5000, generation);
  }
}

// A reload must not lose a restore that is still running on the server.
async function restoreHfProgressOnLoad() {
  try {
    const res = await fetch("/api/hf/restore/status");
    if (!res.ok) return;
    const data = await res.json();
    const state = String((data && data.state) || "idle").toLowerCase();
    if (!HF_RESTORE_ACTIVE_STATES.has(state)) {
      // A completed restore may still owe its "resume" if the page was
      // reloaded in the window between completion and the poll that would
      // have started training.
      const stored = readHfRestoreState();
      clearHfRestoreState();
      const statusRunId = data && data.run_id ? String(data.run_id) : "";
      if (
        state === "completed" &&
        stored &&
        stored.resume &&
        statusRunId &&
        String(stored.runId || "") === statusRunId
      ) {
        await refreshRunList();
        await selectRun(statusRunId);
        await resumeRun();
      }
      return;
    }

    const stored = readHfRestoreState();
    const storedRunId = stored && stored.runId ? String(stored.runId) : "";
    const statusRunId = data && data.run_id ? String(data.run_id) : "";
    hfRestorePending = true;
    hfRestoreResumeRequested = Boolean(
      stored && stored.resume && storedRunId && storedRunId === (statusRunId || storedRunId),
    );
    hfRestoreSelectedRunId = statusRunId || storedRunId;
    hfRestoreGeneration += 1;

    openRecoverDialog(true);
    if (el.inputHfRunId && hfRestoreSelectedRunId) {
      el.inputHfRunId.value = hfRestoreSelectedRunId;
    }
    if (el.inputHfRepo && stored && stored.repo && !el.inputHfRepo.value.trim()) {
      el.inputHfRepo.value = String(stored.repo);
    }
    renderHfRestoreStatus(data);
    syncHfRestoreButtons();
    scheduleHfRestorePoll(2000, hfRestoreGeneration);
  } catch (e) {
    // The cockpit still works without the restore status.
  }
}

async function openHfLocalRun() {
  if (!hfLocalRunId) return;
  const runId = hfLocalRunId;
  closeRecoverDialog();
  await refreshRunList();
  await selectRun(runId);
}

// Connection

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsActive = true;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // A pinned historical run keeps its own telemetry, benchmarks, loss,
        // and stage rail; live data resumes only when Latest is selected.
        if (!viewingHistorical) {
          if (data.telemetry) updateTelemetry(data.telemetry);
          if (data.benchmarks) updateBenchmarks(data.benchmarks);
          if (data.loss_history) updateLossHistory(data.loss_history);
          if (Object.prototype.hasOwnProperty.call(data, "run")) {
            if (data.run) {
              applyLiveRun(data);
            } else {
              updateStorageWarning(data.storage);
              hideRunUi();
            }
          }
        }
      } catch (e) {
        console.error("Error parsing live payload:", e);
      }
    };

    ws.onclose = () => {
      wsActive = false;
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
      wsActive = false;
    };
  } catch (e) {
    wsActive = false;
  }
}

// Resilient HTTP polling fallback (keeps the local demo working without WebSocket)
async function pollTelemetryFallback() {
  if (wsActive || viewingHistorical) return;
  try {
    const [statusRes, benchRes, lossRes] = await Promise.all([
      fetch("/api/status").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/benchmarks").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/loss-history").then((r) => (r.ok ? r.json() : null)),
    ]);
    if (statusRes) updateTelemetry(statusRes);
    if (!viewingHistorical) {
      if (benchRes) updateBenchmarks(benchRes);
      if (lossRes) updateLossHistory(lossRes);
    }
  } catch (e) {
    // Network hiccup: the next poll will retry.
  }
}

// While a historical run is pinned, only the Start/Halt controls follow the
// live job; the rendered run content stays frozen.
async function pollLiveControls() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const tel = await res.json();
    const status = String((tel && tel.status) || "");
    const running =
      !!tel.process_running ||
      ["initializing", "training", "stopping", "evaluating", "evaluating_benchmark"].includes(status);
    fullRunRunning = running;
    syncRunButtons(status === "stopping");
  } catch (e) {
    // Keep the previous control state; the next poll retries.
  }
}

// Run status

function totalEpochs(tel) {
  const step = tel.step || 0;
  const epoch = tel.epoch || 0;
  const maxSteps = tel.max_steps || 0;
  if (maxSteps && step > 0 && epoch >= 0.5) {
    return Math.max(1, Math.round((maxSteps * epoch) / step));
  }
  return Math.max(1, parseInt(el.inputEpochs.value, 10) || DEFAULT_EPOCHS);
}

function renderTicks(total) {
  if (total === renderedTicks) return;
  renderedTicks = total;
  el.progressTicks.innerHTML = Array.from({ length: total - 1 }, (_, i) =>
    `<span class="tick" style="left:${((i + 1) / total) * 100}%"></span>`
  ).join("");
}

const STATE_LABELS = {
  idle: "Idle",
  initializing: "Initializing",
  training: "Training",
  stopping: "Stopping (saving checkpoint)",
  evaluating: "Evaluating on held-out test",
  evaluating_benchmark: "Evaluating on held-out test",
  stopped: "Stopped",
  completed: "Run completed",
  failed: "Run failed (see log)",
  interrupted: "Run interrupted",
};

function syncRunButtons(stopping = false) {
  const busy = fullRunRunning || smokeRunning;
  el.btnStart.disabled = busy;
  el.btnFresh.disabled = busy;
  if (el.btnSmoke) el.btnSmoke.disabled = busy;
  el.btnStop.disabled = !fullRunRunning || stopping;
}

function updateTelemetry(tel, historical = false) {
  const status = tel.status || "idle";
  const running = !!tel.process_running || ["initializing", "training", "stopping", "evaluating_benchmark"].includes(status);

  let state = status === "evaluating_benchmark" ? "evaluating" : status;
  if (!STATE_LABELS[state]) state = running ? "training" : "idle";
  el.statusDot.className = `dot dot-${state}`;
  el.statusLabel.textContent = STATE_LABELS[state];

  // A historical snapshot describes the selected run, not the live job:
  // keep the start/halt controls driven by the actual current process.
  if (!historical) {
    fullRunRunning = running;
    syncRunButtons(state === "stopping");
  }

  const gpu = tel.gpu || {};
  if (gpu.available) {
    el.gpuModel.textContent = gpu.device_name || "NVIDIA GPU";
    el.vramText.textContent = `${gpu.allocated_gb || 0} / ${gpu.total_gb || 0} GB`;
    el.vramFill.style.width = `${gpu.utilization_pct || 0}%`;
  } else if (gpu.device_name && gpu.device_name !== "N/A") {
    // Trainer on CPU reports device_name "CPU": show it instead of "unavailable".
    el.gpuModel.textContent = gpu.device_name;
    el.vramText.textContent = "—";
    el.vramFill.style.width = "0%";
  } else if (!tel.gpu) {
    // No reading yet (first paint before /api/status resolves): keep placeholder.
  } else {
    el.gpuModel.textContent = "GPU unavailable";
    el.vramText.textContent = "—";
    el.vramFill.style.width = "0%";
  }

  const step = tel.step || 0;
  const maxSteps = tel.max_steps || 0;
  const epoch = tel.epoch || 0;
  const total = totalEpochs(tel);
  const started = running || step > 0 || ["completed", "stopped", "failed", "interrupted"].includes(status);
  const pct = maxSteps ? Math.min(100, (step / maxSteps) * 100) : 0;

  renderTicks(total);
  el.progressFill.style.width = `${pct}%`;
  el.runTitle.innerHTML = started
    ? `Epoch ${Math.min(epoch, total).toFixed(2)} <span class="run-title-of">of ${total}</span>`
    : "Not started";

  const parts = [];
  if (maxSteps) {
    parts.push(`Step ${numberFormat.format(step)} of ${numberFormat.format(maxSteps)}`, `${pct.toFixed(1)}%`);
  }
  if (running && tel.eta_seconds) {
    parts.push(`${formatDuration(tel.eta_seconds)} remaining`);
  }
  el.runMeta.textContent = parts.length ? parts.join(" · ") : "Set the run parameters below and start.";

  el.statLoss.textContent = started && tel.loss != null ? Number(tel.loss).toFixed(4) : "--";
  el.statLR.textContent = started ? formatLR(tel.learning_rate) : "--";
  el.statSpeed.textContent = started && tel.speed_s_per_step ? `${Number(tel.speed_s_per_step).toFixed(2)} s` : "--";
  el.statElapsed.textContent = formatDuration(tel.elapsed_seconds);

  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  el.syncedAt.textContent = `Synced ${time}`;
}

// Loss curve

function updateLossHistory(history) {
  if (!Array.isArray(history)) return;
  lossHistory = history.filter((h) => h.loss != null && !isNaN(h.loss));
  drawLoss();
}

function drawLoss() {
  const svg = el.svgChart;
  const width = svg.clientWidth;
  const height = svg.clientHeight;
  el.chartEmpty.hidden = lossHistory.length > 0;

  if (!lossHistory.length || !width || !height) {
    svg.innerHTML = "";
    return;
  }

  const pad = { top: 12, right: 64, bottom: 28, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const losses = lossHistory.map((h) => Number(h.loss));
  const steps = lossHistory.map((h) => Number(h.step));
  const lo = Math.min(...losses) * 0.97;
  const hi = Math.max(...losses) * 1.03;
  const sMin = Math.min(...steps);
  const sMax = Math.max(...steps);

  const yOf = (v) => pad.top + (1 - (v - lo) / (hi - lo || 1)) * plotH;
  const xOf = (s) => pad.left + ((s - sMin) / (sMax - sMin || 1)) * plotW;

  const gridLines = [0, 0.5, 1].map((f) => {
    const value = lo + f * (hi - lo);
    const y = yOf(value).toFixed(1);
    return `<line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${y}" y2="${y}"/>
      <text class="axis-label" x="${pad.left - 8}" y="${y}" dy="4" text-anchor="end">${value.toFixed(3)}</text>`;
  }).join("");

  let epochMarks = "";
  let currentEpoch = Math.floor(lossHistory[0].epoch || 0);
  lossHistory.forEach((h) => {
    const e = Math.floor(h.epoch || 0);
    if (e > currentEpoch) {
      currentEpoch = e;
      const x = xOf(h.step).toFixed(1);
      epochMarks += `<line class="epoch-line" x1="${x}" x2="${x}" y1="${pad.top}" y2="${pad.top + plotH}"/>
        <text class="axis-label" x="${Number(x) + 6}" y="${pad.top + 10}">Epoch ${e}</text>`;
    }
  });

  const path = lossHistory
    .map((h, i) => `${i ? "L" : "M"}${xOf(h.step).toFixed(1)} ${yOf(h.loss).toFixed(1)}`)
    .join(" ");
  const last = lossHistory[lossHistory.length - 1];
  const lx = xOf(last.step).toFixed(1);
  const ly = yOf(last.loss).toFixed(1);

  svg.innerHTML = `
    ${gridLines}
    ${epochMarks}
    <text class="axis-label" x="${pad.left}" y="${height - 6}">step ${numberFormat.format(sMin)}</text>
    <text class="axis-label" x="${width - pad.right}" y="${height - 6}" text-anchor="end">step ${numberFormat.format(sMax)}</text>
    <path class="loss-line" d="${path}"/>
    <circle class="loss-dot" cx="${lx}" cy="${ly}" r="3.5"/>
    <text class="loss-value" x="${Number(lx) + 10}" y="${ly}" dy="4">${Number(last.loss).toFixed(4)}</text>`;
}

// Benchmark ladder

function updateBenchmarks(list) {
  if (!Array.isArray(list)) return;

  if (list.length === 0) {
    el.benchmarkTbody.innerHTML = `
      <tr><td colspan="9" class="empty-cell">No measured baseline yet. Start a run to generate the preflight.</td></tr>`;
    el.benchmarkSummary.textContent = "";
    el.benchmarkSubtitle.textContent = "Held-out test split. Lower is better.";
    return;
  }

  const baseline = findBuzzBaseline(list);
  const baselineWer =
    baseline && baseline.normalized_wer != null && !isNaN(Number(baseline.normalized_wer))
      ? Number(baseline.normalized_wer)
      : null;
  const baselinePhone =
    baseline && baseline.phone_normalized_wer != null && !isNaN(Number(baseline.phone_normalized_wer))
      ? Number(baseline.phone_normalized_wer)
      : null;
  const comparable = (row) => benchmarkComparable(row, baseline);

  const { best, scaleMax } = computeBenchmarkScale(list, baseline);
  const share = (v) => `${Math.min(100, Math.max(0, (v / scaleMax) * 100)).toFixed(2)}%`;

  el.benchmarkTbody.innerHTML = list
    .map((row) => {
      const hasWer = row.normalized_wer != null && !isNaN(Number(row.normalized_wer));
      const wer = hasWer ? Number(row.normalized_wer) : 0;
      const phone =
        row.phone_normalized_wer != null && !isNaN(Number(row.phone_normalized_wer))
          ? Number(row.phone_normalized_wer)
          : null;
      const isBaseline = !!row.is_baseline;
      const isBest = isBestBenchmarkRow(row, best, baseline);

      const rowClass = isBaseline ? "is-baseline" : isBest ? "is-best" : "";
      const barClass = isBaseline ? "is-base" : isBest ? "is-best" : "";
      const tag = isBaseline
        ? `<span class="tag">${row.is_reference ? "Reference" : "Baseline"}</span>`
        : isBest
          ? `<span class="tag tag-best">Best</span>`
          : "";

      const cleanDelta = comparable(row) && hasWer && baselineWer != null ? wer - baselineWer : null;
      const phoneDelta = comparable(row) && phone != null && baselinePhone != null ? phone - baselinePhone : null;
      const baseMark =
        baselineWer != null && hasWer
          ? `<span class="bar-ref" style="left:${share(baselineWer)}"></span>`
          : "";

      return `
        <tr class="${rowClass}">
          <th scope="row"><span class="row-name">${escapeHtml(row.epoch || "Checkpoint")}</span>${tag}</th>
          <td class="wer">
            <div class="wer-inner">
              <div class="bar ${barClass}">
                <span class="bar-fill" style="width:${hasWer ? share(wer) : "0%"}"></span>
                ${baseMark}
              </div>
              <span class="wer-value">${hasWer ? `${wer.toFixed(2)}%` : "—"}</span>
            </div>
          </td>
          <td class="num">${formatDelta(cleanDelta)}</td>
          <td class="num">${formatPct(row.phone_normalized_wer)}</td>
          <td class="num">${formatDelta(phoneDelta)}</td>
          <td class="num opt">${formatPct(row.normalized_cer)}</td>
          <td class="num opt">${formatPct(row.formal_wer)}</td>
          <td class="num opt">${formatPct(row.informal_wer)}</td>
          <td class="num opt">${formatPct(row.diagnostic_wer)}</td>
        </tr>`;
    })
    .join("");

  const sampleCount = baseline && baseline.total_samples != null ? Number(baseline.total_samples) : NaN;
  el.benchmarkSubtitle.textContent =
    Number.isFinite(sampleCount) && sampleCount > 0
      ? `Held-out test split, ${numberFormat.format(sampleCount)}-recording benchmark subset. Lower is better.`
      : "Held-out test split. Lower is better.";

  const phoneRows = list.filter(
    (r) =>
      !r.is_baseline &&
      comparable(r) &&
      r.phone_normalized_wer != null &&
      !isNaN(Number(r.phone_normalized_wer))
  );
  const bestPhoneRow = phoneRows.length
    ? phoneRows.reduce((a, b) => (Number(b.phone_normalized_wer) < Number(a.phone_normalized_wer) ? b : a))
    : null;

  if (bestPhoneRow && baselinePhone != null) {
    const bestPhone = Number(bestPhoneRow.phone_normalized_wer);
    const delta = ` (${formatPp(bestPhone - baselinePhone)})`;
    el.benchmarkSummary.textContent = `Phone WER ${baselinePhone.toFixed(2)}% → ${bestPhone.toFixed(2)}%${delta}`;
  } else if (!baseline) {
    el.benchmarkSummary.textContent = "No measured baseline yet. Start a run to generate the preflight.";
  } else {
    el.benchmarkSummary.textContent = "";
  }

  const latest = [...list]
    .reverse()
    .find((r) => Array.isArray(r.canary_transcriptions) && r.canary_transcriptions.length > 0);
  if (latest) renderCanary(latest.canary_transcriptions);
}

// Canary sentences

const PUNCTUATION = /[.,!?؟،؛:«»"'()\-]/g;

// Word-level LCS diff. Reference words the model missed are dotted; model words that don't match are marked.
function diffWords(refText, predText) {
  const a = refText.split(/\s+/).filter(Boolean);
  const b = predText.split(/\s+/).filter(Boolean);
  const na = a.map((w) => w.replace(PUNCTUATION, ""));
  const nb = b.map((w) => w.replace(PUNCTUATION, ""));
  const n = a.length;
  const m = b.length;

  const lcs = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = na[i] === nb[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const refMatched = new Array(n).fill(false);
  const predMatched = new Array(m).fill(false);
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (na[i] === nb[j]) {
      refMatched[i] = true;
      predMatched[j] = true;
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }

  const render = (words, matched, cls) =>
    words
      .map((w, k) => (matched[k] ? escapeHtml(w) : `<mark class="${cls}">${escapeHtml(w)}</mark>`))
      .join(" ");

  return {
    ref: render(a, refMatched, "gone"),
    pred: render(b, predMatched, "wrong"),
  };
}

function renderCanary(canaries) {
  if (!Array.isArray(canaries) || canaries.length === 0) return;
  el.canaryContainer.innerHTML = canaries
    .map((c) => {
      const ref = c.ref || "";
      const diff = diffWords(ref, c.pred || "");
      const pred = c.pred ? diff.pred : `<span class="pending">Pending</span>`;
      return `
        <article class="canary-row">
          <div class="canary-meta">
            <span class="canary-kind">${c.is_informal ? "Informal" : "Formal"}</span>
            <span class="canary-category">${escapeHtml(c.category || "Canary")}</span>
          </div>
          <div class="canary-lines">
            <p class="canary-ref" dir="rtl">${c.pred ? diff.ref : escapeHtml(ref)}</p>
            <p class="canary-pred" dir="rtl">${pred}</p>
          </div>
        </article>`;
    })
    .join("");
}

// Training log

async function fetchLogs() {
  const target = viewingHistorical && selectedRunId !== "latest" ? selectedRunId : null;
  try {
    const res = await fetch(runLogUrl(selectedRunId, viewingHistorical));
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data.logs)) return;

    // A selection change while this request was in flight must not overwrite
    // the newly selected run's log.
    const current = viewingHistorical && selectedRunId !== "latest" ? selectedRunId : null;
    if (target !== current) return;

    const box = el.consoleBox;
    const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 24;

    box.innerHTML = data.logs
      .map((line) => {
        let cls = "log-line";
        if (line.includes("Error") || line.includes("Traceback")) cls += " is-error";
        if (line.includes("Benchmark saved") || line.includes("Epoch")) cls += " is-success";
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      })
      .join("");

    if (pinned) box.scrollTop = box.scrollHeight;
  } catch (e) {
    // Network hiccup: the next poll will retry.
  }
}

// Controls

function setSmokeResult(kind, text, logLines = []) {
  if (!el.smokeResult || !el.smokeStatus) return;
  el.smokeResult.hidden = kind === "idle";
  el.smokeResult.className = `smoke-result ${kind}`;
  el.smokeStatus.textContent = text;
  const hasLog = logLines.length > 0;
  if (el.smokeLogWrap) el.smokeLogWrap.hidden = !hasLog;
  if (el.smokeLog) el.smokeLog.textContent = hasLog ? logLines.join("\n") : "";
  if (hasLog && el.smokeLogWrap) el.smokeLogWrap.open = kind === "fail";
}

function scheduleSmokePoll(delay = 2000) {
  if (smokePollTimer) clearTimeout(smokePollTimer);
  smokePollTimer = setTimeout(pollSmokeStatus, delay);
}

async function pollSmokeStatus() {
  if (!el.smokeResult) return;
  if (smokePollTimer) {
    clearTimeout(smokePollTimer);
    smokePollTimer = null;
  }
  try {
    const res = await fetch("/api/train/smoke/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "running") {
      smokeRunning = true;
      setSmokeResult("running", "Smoke test running — 200 train / 50 eval samples.");
      syncRunButtons();
      scheduleSmokePoll();
      return;
    }
    smokeRunning = false;
    if (data.status === "passed") {
      setSmokeResult("pass", "Smoke test passed. Ready to start the full run.");
    } else if (data.status === "failed") {
      const exit = data.exit_code != null ? ` (exit ${data.exit_code})` : "";
      const reasons = {
        nonzero_exit: "Trainer exited non-zero.",
        missing_adapter: "Adapter was not saved.",
        ended_without_result: "Run ended without a result.",
      };
      const reason = reasons[data.reason] ? ` ${reasons[data.reason]}` : "";
      setSmokeResult("fail", `Smoke test failed${exit}.${reason}`, data.log_tail || []);
    } else {
      setSmokeResult("idle", "");
    }
    syncRunButtons();
    pollTelemetryFallback();
  } catch (e) {
    smokeRunning = true;
    setSmokeResult("running", "Smoke status unavailable — retrying…");
    syncRunButtons();
    scheduleSmokePoll(5000);
  }
}

async function startSmokeRun() {
  if (!el.btnSmoke || !el.smokeResult || smokeRunning) return;
  smokeRunning = true;
  syncRunButtons();
  setSmokeResult("running", "Smoke test running — 200 train / 50 eval samples.");
  try {
    const res = await fetch("/api/train/smoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_size: parseInt(el.inputBatchSize.value, 10) || 8,
        grad_accum: parseInt(el.inputGradAccum.value, 10) || 4,
        learning_rate: parseFloat(el.inputLearningRate.value) || 1e-4,
        gradient_checkpointing: el.inputGradCheckpoint.value === "on",
      }),
    });
    if (!res.ok) {
      const text = await res.text();
      let detail = text;
      try {
        const err = JSON.parse(text);
        detail = err.detail || err.error || text || "Unknown error";
      } catch {
        // Keep raw text.
      }
      smokeRunning = false;
      setSmokeResult("fail", `Could not start smoke test: ${detail}`);
      syncRunButtons();
      pollTelemetryFallback();
      return;
    }
    scheduleSmokePoll(0);
  } catch (e) {
    smokeRunning = false;
    setSmokeResult("fail", `Could not start smoke test: ${e}`);
    syncRunButtons();
    pollTelemetryFallback();
  }
}

async function startRun(freshStart) {
  el.btnStart.disabled = true;
  el.btnFresh.disabled = true;
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: parseInt(el.inputEpochs.value, 10) || DEFAULT_EPOCHS,
        telephony_mode: el.inputTelephonyMode ? el.inputTelephonyMode.value : "paired",
        batch_size: parseInt(el.inputBatchSize.value, 10) || 8,
        grad_accum: parseInt(el.inputGradAccum.value, 10) || 4,
        gradient_checkpointing: el.inputGradCheckpoint.value === "on",
        eval_steps: parseInt(el.inputEvalSteps.value, 10) || 1000,
        learning_rate: parseFloat(el.inputLearningRate.value) || 1e-4,
        fresh_start: freshStart,
      }),
    });
    if (!res.ok) {
      alert(`Start failed: ${await errorDetail(res)}`);
      el.btnStart.disabled = false;
      el.btnFresh.disabled = false;
    } else {
      // A new run is now the live target, even if a historical run was pinned.
      await refreshRunList();
      await selectRun("latest");
    }
  } catch (e) {
    alert(`Failed to start training: ${e}`);
    el.btnStart.disabled = false;
    el.btnFresh.disabled = false;
  }
}

el.btnStart.addEventListener("click", () => startRun(false));
if (el.btnSmoke) el.btnSmoke.addEventListener("click", startSmokeRun);
if (el.runSelect) {
  el.runSelect.addEventListener("change", () => selectRun(el.runSelect.value));
}
if (el.inputTelephonyMode) {
  el.inputTelephonyMode.addEventListener("change", () => {
    el.inputEpochs.value = epochsForTelephonyMode(el.inputTelephonyMode.value);
  });
}
if (el.btnResume) el.btnResume.addEventListener("click", resumeRun);
if (el.btnMerge) el.btnMerge.addEventListener("click", mergeRun);
if (el.btnRecover) el.btnRecover.addEventListener("click", () => openRecoverDialog());
if (el.btnHfClose) el.btnHfClose.addEventListener("click", closeRecoverDialog);
if (el.btnHfList) el.btnHfList.addEventListener("click", loadRemoteRuns);
if (el.btnHfRestore) el.btnHfRestore.addEventListener("click", () => startRestore(false));
if (el.btnHfRestoreResume) {
  el.btnHfRestoreResume.addEventListener("click", () => startRestore(true));
}
if (el.btnHfOpenLocal) el.btnHfOpenLocal.addEventListener("click", openHfLocalRun);
if (el.inputHfRunId) {
  el.inputHfRunId.addEventListener("input", () => {
    syncHfRestoreButtons();
    const { run, state } = hfRestoreSelection();
    if (run && state.reason) setHfRestoreStatus(state.reason);
  });
}
syncHfRestoreButtons();

el.btnFresh.addEventListener("click", () => {
  if (!confirm("Fresh start deletes existing checkpoints, the best adapter, reports and telemetry history before training. This cannot be undone. Continue?")) {
    return;
  }
  startRun(true);
});

el.btnStop.addEventListener("click", async () => {
  if (!confirm("Halt the training run? A checkpoint will be saved first.")) {
    return;
  }
  el.btnStop.disabled = true;
  try {
    await fetch("/api/train/stop", { method: "POST" });
  } catch (e) {
    console.error("Halt failed", e);
  }
});

// Init

window.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
  pollTelemetryFallback();
  setInterval(pollTelemetryFallback, 1000);
  refreshRunList();
  loadRunState();
  setInterval(() => {
    if (viewingHistorical) {
      // Only the picker and the live Start/Halt controls follow the ledger
      // while a historical run is pinned; the rendered content stays historical.
      refreshRunList();
      pollLiveControls();
      return;
    }
    if (!wsActive) loadRunState();
  }, 2000);
  fetchLogs();
  setInterval(fetchLogs, 2500);
  pollSmokeStatus();
  loadHfConfig();
  restoreHfProgressOnLoad();
  new ResizeObserver(() => drawLoss()).observe(el.svgChart);
});
