/**
 * Pure run-provenance helpers for the training cockpit.
 *
 * Loaded before app.js in the browser and importable from Node for behavior
 * tests. The run ledger may omit stages from older manifests, so every view
 * reads through stageMap() to get the same four-stage rail in a known order.
 */

const STAGE_ORDER = ["preflight", "training", "post_eval", "merge"];

const STAGE_LABELS = {
  preflight: "Preflight",
  training: "Training",
  post_eval: "Post-eval",
  merge: "Merge",
};

const RUN_STATUS_LABELS = {
  starting: "Run starting",
  running: "Run running",
  completed: "Run completed",
  stopped: "Run stopped",
  failed: "Run failed",
  interrupted: "Run interrupted",
};

function stageMap(run) {
  const map = new Map();
  for (const name of STAGE_ORDER) {
    map.set(name, { name, status: "pending" });
  }
  const stages = run && Array.isArray(run.stages) ? run.stages : [];
  for (const stage of stages) {
    if (!stage || typeof stage !== "object") continue;
    const name = stage.name;
    if (!STAGE_ORDER.includes(name)) continue;
    map.set(name, { ...stage, name, status: stage.status || "pending" });
  }
  return map;
}

function stageLabel(name) {
  if (STAGE_LABELS[name]) return STAGE_LABELS[name];
  if (!name) return "Unknown stage";
  const words = String(name).replace(/_/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Unknown stage";
}

function runStatusLabel(run) {
  const status = run && run.status ? String(run.status) : "";
  return RUN_STATUS_LABELS[status] || `Run ${status || "unknown"}`;
}

function savedStepsLabel(steps) {
  let count = steps && steps.count != null ? Number(steps.count) : 0;
  if (!Number.isFinite(count) || count < 0) count = 0;
  count = Math.floor(count);
  const label = `${count.toLocaleString("en-US")} step${count === 1 ? "" : "s"} saved`;
  if (!steps || steps.last_saved_at == null) return label;
  const seconds = Number(steps.last_saved_at);
  if (!Number.isFinite(seconds) || seconds <= 0) return label;
  const date = new Date(seconds * 1000);
  if (isNaN(date.getTime())) return label;
  const time = date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return `${label} · last at ${time}`;
}

function runLogUrl(runId, historical = false) {
  const id = runId == null ? "" : String(runId);
  if (historical && id && id !== "latest") {
    return `/api/runs/${encodeURIComponent(id)}/logs?tail=80`;
  }
  return "/api/logs?tail=80";
}

function isMergeAvailable(run, adapterExists = false) {
  if (!run || !adapterExists) return false;
  // The server blocks merge while the run is still starting/running, even if
  // its training stage already completed (R12).
  if (run.status === "starting" || run.status === "running") return false;
  const stages = stageMap(run);
  if (stages.get("training").status !== "completed") return false;
  if (stages.get("merge").status === "completed") return false;
  for (const stage of stages.values()) {
    if (stage.status === "running") return false;
  }
  return true;
}

// Resume/Merge visibility for renderRun(). Pure so Node can pin the gates:
// Resume for interrupted/stopped/failed runs, Merge via isMergeAvailable().
function runViewState(run, adapterExists = false) {
  if (!run || typeof run !== "object") return { resumeVisible: false, mergeVisible: false };
  const resumeVisible = ["interrupted", "stopped", "failed"].includes(run.status);
  const adapterAvailable = Boolean(adapterExists) || Boolean(run.adapter_exists);
  return { resumeVisible, mergeVisible: isMergeAvailable(run, adapterAvailable) };
}

function truthyFlagValue(value) {
  if (typeof value === "string") {
    return ["1", "on", "true", "yes"].includes(value.toLowerCase());
  }
  return Boolean(value);
}

// Map a run config onto form field values. Missing keys stay missing so the
// form keeps its defaults; unknown keys never leak into the form.
function formValuesFromConfig(config) {
  if (!config || typeof config !== "object") return {};
  const values = {};
  for (const key of [
    "epochs",
    "batch_size",
    "grad_accum",
    "eval_steps",
    "learning_rate",
    "telephony_mode",
  ]) {
    const value = config[key];
    if (value != null && value !== "") values[key] = String(value);
  }
  if (Object.prototype.hasOwnProperty.call(config, "gradient_checkpointing")) {
    values.gradient_checkpointing = truthyFlagValue(config.gradient_checkpointing)
      ? "on"
      : "off";
  }
  // Pre-change run configs have no telephony_mode. Every such run trained
  // disjoint, so map it explicitly instead of leaving the selector on its
  // paired default next to the restored disjoint epoch count.
  if (!values.telephony_mode && Object.keys(values).length > 0) {
    values.telephony_mode = "disjoint";
  }
  return values;
}

// Paired training consumes both clean and phone rows in one pass, so it needs
// half the epochs of the disjoint/clean-only datasets.
function epochsForTelephonyMode(mode) {
  return String(mode || "paired") === "paired" ? 2 : 4;
}

// Pick the picker value after a /api/runs refresh: keep a still-listed
// selection, otherwise fall back to Latest.
function resolvePickerSelection(previous, runs) {
  const list = Array.isArray(runs) ? runs : [];
  if (previous === "latest" || previous == null || previous === "") return "latest";
  const stillListed = list.some(
    (run) => run && run.run_id != null && String(run.run_id) === String(previous),
  );
  return stillListed ? String(previous) : "latest";
}

// A stale /api/runs response must not overwrite an in-flight picker selection:
// apply it only when neither the DOM value nor the selected run id moved.
function shouldApplyPickerRefresh(selectionAtStart, selectionNow) {
  return String(selectionAtStart ?? "latest") === String(selectionNow ?? "latest");
}

// Restore controls for one remote run summary from /api/hf/runs. A completed
// run can still be restored, but its training must not be resumed. A remote
// run without a complete checkpoint is restore-only too; the restore script
// installs the run and reports the missing checkpoint without training.
function recoverActionState(remoteRun) {
  const run = remoteRun && typeof remoteRun === "object" ? remoteRun : {};
  const runId = run.run_id == null ? "" : String(run.run_id).trim();
  if (!runId) {
    return {
      canRestore: false,
      canResume: false,
      reason: "Choose a remote run to restore.",
    };
  }
  const status = String(run.status || "").trim().toLowerCase();
  if (status === "completed") {
    return {
      canRestore: true,
      canResume: false,
      reason: "Run is completed; restoring it will not resume training.",
    };
  }
  if (!truthyFlagValue(run.has_checkpoint)) {
    return {
      canRestore: true,
      canResume: false,
      reason: "No complete checkpoint on the remote; restore only.",
    };
  }
  return { canRestore: true, canResume: true, reason: "" };
}

// Resolve the dialog's current run id against the listed remote runs and gate
// its actions. A typed id with no matching row has no checkpoint metadata, so
// it stays restore-only until the user lists and selects it.
function recoverSelectionState(runId, runs) {
  const id = runId == null ? "" : String(runId).trim();
  const list = Array.isArray(runs) ? runs : [];
  const run = id
    ? list.find((candidate) => candidate && String(candidate.run_id || "") === id) || null
    : null;
  if (run) return { runId: id, run, state: recoverActionState(run) };
  if (!id) {
    return {
      runId: id,
      run: null,
      state: {
        canRestore: false,
        canResume: false,
        reason: "Choose a remote run to restore.",
      },
    };
  }
  return {
    runId: id,
    run: null,
    state: {
      canRestore: true,
      canResume: false,
      reason:
        "List and select this run to verify its checkpoint before resuming.",
    },
  };
}

// The HF sync badge. ``enabled: false`` means sync is off, so there is no
// badge; otherwise reflect uploading, the retrying error state, or the last
// successfully uploaded step with its local time.
function hfSyncLabel(sync) {
  if (!sync || typeof sync !== "object" || sync.enabled === false) return "";
  const status = String(sync.status || "").trim().toLowerCase();
  if (status === "uploading") return "HF: uploading…";
  if (status === "error") return "HF: error — retrying";
  const step = Number(sync.last_success_step);
  if (sync.last_success_step == null || !Number.isFinite(step)) {
    return "HF: waiting for first upload";
  }
  let label = `HF: synced ${step.toLocaleString("en-US")}`;
  const at = Number(sync.last_success_at);
  if (sync.last_success_at != null && Number.isFinite(at) && at > 0) {
    const date = new Date(at * 1000);
    if (!isNaN(date.getTime())) {
      label += ` · ${date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })}`;
    }
  }
  return label;
}

// One row of the recover dialog's remote run list.
function remoteRunOptionLabel(run) {
  const row = run && typeof run === "object" ? run : {};
  const runId = row.run_id == null ? "" : String(row.run_id);
  const status = String(row.status || "unknown");
  const step = Number(row.last_step);
  const stepLabel =
    row.last_step != null && Number.isFinite(step) ? step.toLocaleString("en-US") : "—";
  return `${runId} · ${status} · step ${stepLabel}`;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    STAGE_ORDER,
    stageMap,
    stageLabel,
    runStatusLabel,
    savedStepsLabel,
    isMergeAvailable,
    runLogUrl,
    runViewState,
    formValuesFromConfig,
    epochsForTelephonyMode,
    resolvePickerSelection,
    shouldApplyPickerRefresh,
    recoverActionState,
    recoverSelectionState,
    hfSyncLabel,
    remoteRunOptionLabel,
  };
}
