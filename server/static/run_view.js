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
  for (const key of ["epochs", "batch_size", "grad_accum", "eval_steps", "learning_rate"]) {
    const value = config[key];
    if (value != null && value !== "") values[key] = String(value);
  }
  if (Object.prototype.hasOwnProperty.call(config, "gradient_checkpointing")) {
    values.gradient_checkpointing = truthyFlagValue(config.gradient_checkpointing)
      ? "on"
      : "off";
  }
  return values;
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
    resolvePickerSelection,
    shouldApplyPickerRefresh,
  };
}
