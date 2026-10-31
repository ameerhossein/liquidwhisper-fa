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
  return `${count.toLocaleString("en-US")} step${count === 1 ? "" : "s"} saved`;
}

function runLogUrl(runId, historical = false) {
  const id = runId == null ? "" : String(runId);
  if (historical && id && id !== "latest") {
    return `/api/runs/${encodeURIComponent(id)}/logs?tail=80`;
  }
  return "/api/logs?tail=80";
}

function isMergeAvailable(run, adapterExists = false) {
  if (!run || !adapterExists || run.status !== "completed") return false;
  const stages = stageMap(run);
  if (stages.get("merge").status === "completed") return false;
  for (const stage of stages.values()) {
    if (stage.status === "running") return false;
  }
  return true;
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
  };
}
