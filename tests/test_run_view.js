/**
 * Behavior tests for server/static/run_view.js.
 * Run with: node tests/test_run_view.js
 */

const assert = require("node:assert");
const {
  STAGE_ORDER,
  stageMap,
  stageLabel,
  runStatusLabel,
  savedStepsLabel,
  isMergeAvailable,
  runLogUrl,
  formValuesFromConfig,
  epochsForTelephonyMode,
  runViewState,
  resolvePickerSelection,
  shouldApplyPickerRefresh,
} = require("../server/static/run_view.js");

assert.deepStrictEqual(STAGE_ORDER, ["preflight", "training", "post_eval", "merge"]);

const run = {
  status: "completed",
  stages: [
    { name: "preflight", status: "completed" },
    { name: "training", status: "completed" },
    { name: "post_eval", status: "completed" },
    { name: "merge", status: "pending" },
  ],
};
assert.strictEqual(stageMap(run).get("post_eval").status, "completed");
assert.strictEqual(runStatusLabel(run), "Run completed");
assert.strictEqual(savedStepsLabel({ count: 1234 }), "1,234 steps saved");
assert.strictEqual(isMergeAvailable(run, true), true);
assert.strictEqual(isMergeAvailable({ ...run, status: "running" }, true), false);
assert.strictEqual(isMergeAvailable({ ...run, status: "starting" }, true), false);
assert.strictEqual(isMergeAvailable(run, false), false);

// Missing stages and stage metadata normalize to pending.
const partial = stageMap({
  stages: [{ name: "training", status: "running", detail: "step 40" }],
});
assert.strictEqual(partial.size, STAGE_ORDER.length);
assert.strictEqual(partial.get("training").status, "running");
assert.strictEqual(partial.get("training").detail, "step 40");
for (const name of ["preflight", "post_eval", "merge"]) {
  assert.strictEqual(partial.get(name).status, "pending", `${name} must default to pending`);
}
assert.strictEqual(stageMap(null).get("preflight").status, "pending");
assert.strictEqual(stageMap({ stages: "not-a-list" }).get("merge").status, "pending");

assert.strictEqual(stageLabel("preflight"), "Preflight");
assert.strictEqual(stageLabel("training"), "Training");
assert.strictEqual(stageLabel("post_eval"), "Post-eval");
assert.strictEqual(stageLabel("merge"), "Merge");
assert.strictEqual(stageLabel("unknown_stage"), "Unknown stage");

for (const [status, label] of [
  ["starting", "Run starting"],
  ["running", "Run running"],
  ["completed", "Run completed"],
  ["stopped", "Run stopped"],
  ["failed", "Run failed"],
  ["interrupted", "Run interrupted"],
]) {
  assert.strictEqual(runStatusLabel({ status }), label);
}
assert.strictEqual(runStatusLabel({}), "Run unknown");

assert.strictEqual(savedStepsLabel({ count: 0 }), "0 steps saved");
assert.strictEqual(savedStepsLabel({ count: 1 }), "1 step saved");
assert.strictEqual(savedStepsLabel({ count: "2500" }), "2,500 steps saved");
assert.strictEqual(savedStepsLabel({ count: "not-a-number" }), "0 steps saved");
assert.strictEqual(savedStepsLabel(null), "0 steps saved");
const lastSavedAt = 1761846420;
const lastSavedLabel = savedStepsLabel({ count: 12, last_saved_at: lastSavedAt });
const expectedLastAt = new Date(lastSavedAt * 1000).toLocaleTimeString([], {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});
assert.strictEqual(lastSavedLabel, `12 steps saved · last at ${expectedLastAt}`);
assert.strictEqual(
  savedStepsLabel({ count: 12, last_saved_at: "not-a-time" }),
  "12 steps saved",
);
assert.strictEqual(
  savedStepsLabel({ count: 12, last_saved_at: null }),
  "12 steps saved",
);

const mergeable = {
  status: "completed",
  stages: [
    { name: "preflight", status: "completed" },
    { name: "training", status: "completed" },
    { name: "post_eval", status: "completed" },
    { name: "merge", status: "pending" },
  ],
};
assert.strictEqual(isMergeAvailable(mergeable, true), true);
assert.strictEqual(isMergeAvailable(mergeable, "yes"), true);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, stages: mergeable.stages.slice(0, 2) }, true),
  true,
  "missing post_eval/merge stages normalize to pending",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "interrupted" }, true),
  true,
  "training completion is what matters, not the overall run status",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "failed" }, true),
  true,
  "a failed post-eval must not hide a completed training adapter",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "stopped" }, true),
  true,
  "a stopped post-eval must not hide a completed training adapter",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "starting" }, true),
  false,
  "a starting run must not offer merge",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "running" }, true),
  false,
  "a running run must not offer merge",
);
assert.strictEqual(
  isMergeAvailable(
    {
      ...mergeable,
      stages: mergeable.stages.map((stage) =>
        stage.name === "training" ? { ...stage, status: "pending" } : stage,
      ),
    },
    true,
  ),
  false,
  "merge requires training to have completed",
);
assert.strictEqual(
  isMergeAvailable(
    {
      ...mergeable,
      stages: mergeable.stages.map((stage) =>
        stage.name === "training" ? { ...stage, status: "failed" } : stage,
      ),
    },
    true,
  ),
  false,
  "merge requires the training stage to be completed",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "completed" }, 0),
  false,
  "merge requires an adapter artifact",
);
assert.strictEqual(
  isMergeAvailable(
    {
      ...mergeable,
      stages: mergeable.stages.map((stage) =>
        stage.name === "merge" ? { ...stage, status: "completed" } : stage,
      ),
    },
    true,
  ),
  false,
  "an already-merged run must not offer merge again",
);
assert.strictEqual(
  isMergeAvailable(
    {
      ...mergeable,
      stages: mergeable.stages.map((stage) =>
        stage.name === "training" ? { ...stage, status: "running" } : stage,
      ),
    },
    true,
  ),
  false,
  "a running stage blocks merge",
);

// Task 8 action gates: merge needs an artifact and interrupted runs read clearly.
assert.strictEqual(
  isMergeAvailable(
    { status: "completed", stages: [{ name: "training", status: "completed" }] },
    false,
  ),
  false,
  "merge requires an adapter artifact",
);
assert.strictEqual(runStatusLabel({ status: "interrupted" }), "Run interrupted");

// Historical selection must read that run's log, never the live buffer.
assert.strictEqual(
  runLogUrl("20261031-101500-aaaaaa", true),
  "/api/runs/20261031-101500-aaaaaa/logs?tail=80",
);
assert.strictEqual(runLogUrl("latest", true), "/api/logs?tail=80");
assert.strictEqual(runLogUrl("20261031-101500-aaaaaa", false), "/api/logs?tail=80");
assert.strictEqual(runLogUrl(null, true), "/api/logs?tail=80");
assert.strictEqual(
  runLogUrl("run/with space", true),
  "/api/runs/run%2Fwith%20space/logs?tail=80",
);

// formValuesFromConfig maps a run config onto form field values.
const fullConfig = {
  epochs: 4,
  batch_size: 8,
  grad_accum: 4,
  eval_steps: 1000,
  learning_rate: 0.0001,
  gradient_checkpointing: true,
};
assert.deepStrictEqual(formValuesFromConfig(fullConfig), {
  epochs: "4",
  batch_size: "8",
  grad_accum: "4",
  eval_steps: "1000",
  learning_rate: "0.0001",
  gradient_checkpointing: "on",
});
assert.strictEqual(
  formValuesFromConfig({ ...fullConfig, gradient_checkpointing: false }).gradient_checkpointing,
  "off",
);
for (const truthy of ["1", "on", "true", "yes", "ON", "True", 1, true]) {
  assert.strictEqual(
    formValuesFromConfig({ gradient_checkpointing: truthy }).gradient_checkpointing,
    "on",
    `must treat ${JSON.stringify(truthy)} as on`,
  );
}
for (const falsy of ["0", "off", "false", "no", "", 0, false, null]) {
  assert.strictEqual(
    formValuesFromConfig({ gradient_checkpointing: falsy }).gradient_checkpointing,
    "off",
    `must treat ${JSON.stringify(falsy)} as off`,
  );
}

// Missing keys stay missing so the form keeps its defaults.
assert.deepStrictEqual(formValuesFromConfig({}), {});
assert.deepStrictEqual(formValuesFromConfig({ epochs: 2 }), { epochs: "2" });
assert.deepStrictEqual(formValuesFromConfig({ batch_size: null, epochs: "" }), {});

// Extra keys never leak into the form mapping.
assert.deepStrictEqual(
  formValuesFromConfig({ epochs: 3, trainer_args: { lora_r: 64 }, unknown_key: "x" }),
  { epochs: "3" },
);

// Non-object configs map to nothing.
assert.deepStrictEqual(formValuesFromConfig(null), {});
assert.deepStrictEqual(formValuesFromConfig(undefined), {});
assert.deepStrictEqual(formValuesFromConfig("epochs=4"), {});

// Training-data mode maps through and drives the epoch default: paired uses
// two epochs, disjoint/clean use four, and a missing mode behaves as paired.
assert.strictEqual(epochsForTelephonyMode("paired"), 2);
assert.strictEqual(epochsForTelephonyMode("disjoint"), 4);
assert.strictEqual(epochsForTelephonyMode("clean"), 4);
assert.strictEqual(epochsForTelephonyMode(undefined), 2);
assert.strictEqual(epochsForTelephonyMode(null), 2);
assert.strictEqual(formValuesFromConfig({ telephony_mode: "clean" }).telephony_mode, "clean");
assert.strictEqual(
  formValuesFromConfig({ ...fullConfig, telephony_mode: "clean" }).telephony_mode,
  "clean",
);

// runViewState drives Resume/Merge visibility in renderRun.
const resumable = (status) =>
  runViewState(
    {
      status,
      stages: [
        { name: "training", status: "completed" },
        { name: "post_eval", status: "completed" },
        { name: "merge", status: "pending" },
      ],
    },
    true,
  );
for (const status of ["interrupted", "stopped", "failed"]) {
  assert.strictEqual(
    resumable(status).resumeVisible,
    true,
    `${status} must offer Resume`,
  );
}
for (const status of ["starting", "running", "completed", undefined, "unknown"]) {
  assert.strictEqual(
    runViewState({ status, stages: [] }, false).resumeVisible,
    false,
    `${status} must hide Resume`,
  );
}
assert.deepStrictEqual(runViewState(null, true), {
  resumeVisible: false,
  mergeVisible: false,
});
assert.strictEqual(
  runViewState({ status: "completed", adapter_exists: false }, false).mergeVisible,
  false,
  "merge hides without an adapter artifact",
);
assert.strictEqual(
  resumable("completed").mergeVisible,
  true,
  "a completed run with an adapter offers Merge",
);
assert.strictEqual(
  resumable("failed").mergeVisible,
  true,
  "a failed post-eval keeps Merge when training completed",
);
assert.strictEqual(
  runViewState({ status: "running", stages: [{ name: "training", status: "completed" }] }, true)
    .mergeVisible,
  false,
  "a running run never offers Merge",
);
assert.strictEqual(
  runViewState({ status: "completed", stages: [] }, false).mergeVisible,
  false,
  "merge hides when training did not complete",
);

// Picker race: a stale /api/runs response must not clobber an in-flight selection.
assert.strictEqual(shouldApplyPickerRefresh("latest", "latest"), true);
assert.strictEqual(shouldApplyPickerRefresh("20261031-101500-aaaaaa", "20261031-101500-aaaaaa"), true);
assert.strictEqual(shouldApplyPickerRefresh("latest", "20261031-101500-aaaaaa"), false);
assert.strictEqual(shouldApplyPickerRefresh("20261031-101500-aaaaaa", "latest"), false);
assert.strictEqual(shouldApplyPickerRefresh("run-a", "run-b"), false);
assert.strictEqual(resolvePickerSelection("latest", [{ run_id: "run-a" }]), "latest");
assert.strictEqual(
  resolvePickerSelection("run-a", [{ run_id: "run-a" }, { run_id: "run-b" }]),
  "run-a",
);
assert.strictEqual(resolvePickerSelection("run-gone", [{ run_id: "run-a" }]), "latest");
assert.strictEqual(resolvePickerSelection("run-a", []), "latest");
assert.strictEqual(resolvePickerSelection("run-a", null), "latest");

console.log("run_view behavior tests passed");
