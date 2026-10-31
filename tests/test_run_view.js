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

console.log("run_view behavior tests passed");
