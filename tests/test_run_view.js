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
  false,
  "only completed runs may merge",
);
assert.strictEqual(
  isMergeAvailable({ ...mergeable, status: "failed" }, true),
  false,
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

console.log("run_view behavior tests passed");
