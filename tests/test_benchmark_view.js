/**
 * Behavior tests for server/static/benchmark_view.js.
 * Run with: node tests/test_benchmark_view.js
 */

const assert = require("node:assert");
const {
  benchmarkComparable,
  computeBenchmarkScale,
  findBuzzBaseline,
  isBestBenchmarkRow,
} = require("../server/static/benchmark_view.js");

const baseline = {
  epoch: "BuzzASR Base (preflight, 1,000)",
  is_baseline: true,
  total_samples: 1000,
  normalized_wer: 17.2,
  phone_normalized_wer: 24.1,
};
const list = [
  baseline,
  {
    epoch: "Whisper (preflight)",
    is_baseline: true,
    is_reference: true,
    total_samples: 1000,
    normalized_wer: 28.5,
  },
  { epoch: "Epoch 1", total_samples: 1000, normalized_wer: 15.0 },
  { epoch: "Legacy 300-row run", total_samples: 300, normalized_wer: 5.0 },
  { epoch: "Short smoke run", total_samples: 300, normalized_wer: 99.0 },
];

assert.strictEqual(findBuzzBaseline(list), baseline);
assert.strictEqual(benchmarkComparable(list[2], baseline), true);
assert.strictEqual(benchmarkComparable(list[3], baseline), false);

const gated = computeBenchmarkScale(list, baseline);
assert.strictEqual(gated.best, 15.0, "best must ignore non-comparable rows");
assert.ok(
  gated.scaleMax > 28.5 && gated.scaleMax < 40,
  `scaleMax must ignore the 99% incomparable row (got ${gated.scaleMax})`
);
assert.strictEqual(isBestBenchmarkRow(list[2], gated.best, baseline), true);
assert.strictEqual(
  isBestBenchmarkRow({ ...list[2], total_samples: 300 }, gated.best, baseline),
  false,
  "an equal-WER row from a different sample count must not be tagged Best"
);

const ungated = computeBenchmarkScale(list, null);
assert.strictEqual(
  ungated.best,
  5.0,
  "without a measured baseline every checkpoint row remains eligible"
);
assert.strictEqual(isBestBenchmarkRow(list[3], ungated.best, null), true);

const empty = computeBenchmarkScale([], baseline);
assert.strictEqual(empty.best, null);
assert.strictEqual(empty.scaleMax, 1.08);

console.log("benchmark_view behavior tests passed");
