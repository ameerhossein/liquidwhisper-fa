/**
 * Pure benchmark-selection helpers for the training cockpit.
 *
 * Loaded before app.js in the browser and importable from Node for behavior
 * tests. Keeping the comparable-row gating here prevents a legacy/smoke row
 * with a different sample count from being tagged "Best" or stretching the
 * WER bars.
 */

function benchmarkSampleCount(row) {
  if (!row || row.total_samples == null) return null;
  const count = Number(row.total_samples);
  return Number.isFinite(count) ? count : null;
}

function findBuzzBaseline(list) {
  if (!Array.isArray(list)) return null;
  return (
    list.find((r) => r.is_baseline && !r.is_reference && /BuzzASR/i.test(r.epoch || "")) ||
    list.find((r) => r.is_baseline && !r.is_reference) ||
    null
  );
}

function benchmarkComparable(row, baseline) {
  const rowCount = benchmarkSampleCount(row);
  const baseCount = benchmarkSampleCount(baseline);
  return baseline != null && rowCount != null && baseCount != null && rowCount === baseCount;
}

function computeBenchmarkScale(list, baseline) {
  const rows = (Array.isArray(list) ? list : []).filter(
    (row) =>
      row &&
      row.normalized_wer != null &&
      !isNaN(Number(row.normalized_wer)) &&
      (baseline == null || benchmarkComparable(row, baseline))
  );
  const checkpointRows = rows.filter((row) => !row.is_baseline);
  return {
    best: checkpointRows.length
      ? Math.min(...checkpointRows.map((row) => Number(row.normalized_wer)))
      : null,
    scaleMax: Math.max(1, ...rows.map((row) => Number(row.normalized_wer) || 0)) * 1.08,
  };
}

function isBestBenchmarkRow(row, best, baseline) {
  if (!row || row.is_baseline || best == null) return false;
  if (row.normalized_wer == null || isNaN(Number(row.normalized_wer))) return false;
  if (baseline != null && !benchmarkComparable(row, baseline)) return false;
  return Math.abs(Number(row.normalized_wer) - best) < 0.001;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    benchmarkSampleCount,
    findBuzzBaseline,
    benchmarkComparable,
    computeBenchmarkScale,
    isBestBenchmarkRow,
  };
}
