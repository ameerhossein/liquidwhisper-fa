/**
 * Persian ASR training console.
 * Live telemetry over WebSocket with HTTP polling fallback, loss curve,
 * epoch benchmark ladder, canary word diffs, and the training log.
 */

const DEFAULT_EPOCHS = 4;
const WHISPER_WER_FALLBACK = 28.5;
const BUZZASR_WER_FALLBACK = 17.2;
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
  inputEpochs: document.getElementById("inputEpochs"),
  inputBatchSize: document.getElementById("inputBatchSize"),
  inputLearningRate: document.getElementById("inputLearningRate"),
  benchmarkTbody: document.getElementById("benchmarkTbody"),
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
  if (Math.abs(value) < 0.005) return `<span class="delta flat">0.00 pts</span>`;
  const sign = value < 0 ? "−" : "+";
  const cls = value < 0 ? "good" : "bad";
  return `<span class="delta ${cls}">${sign}${Math.abs(value).toFixed(2)} pts</span>`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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
        if (data.telemetry) updateTelemetry(data.telemetry);
        if (data.benchmarks) updateBenchmarks(data.benchmarks);
        if (data.loss_history) updateLossHistory(data.loss_history);
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
  if (wsActive) return;
  try {
    const [statusRes, benchRes, lossRes] = await Promise.all([
      fetch("/api/status").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/benchmarks").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/loss-history").then((r) => (r.ok ? r.json() : null)),
    ]);
    if (statusRes) updateTelemetry(statusRes);
    if (benchRes) updateBenchmarks(benchRes);
    if (lossRes) updateLossHistory(lossRes);
  } catch (e) {
    // Network hiccup: the next poll will retry.
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

function updateTelemetry(tel) {
  const status = tel.status || "idle";
  const evaluating = status === "evaluating_benchmark";
  const running = !!tel.process_running || status === "training" || evaluating;

  const state = evaluating ? "evaluating" : running ? "training" : status === "completed" ? "completed" : "idle";
  const labels = {
    evaluating: "Evaluating on held-out test",
    training: "Training",
    completed: "Run completed",
    idle: "Idle",
  };
  el.statusDot.className = `dot dot-${state}`;
  el.statusLabel.textContent = labels[state];

  el.btnStart.disabled = running;
  el.btnFresh.disabled = running;
  el.btnStop.disabled = !running;

  const gpu = tel.gpu || {};
  if (gpu.available) {
    el.gpuModel.textContent = gpu.device_name || "NVIDIA GPU";
    el.vramText.textContent = `${gpu.allocated_gb || 0} / ${gpu.total_gb || 0} GB`;
    el.vramFill.style.width = `${gpu.utilization_pct || 0}%`;
  } else {
    el.gpuModel.textContent = "GPU unavailable";
    el.vramText.textContent = "—";
    el.vramFill.style.width = "0%";
  }

  const step = tel.step || 0;
  const maxSteps = tel.max_steps || 0;
  const epoch = tel.epoch || 0;
  const total = totalEpochs(tel);
  const started = running || step > 0 || status === "completed";
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

function findBaseline(list, pattern, fallback) {
  const row = list.find((r) => r.is_baseline && pattern.test(r.epoch || ""));
  return row && row.normalized_wer != null ? Number(row.normalized_wer) : fallback;
}

function updateBenchmarks(list) {
  if (!Array.isArray(list) || list.length === 0) return;

  const checkpoints = list.filter((r) => !r.is_baseline && r.normalized_wer != null);
  const best = checkpoints.length ? Math.min(...checkpoints.map((r) => Number(r.normalized_wer))) : null;
  const whisper = findBaseline(list, /Whisper/i, WHISPER_WER_FALLBACK);
  const buzz = findBaseline(list, /BuzzASR/i, BUZZASR_WER_FALLBACK);
  const scaleMax = Math.max(whisper, ...list.map((r) => Number(r.normalized_wer) || 0)) * 1.08;
  const share = (v) => `${Math.min(100, Math.max(0, (v / scaleMax) * 100)).toFixed(2)}%`;

  el.benchmarkTbody.innerHTML = list
    .map((row) => {
      const wer = Number(row.normalized_wer) || 0;
      const isBaseline = !!row.is_baseline;
      const isBest = !isBaseline && best !== null && Math.abs(wer - best) < 0.001;
      const isWhisper = /Whisper/i.test(row.epoch || "");
      const isBuzz = /BuzzASR/i.test(row.epoch || "");

      const rowClass = isBaseline ? "is-baseline" : isBest ? "is-best" : "";
      const barClass = isBaseline ? "is-base" : isBest ? "is-best" : "";
      const tag = isBaseline
        ? `<span class="tag">Baseline</span>`
        : isBest
          ? `<span class="tag tag-best">Best</span>`
          : "";

      return `
        <tr class="${rowClass}">
          <th scope="row"><span class="row-name">${escapeHtml(row.epoch || "Checkpoint")}</span>${tag}</th>
          <td class="wer">
            <div class="wer-inner">
              <div class="bar ${barClass}">
                <span class="bar-fill" style="width:${share(wer)}"></span>
                <span class="bar-ref" style="left:${share(buzz)}"></span>
              </div>
              <span class="wer-value">${wer.toFixed(2)}%</span>
            </div>
          </td>
          <td class="num">${isWhisper ? formatDelta(null) : formatDelta(wer - whisper)}</td>
          <td class="num">${isBuzz ? formatDelta(null) : formatDelta(wer - buzz)}</td>
          <td class="num">${formatPct(row.phone_normalized_wer)}</td>
          <td class="num opt">${formatPct(row.normalized_cer)}</td>
          <td class="num opt">${formatPct(row.formal_wer)}</td>
          <td class="num opt">${formatPct(row.informal_wer)}</td>
          <td class="num opt">${formatPct(row.diagnostic_wer)}</td>
        </tr>`;
    })
    .join("");

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
  try {
    const res = await fetch("/api/logs?tail=80");
    const data = await res.json();
    if (!Array.isArray(data.logs)) return;

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

async function startRun(freshStart) {
  el.btnStart.disabled = true;
  el.btnFresh.disabled = true;
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: parseInt(el.inputEpochs.value, 10) || DEFAULT_EPOCHS,
        batch_size: parseInt(el.inputBatchSize.value, 10) || 16,
        learning_rate: parseFloat(el.inputLearningRate.value) || 1e-4,
        fresh_start: freshStart,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Start failed: ${err.detail || "Unknown error"}`);
      el.btnStart.disabled = false;
      el.btnFresh.disabled = false;
    }
  } catch (e) {
    alert(`Failed to start training: ${e}`);
    el.btnStart.disabled = false;
    el.btnFresh.disabled = false;
  }
}

el.btnStart.addEventListener("click", () => startRun(false));

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
  fetchLogs();
  setInterval(fetchLogs, 2500);
  new ResizeObserver(() => drawLoss()).observe(el.svgChart);
});
