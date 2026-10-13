/**
 * Persian ASR RunPod Cockpit Frontend Logic
 * Live WebSocket telemetry, responsive SVG loss chart, and reactive benchmark matrix.
 */

// State
let isRunning = false;
let ws = null;
let lossHistory = [];
let benchmarks = [];
let autoScrollLogs = true;

// DOM Elements
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const vramText = document.getElementById("vramText");
const vramFill = document.getElementById("vramFill");
const gpuModel = document.getElementById("gpuModel");

// Controls
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const inputEpochs = document.getElementById("inputEpochs");
const inputBatchSize = document.getElementById("inputBatchSize");
const inputLearningRate = document.getElementById("inputLearningRate");

// Stats
const statStep = document.getElementById("statStep");
const statLoss = document.getElementById("statLoss");
const statLR = document.getElementById("statLR");
const statSpeed = document.getElementById("statSpeed");
const statElapsed = document.getElementById("statElapsed");
const statEta = document.getElementById("statEta");

// Sections
const benchmarkTbody = document.getElementById("benchmarkTbody");
const canaryContainer = document.getElementById("canaryContainer");
const consoleBox = document.getElementById("consoleBox");
const svgChart = document.getElementById("svgChart");

// Formatters
function formatSeconds(secs) {
  if (!secs || secs <= 0) return "--:--";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  const h = Math.floor(m / 60);
  if (h > 0) {
    return `${h}h ${m % 60}m`;
  }
  return `${m}m ${s < 10 ? '0' : ''}${s}s`;
}

function formatLR(lr) {
  if (!lr) return "0.0";
  return lr.toExponential(2);
}

let wsActive = false;

// WebSocket connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsActive = true;
      console.log("[Cockpit] WebSocket connected to telemetry stream.");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.telemetry) updateTelemetry(data.telemetry);
        if (data.benchmarks) updateBenchmarks(data.benchmarks);
        if (data.loss_history) updateLossHistory(data.loss_history);
      } catch (e) {
        console.error("[Cockpit] Error parsing WS payload:", e);
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

// Resilient HTTP Polling Fallback (ensures local demo works without WebSocket)
async function pollTelemetryFallback() {
  if (wsActive) return;
  try {
    const [statusRes, benchRes, lossRes] = await Promise.all([
      fetch("/api/status").then(r => r.ok ? r.json() : null),
      fetch("/api/benchmarks").then(r => r.ok ? r.json() : null),
      fetch("/api/loss-history").then(r => r.ok ? r.json() : null)
    ]);
    if (statusRes) updateTelemetry(statusRes);
    if (benchRes) updateBenchmarks(benchRes);
    if (lossRes) updateLossHistory(lossRes);
  } catch (e) {
    // quiet network fail
  }
}

// Update Telemetry Header & Stats
function updateTelemetry(tel) {
  const status = tel.status || "idle";
  isRunning = tel.process_running || status === "training" || status === "evaluating_benchmark";

  statusPill.className = `status-pill ${status === 'evaluating_benchmark' ? 'evaluating' : (isRunning ? 'training' : 'idle')}`;
  
  if (status === "evaluating_benchmark") {
    statusText.textContent = `Evaluating Benchmark (Epoch ${tel.epoch || 1})`;
  } else if (isRunning) {
    statusText.textContent = `Training Active (Epoch ${tel.epoch || 1})`;
  } else if (status === "completed") {
    statusText.textContent = "Run Completed";
  } else {
    statusText.textContent = "Engine Ready";
  }

  btnStart.disabled = isRunning;
  btnStop.disabled = !isRunning;

  // GPU
  const gpu = tel.gpu || {};
  if (gpu.available) {
    gpuModel.textContent = gpu.device_name || "NVIDIA GPU";
    vramText.textContent = `${gpu.allocated_gb || 0} / ${gpu.total_gb || 0} GB`;
    vramFill.style.width = `${gpu.utilization_pct || 0}%`;
  } else {
    gpuModel.textContent = "GPU Disconnected";
    vramText.textContent = "N/A";
    vramFill.style.width = "0%";
  }

  // Stats
  statStep.textContent = `${tel.step || 0} / ${tel.max_steps || '--'}`;
  statLoss.textContent = tel.loss !== null && tel.loss !== undefined ? Number(tel.loss).toFixed(4) : "--";
  statLR.textContent = formatLR(tel.learning_rate);
  statSpeed.textContent = tel.speed_s_per_step ? `${tel.speed_s_per_step}s / it` : "--";
  statElapsed.textContent = formatSeconds(tel.elapsed_seconds);
  statEta.textContent = formatSeconds(tel.eta_seconds);
}

// Draw Responsive SVG Loss Chart
function updateLossHistory(history) {
  if (!history || history.length === 0) return;
  lossHistory = history;

  const width = svgChart.clientWidth || 500;
  const height = svgChart.clientHeight || 180;
  const padX = 35;
  const padY = 20;

  const validPoints = history.filter(h => h.loss !== null && !isNaN(h.loss));
  if (validPoints.length === 0) return;

  const minLoss = Math.min(...validPoints.map(h => h.loss)) * 0.95;
  const maxLoss = Math.max(...validPoints.map(h => h.loss)) * 1.05;
  const minStep = validPoints[0].step;
  const maxStep = Math.max(...validPoints.map(h => h.step)) || (minStep + 1);

  const getX = (step) => padX + ((step - minStep) / (maxStep - minStep || 1)) * (width - padX - 15);
  const getY = (loss) => height - padY - ((loss - minLoss) / (maxLoss - minLoss || 1)) * (height - padY * 2);

  let pathD = "";
  let areaD = `M ${getX(validPoints[0].step)} ${height - padY}`;

  validPoints.forEach((pt, i) => {
    const x = getX(pt.step);
    const y = getY(pt.loss);
    if (i === 0) {
      pathD += `M ${x} ${y}`;
    } else {
      pathD += ` L ${x} ${y}`;
    }
    areaD += ` L ${x} ${y}`;
  });

  const lastPt = validPoints[validPoints.length - 1];
  areaD += ` L ${getX(lastPt.step)} ${height - padY} Z`;

  svgChart.innerHTML = `
    <defs>
      <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#22d3ee" stop-opacity="0.3"/>
        <stop offset="100%" stop-color="#22d3ee" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    <!-- Background grid lines -->
    <line x1="${padX}" y1="${getY(maxLoss)}" x2="${width - 15}" y2="${getY(maxLoss)}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4"/>
    <line x1="${padX}" y1="${getY((minLoss + maxLoss)/2)}" x2="${width - 15}" y2="${getY((minLoss + maxLoss)/2)}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4"/>
    <line x1="${padX}" y1="${height - padY}" x2="${width - 15}" y2="${height - padY}" stroke="rgba(255,255,255,0.1)"/>
    
    <!-- Y-axis labels -->
    <text x="${padX - 5}" y="${getY(maxLoss) + 4}" fill="#64748b" font-size="10" text-anchor="end">${maxLoss.toFixed(2)}</text>
    <text x="${padX - 5}" y="${getY(minLoss) + 4}" fill="#64748b" font-size="10" text-anchor="end">${minLoss.toFixed(2)}</text>

    <!-- Path Area & Stroke -->
    <path d="${areaD}" fill="url(#lossGradient)"/>
    <path d="${pathD}" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Latest Point Highlight -->
    <circle cx="${getX(lastPt.step)}" cy="${getY(lastPt.loss)}" r="4" fill="#22d3ee" stroke="#090b0e" stroke-width="2"/>
  `;
}

function safeNum(val, fallback = 0.0) {
  if (val === null || val === undefined) return fallback;
  const n = parseFloat(val);
  return isNaN(n) ? fallback : n;
}

// Update Epoch Benchmark Table
function updateBenchmarks(list) {
  if (!list || !Array.isArray(list) || list.length === 0) return;
  benchmarks = list;

  // Find lowest normalized WER among checkpoints
  const checkpoints = list.filter(r => !r.is_baseline && r.normalized_wer != null);
  let bestWer = 999.0;
  if (checkpoints.length > 0) {
    bestWer = Math.min(...checkpoints.map(c => safeNum(c.normalized_wer, 999.0)));
  }

  // Find BuzzASR base WER
  const buzzRow = list.find(r => r.epoch && r.epoch.includes("BuzzASR Base"));
  const buzzBaseWer = buzzRow ? safeNum(buzzRow.normalized_wer, 17.20) : 17.20;

  let html = "";
  list.forEach((row) => {
    const isBase = !!row.is_baseline;
    const rowWer = safeNum(row.normalized_wer);
    const isBest = !isBase && Math.abs(rowWer - bestWer) < 0.001;
    const rowClass = isBase ? "baseline-row" : (isBest ? "best-row" : "");

    // Delta vs. Whisper Large v3
    let deltaWhisperBadge = `<span class="delta-badge neutral">Baseline</span>`;
    if (!isBase) {
      const delta = safeNum(row.delta_vs_whisper);
      if (delta < 0) {
        deltaWhisperBadge = `<span class="delta-badge win">↓ ${Math.abs(delta).toFixed(1)}%</span>`;
      } else if (delta > 0) {
        deltaWhisperBadge = `<span class="delta-badge loss">↑ +${delta.toFixed(1)}%</span>`;
      } else {
        deltaWhisperBadge = `<span class="delta-badge neutral">0.0%</span>`;
      }
    }

    // Delta vs. BuzzASR Base
    let deltaBuzzBadge = `<span class="delta-badge neutral">Baseline</span>`;
    if (row.epoch.includes("Whisper")) {
      deltaBuzzBadge = `<span class="delta-badge loss">↑ +11.3%</span>`;
    } else if (!row.epoch.includes("BuzzASR Base")) {
      const deltaBuzz = row.delta_vs_buzzasr != null ? safeNum(row.delta_vs_buzzasr) : safeNum(rowWer - buzzBaseWer);
      if (deltaBuzz < 0) {
        deltaBuzzBadge = `<span class="delta-badge win">↓ ${Math.abs(deltaBuzz).toFixed(1)}%</span>`;
      } else if (deltaBuzz > 0) {
        deltaBuzzBadge = `<span class="delta-badge loss">↑ +${deltaBuzz.toFixed(1)}%</span>`;
      } else {
        deltaBuzzBadge = `<span class="delta-badge neutral">0.0%</span>`;
      }
    }

    const epochTitle = isBest ? `★ ${row.epoch}` : (row.epoch || "Checkpoint");
    const cerStr = row.normalized_cer != null ? safeNum(row.normalized_cer).toFixed(2) + '%' : '--';
    const diagStr = row.diagnostic_wer != null ? safeNum(row.diagnostic_wer).toFixed(2) + '%' : '--';
    const formalStr = row.formal_wer != null ? safeNum(row.formal_wer).toFixed(2) + '%' : '--';
    const informalStr = row.informal_wer != null ? safeNum(row.informal_wer).toFixed(2) + '%' : '--';

    html += `
      <tr class="${rowClass}">
        <td><span class="badge-epoch">${epochTitle}</span></td>
        <td style="font-weight: 600; color: ${isBest ? '#10b981' : '#f8fafc'}">${rowWer.toFixed(2)}%</td>
        <td>${cerStr}</td>
        <td>${diagStr}</td>
        <td>${formalStr}</td>
        <td>${informalStr}</td>
        <td>${deltaWhisperBadge}</td>
        <td>${deltaBuzzBadge}</td>
      </tr>
    `;
  });

  if (benchmarkTbody) {
    benchmarkTbody.innerHTML = html;
  }

  // Update canary cards with latest checkpoint's canary transcriptions
  const latestWithCanary = [...list].reverse().find(r => r.canary_transcriptions && r.canary_transcriptions.length > 0);
  if (latestWithCanary) {
    renderCanary(latestWithCanary.canary_transcriptions);
  }
}

// Render Canary Audio Tokens
function renderCanary(canaries) {
  if (!canaries || canaries.length === 0) return;
  let html = "";
  canaries.forEach(c => {
    html += `
      <div class="canary-item">
        <div class="canary-meta">
          <span class="canary-tag">${c.category || 'Canary'}</span>
          <span style="color: #64748b; font-size: 11px;">${c.is_informal ? 'گفتاری' : 'رسمی'}</span>
        </div>
        <div class="canary-ref">${c.ref}</div>
        <div class="canary-pred">${c.pred || '...در حال پردازش'}</div>
      </div>
    `;
  });
  canaryContainer.innerHTML = html;
}

// Fetch periodic logs
async function fetchLogs() {
  try {
    const res = await fetch("/api/logs?tail=80");
    const data = await res.json();
    if (data.logs && Array.isArray(data.logs)) {
      consoleBox.innerHTML = data.logs.map(line => {
        let cls = "log-line";
        if (line.includes("Error") || line.includes("Traceback")) cls += " error";
        if (line.includes("Benchmark saved") || line.includes("Epoch")) cls += " success";
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      }).join("");

      if (autoScrollLogs) {
        consoleBox.scrollTop = consoleBox.scrollHeight;
      }
    }
  } catch (e) {
    // quiet network fail
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Button Events
btnStart.addEventListener("click", async () => {
  btnStart.disabled = true;
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: parseInt(inputEpochs.value, 10) || 4,
        batch_size: parseInt(inputBatchSize.value, 10) || 16,
        learning_rate: parseFloat(inputLearningRate.value) || 1e-4
      })
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Start failed: ${err.detail || 'Unknown error'}`);
      btnStart.disabled = false;
    }
  } catch (e) {
    alert(`Failed to trigger training: ${e}`);
    btnStart.disabled = false;
  }
});

btnStop.addEventListener("click", async () => {
  if (!confirm("Are you sure you want to halt the training job? A checkpoint will be saved.")) {
    return;
  }
  btnStop.disabled = true;
  try {
    await fetch("/api/train/stop", { method: "POST" });
  } catch (e) {
    console.error("Stop failed", e);
  }
});

// Init
window.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
  pollTelemetryFallback();
  setInterval(pollTelemetryFallback, 1000); // 1s polling fallback if WS inactive
  fetchLogs();
  setInterval(fetchLogs, 2500); // Poll logs every 2.5s
});
