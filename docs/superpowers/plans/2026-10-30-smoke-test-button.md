# Smoke-Test Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click **Smoke test** button to the training cockpit that runs an isolated 200-sample training check and stops, without touching the production run.

**Architecture:** A dedicated FastAPI start endpoint builds the existing `scripts/train_lora.py` command with fixed smoke flags and isolated `smoke-test/` paths. A status endpoint reports running/passed/failed from the smoke process, a PID file, an attempt marker, and the saved adapter. The vanilla-JS dashboard adds a button that starts the run and polls status.

**Tech Stack:** Python 3, FastAPI, pytest, vanilla JavaScript, HTML/CSS.

**Spec:** `docs/superpowers/specs/2026-10-30-smoke-test-button-design.md`

## Global Constraints

- Smoke samples: `--max_train_samples 200`, `--max_eval_samples 50`.
- Smoke schedule: `--epochs 1`, `--eval_steps 5`, `--save_steps 5`, `--warmup_steps 2`.
- Smoke benchmark: `--max_benchmark_samples 25`; the final full-test evaluation is disabled with `--no-run-post-test-eval`.
- Smoke starts clean: `--fresh_start` and `--no-auto_resume`.
- Smoke output and results paths end in `smoke-test` and must never contain the production output/results paths.
- Pass means exit code `0` **and** `best_adapter/adapter_config.json` exists.
- The full-run Start/Fresh/Halt behavior must remain unchanged, except that Start also rejects while a smoke run is active.
- Do not modify `scripts/train_lora.py`, `run_train.sh`, or YAML configs.
- No automatic full run after a passed smoke test.
- Do not commit unrelated untracked files.

## Review Focus

- **Server restarts mid-smoke:** a live PID file must report `running`; an attempt with no result must report `failed`, and an attempt with the adapter artifact may report `passed`.
- **Stale adapter from an earlier smoke run:** a previous `best_adapter` must not create a false pass; it is removed before the new attempt, and a non-zero exit always fails.
- **Concurrent jobs:** a running full run must block smoke start, and a running smoke run must block full start; a finished smoke run must not block the full run.
- **Isolation:** the smoke command must never contain the production output or results directory.
- **Bad controls:** batch size above 8 without gradient checkpointing, invalid accumulation, or invalid learning rate must be rejected before spawning.

---

### Task 1: Smoke status state and endpoint

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: existing `RESULTS_DIR`, `read_json_safely`, `_is_process_running`, `_default_output_dir`, `_sanitize_output_dir`.
- Produces:
  - `SMOKE_DIR_NAME: str = "smoke-test"`
  - `SMOKE_RESULTS_DIR: Path`
  - `SMOKE_LOG_FILE: Path`
  - `SMOKE_PID_FILE: Path`
  - `SMOKE_ATTEMPT_FILE: Path`
  - `smoke_process: Optional[subprocess.Popen]`
  - `smoke_log_fd`
  - `smoke_last_exit_code: Optional[int]`
  - `_abs_output_path(path: str) -> Path`
  - `_smoke_output_dir() -> str`
  - `_smoke_running() -> bool`
  - `_close_smoke_log_fd() -> None`
  - `_read_log_tail(path: Path, tail: int = 30) -> List[str]`
  - `GET /api/train/smoke/status`

- [ ] **Step 1: Write the failing status tests**

Add `client.mod = mod` to the existing `client` fixture in `tests/test_server_train.py`. Add `import os` at the top.

```python
def _write_smoke_attempt(client, *, exit_code=None, adapter=True, log_lines=()):
    out = Path(client.mod._smoke_output_dir())
    if adapter:
        adapter_dir = out / "best_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    client.mod.SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client.mod.SMOKE_ATTEMPT_FILE.write_text("{}", encoding="utf-8")
    client.mod.smoke_last_exit_code = exit_code
    if log_lines:
        client.mod.SMOKE_LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")


def test_smoke_status_is_idle_before_any_attempt(client):
    res = client.get("/api/train/smoke/status")
    assert res.status_code == 200
    assert res.json()["status"] == "idle"


def test_smoke_status_reports_running_from_live_pid(client):
    client.mod.SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client.mod.SMOKE_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    res = client.get("/api/train/smoke/status")
    assert res.status_code == 200
    assert res.json()["status"] == "running"
    assert res.json()["running"] is True


def test_smoke_status_passes_on_zero_exit_and_adapter(client):
    _write_smoke_attempt(client, exit_code=0, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "passed"
    assert res.json()["adapter_exists"] is True


def test_smoke_status_passes_after_restart_when_adapter_exists(client):
    _write_smoke_attempt(client, exit_code=None, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "passed"


def test_smoke_status_fails_on_nonzero_exit_even_with_adapter(client):
    _write_smoke_attempt(client, exit_code=1, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["exit_code"] == 1


def test_smoke_status_fails_when_adapter_is_missing(client):
    _write_smoke_attempt(client, exit_code=0, adapter=False)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["adapter_exists"] is False


def test_smoke_status_fails_when_attempt_has_no_result(client):
    _write_smoke_attempt(client, exit_code=None, adapter=False)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"


def test_smoke_status_includes_log_tail_on_failure(client):
    _write_smoke_attempt(
        client,
        exit_code=1,
        adapter=False,
        log_lines=["loading model", "RuntimeError: out of memory"],
    )
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["log_tail"][-1] == "RuntimeError: out of memory"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server_train.py -v -k smoke_status`
Expected: FAIL because the endpoint and smoke state do not exist.

- [ ] **Step 3: Implement the smoke state and status endpoint**

In `server/app.py`, add the constants after `PID_FILE`:

```python
SMOKE_DIR_NAME = "smoke-test"
SMOKE_RESULTS_DIR = RESULTS_DIR / SMOKE_DIR_NAME
SMOKE_LOG_FILE = SMOKE_RESULTS_DIR / "training_live.log"
SMOKE_PID_FILE = SMOKE_RESULTS_DIR / "training.pid"
SMOKE_ATTEMPT_FILE = SMOKE_RESULTS_DIR / "smoke_attempt.json"
smoke_process: Optional[subprocess.Popen] = None
smoke_log_fd = None
smoke_last_exit_code: Optional[int] = None
```

Add helpers near `_training_running`:

```python
def _abs_output_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def _smoke_output_dir() -> str:
    base = Path(_default_output_dir())
    return _sanitize_output_dir(str(base.parent / SMOKE_DIR_NAME))


def _close_smoke_log_fd() -> None:
    global smoke_log_fd
    try:
        if smoke_log_fd is not None:
            try:
                smoke_log_fd.flush()
            except Exception:
                pass
            try:
                smoke_log_fd.close()
            except Exception:
                pass
    finally:
        smoke_log_fd = None


def _smoke_running() -> bool:
    global smoke_process, smoke_last_exit_code
    if smoke_process is not None:
        code = smoke_process.poll()
        if code is None:
            return True
        smoke_last_exit_code = code
        smoke_process = None
        _close_smoke_log_fd()
        try:
            SMOKE_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        if SMOKE_PID_FILE.exists():
            pid = int(SMOKE_PID_FILE.read_text(encoding="utf-8").strip() or "0")
            if _is_process_running(pid):
                return True
            SMOKE_PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def _read_log_tail(path: Path, tail: int = 30) -> List[str]:
    try:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip() for line in f.readlines()[-tail:]]
    except Exception:
        return []
```

Add the endpoint after `/api/train/start` (or before it):

```python
@app.get("/api/train/smoke/status")
async def get_smoke_status():
    running = _smoke_running()
    try:
        output_dir = _smoke_output_dir()
        adapter_exists = (
            _abs_output_path(output_dir) / "best_adapter" / "adapter_config.json"
        ).is_file()
    except Exception:
        output_dir = ""
        adapter_exists = False

    attempt_exists = SMOKE_ATTEMPT_FILE.is_file()
    if running:
        status = "running"
    elif not attempt_exists:
        status = "idle"
    elif smoke_last_exit_code == 0 and adapter_exists:
        status = "passed"
    elif smoke_last_exit_code is None and adapter_exists:
        status = "passed"
    else:
        status = "failed"

    payload: Dict[str, Any] = {
        "status": status,
        "running": running,
        "exit_code": smoke_last_exit_code,
        "output_dir": output_dir,
        "adapter_exists": adapter_exists,
    }
    if status == "failed":
        payload["log_tail"] = _read_log_tail(SMOKE_LOG_FILE)
    return payload
```

- [ ] **Step 4: Run the status tests**

Run: `pytest tests/test_server_train.py -v -k smoke_status`
Expected: PASS.

- [ ] **Step 5: Run the existing server tests**

Run: `pytest tests/test_server_train.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: add smoke-test status endpoint"
```

---

### Task 2: Smoke start endpoint and concurrency

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: Task 1 constants and helpers; existing `_check_auth`, `_check_audio_backend_or_400`, `_default_output_dir`, `_sanitize_output_dir`, `BASE_DIR`.
- Produces:
  - `_any_training_running() -> bool`
  - `POST /api/train/smoke` handled by `async def start_smoke_test(config: Dict[str, Any] = None)`
  - Start response: `{"status": "started", "mode": "smoke", "pid": int, "output_dir": str, "results_dir": str}`

- [ ] **Step 1: Write the failing smoke-start tests**

Add these helpers and tests to `tests/test_server_train.py`:

```python
def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class _RunningProcess:
    pid = 999_999

    def poll(self):
        return None


def test_smoke_start_builds_fast_isolated_command(client):
    res = client.post(
        "/api/train/smoke",
        json={
            "batch_size": 4,
            "grad_accum": 8,
            "learning_rate": 2e-4,
            "gradient_checkpointing": False,
        },
    )
    assert res.status_code == 200, res.text
    cmd = client.launched[0]
    assert _flag_value(cmd, "--max_train_samples") == "200"
    assert _flag_value(cmd, "--max_eval_samples") == "50"
    assert _flag_value(cmd, "--epochs") == "1"
    assert _flag_value(cmd, "--eval_steps") == "5"
    assert _flag_value(cmd, "--save_steps") == "5"
    assert _flag_value(cmd, "--warmup_steps") == "2"
    assert _flag_value(cmd, "--max_benchmark_samples") == "25"
    assert _flag_value(cmd, "--batch_size") == "4"
    assert _flag_value(cmd, "--grad_accum") == "8"
    assert _flag_value(cmd, "--learning_rate") == "0.0002"
    assert "--no-run-post-test-eval" in cmd
    assert "--fresh_start" in cmd
    assert "--no-auto_resume" in cmd
    assert Path(_flag_value(cmd, "--output_dir")).name == "smoke-test"
    assert Path(_flag_value(cmd, "--results_dir")).name == "smoke-test"
    assert res.json()["mode"] == "smoke"


def test_smoke_start_removes_stale_adapter_and_writes_attempt(client):
    out = Path(client.mod._smoke_output_dir())
    adapter_dir = out / "best_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 200, res.text
    assert not adapter_dir.exists()
    assert client.mod.SMOKE_ATTEMPT_FILE.is_file()


def test_smoke_start_rejects_batch_above_8_without_checkpointing(client):
    res = client.post(
        "/api/train/smoke",
        json={"batch_size": 16, "gradient_checkpointing": False},
    )
    assert res.status_code == 400
    assert "gradient_checkpointing" in res.json()["detail"]
    assert client.launched == []


def test_smoke_start_rejects_invalid_accumulation(client):
    res = client.post("/api/train/smoke", json={"grad_accum": 0})
    assert res.status_code == 400
    assert client.launched == []


def test_smoke_start_rejects_invalid_learning_rate(client):
    res = client.post("/api/train/smoke", json={"learning_rate": 1.0})
    assert res.status_code == 400
    assert client.launched == []


def test_smoke_start_rejected_while_full_run_active(client):
    client.mod.training_process = _RunningProcess()
    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 400
    assert client.launched == []


def test_full_start_rejected_while_smoke_active(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    res = client.post("/api/train/start", json={})
    assert res.status_code == 400
    assert len(client.launched) == 1


def test_full_start_works_after_smoke_finishes(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    client.mod.smoke_process = None
    client.mod.smoke_last_exit_code = 0
    client.mod.SMOKE_PID_FILE.unlink(missing_ok=True)

    res = client.post("/api/train/start", json={})
    assert res.status_code == 200
    assert len(client.launched) == 2


def test_smoke_status_reports_running_after_start(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "running"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server_train.py -v -k "smoke_start or smoke_active or after_smoke"`
Expected: FAIL because `/api/train/smoke` and `_any_training_running` do not exist.

- [ ] **Step 3: Implement the start endpoint and shared busy check**

Add `import shutil` to `server/app.py`.

Add near `_training_running`:

```python
def _any_training_running() -> bool:
    return _training_running() or _smoke_running()
```

Change the top of `start_training` to replace the existing process/PID checks with:

```python
    if _any_training_running():
        raise HTTPException(status_code=400, detail="Training or smoke test is already running.")
```

Add the smoke endpoint before `/api/train/start`:

```python
@app.post("/api/train/smoke")
async def start_smoke_test(config: Dict[str, Any] = None):
    global smoke_process, smoke_log_fd, smoke_last_exit_code

    _check_auth(config)
    _check_audio_backend_or_400()
    if _any_training_running():
        raise HTTPException(status_code=400, detail="Training or smoke test is already running.")

    config = config or {}
    try:
        batch_i = int(config.get("batch_size", 8))
        accum_i = int(config.get("grad_accum", 4))
        lr_f = float(config.get("learning_rate", 1e-4))
        eval_bs_i = int(config.get("eval_batch_size", 16))
        num_proc_i = int(config.get("num_proc", 8))
        tele_f = float(config.get("telephony_prob", 0.5))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="batch_size/grad_accum/eval_batch_size/num_proc must be integers; "
                   "learning_rate/telephony_prob numeric.",
        )
    if not (1 <= batch_i <= 64):
        raise HTTPException(status_code=400, detail="batch_size must be 1..64.")
    if not (1 <= accum_i <= 16):
        raise HTTPException(status_code=400, detail="grad_accum must be 1..16.")
    if not (1e-7 <= lr_f <= 1e-2):
        raise HTTPException(status_code=400, detail="learning_rate must be 1e-7..1e-2.")
    if not (1 <= eval_bs_i <= 64):
        raise HTTPException(status_code=400, detail="eval_batch_size must be 1..64.")
    if not (1 <= num_proc_i <= 16):
        raise HTTPException(status_code=400, detail="num_proc must be 1..16.")
    if not (0.0 <= tele_f <= 1.0):
        raise HTTPException(status_code=400, detail="telephony_prob must be 0..1.")

    ckpt_raw = config.get("gradient_checkpointing", False)
    gradient_ckpt = (
        str(ckpt_raw).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(ckpt_raw, str)
        else bool(ckpt_raw)
    )
    if batch_i > 8 and not gradient_ckpt:
        raise HTTPException(
            status_code=400,
            detail="batch_size above 8 needs gradient_checkpointing on; "
                   "without it batch 16 runs out of memory on a 48 GB GPU.",
        )

    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar")).strip() or "shekar-ai/Neyshekar"
    if len(dataset_name) > 128 or any(c in dataset_name for c in ";&|$` \n"):
        raise HTTPException(status_code=400, detail="Invalid dataset_name.")

    output_dir = _smoke_output_dir()
    SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _abs_output_path(output_dir).mkdir(parents=True, exist_ok=True)

    # A new attempt must never inherit a previous pass.
    shutil.rmtree(_abs_output_path(output_dir) / "best_adapter", ignore_errors=True)
    SMOKE_PID_FILE.unlink(missing_ok=True)
    SMOKE_ATTEMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_ATTEMPT_FILE.write_text(
        json.dumps({"started_at": time.time()}), encoding="utf-8"
    )

    cmd = [
        sys.executable,
        "-u",
        str(BASE_DIR / "scripts" / "train_lora.py"),
        "--base_model", "BuzzASR/persian",
        "--dataset_name", dataset_name,
        "--output_dir", output_dir,
        "--results_dir", str(SMOKE_RESULTS_DIR),
        "--epochs", "1",
        "--batch_size", str(batch_i),
        "--grad_accum", str(accum_i),
        "--learning_rate", str(lr_f),
        "--eval_batch_size", str(eval_bs_i),
        "--max_train_samples", "200",
        "--max_eval_samples", "50",
        "--max_benchmark_samples", "25",
        "--num_proc", str(num_proc_i),
        "--telephony_prob", str(tele_f),
        "--warmup_steps", "2",
        "--eval_steps", "5",
        "--save_steps", "5",
        "--enable_telemetry",
        "--run_epoch_benchmark",
        "--no-run-post-test-eval",
        "--fresh_start",
        "--no-auto_resume",
        "--gradient_checkpointing" if gradient_ckpt else "--no-gradient_checkpointing",
    ]

    _close_smoke_log_fd()
    smoke_log_fd = open(SMOKE_LOG_FILE, "w", encoding="utf-8")
    smoke_log_fd.write(f"=== Smoke test launched at {time.ctime()} ===\n")
    smoke_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    smoke_log_fd.flush()
    smoke_last_exit_code = None

    try:
        smoke_process = subprocess.Popen(
            cmd,
            stdout=smoke_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
    except Exception as e:
        try:
            smoke_log_fd.write(f"\n!!! Failed to launch smoke test: {e}\n")
            smoke_log_fd.flush()
        except Exception:
            pass
        _close_smoke_log_fd()
        smoke_process = None
        raise HTTPException(status_code=500, detail=f"Failed to launch smoke test: {e}")

    try:
        SMOKE_PID_FILE.write_text(str(smoke_process.pid), encoding="utf-8")
    except Exception:
        pass

    return {
        "status": "started",
        "mode": "smoke",
        "pid": smoke_process.pid,
        "output_dir": output_dir,
        "results_dir": str(SMOKE_RESULTS_DIR),
    }
```

- [ ] **Step 4: Run the smoke-start tests**

Run: `pytest tests/test_server_train.py -v -k "smoke_start or smoke_active or after_smoke or smoke_status_reports_running_after_start"`
Expected: PASS.

- [ ] **Step 5: Run all server tests**

Run: `pytest tests/test_server_train.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: add isolated smoke-test start endpoint"
```

---

### Task 3: Dashboard button and status polling

**Files:**
- Modify: `server/static/index.html`
- Modify: `server/static/app.js`
- Modify: `server/static/style.css`
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: `POST /api/train/smoke`, `GET /api/train/smoke/status`.
- Produces: `btnSmoke`, `smokeResult`, `smokeStatus`, `smokeLogWrap`, `smokeLog` DOM nodes; `startSmokeRun()`, `pollSmokeStatus()`, `scheduleSmokePoll(delay)`, `setSmokeResult(kind, text, logLines)`; `smokeRunning` and `smokePollTimer` module state.

- [ ] **Step 1: Write the failing static UI test**

Add to `tests/test_server_train.py`:

```python
def test_smoke_controls_are_present_in_static_ui():
    html = (ROOT / "server" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "server" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="btnSmoke"' in html
    assert 'id="smokeResult"' in html
    assert 'id="smokeStatus"' in html
    assert 'id="smokeLog"' in html
    assert "/api/train/smoke" in js
    assert "/api/train/smoke/status" in js
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_server_train.py::test_smoke_controls_are_present_in_static_ui -v`
Expected: FAIL because the button and status markup do not exist.

- [ ] **Step 3: Add the HTML controls**

In `server/static/index.html`, add the button to the `.config-actions` row before **Start run**:

```html
<button id="btnSmoke" type="button" class="btn btn-fresh"
        title="Run a 200-sample, 1-epoch check in an isolated directory; does not affect this run">
  Smoke test
</button>
```

After the closing `</form>` of the config form, add:

```html
<div id="smokeResult" class="smoke-result" hidden>
  <p id="smokeStatus"></p>
  <details id="smokeLogWrap" hidden>
    <summary>Show smoke test log</summary>
    <pre id="smokeLog"></pre>
  </details>
</div>
```

- [ ] **Step 4: Add the JavaScript state and polling**

In `server/static/app.js`, add to the `el` object:

```javascript
btnSmoke: document.getElementById("btnSmoke"),
smokeResult: document.getElementById("smokeResult"),
smokeStatus: document.getElementById("smokeStatus"),
smokeLogWrap: document.getElementById("smokeLogWrap"),
smokeLog: document.getElementById("smokeLog"),
```

Add module state near `let ws = null;`:

```javascript
let smokeRunning = false;
let smokePollTimer = null;
```

In `updateTelemetry`, change the button-state lines to:

```javascript
el.btnStart.disabled = running || smokeRunning;
el.btnFresh.disabled = running || smokeRunning;
el.btnSmoke.disabled = running || smokeRunning;
el.btnStop.disabled = !running || state === "stopping";
```

Add these functions before the `// Controls` section:

```javascript
function setSmokeResult(kind, text, logLines = []) {
  el.smokeResult.hidden = kind === "idle";
  el.smokeResult.className = `smoke-result ${kind}`;
  el.smokeStatus.textContent = text;
  const hasLog = logLines.length > 0;
  el.smokeLogWrap.hidden = !hasLog;
  el.smokeLog.textContent = hasLog ? logLines.join("\n") : "";
  if (hasLog) el.smokeLogWrap.open = kind === "fail";
}

function scheduleSmokePoll(delay = 2000) {
  if (smokePollTimer) clearTimeout(smokePollTimer);
  smokePollTimer = setTimeout(pollSmokeStatus, delay);
}

async function pollSmokeStatus() {
  if (smokePollTimer) {
    clearTimeout(smokePollTimer);
    smokePollTimer = null;
  }
  try {
    const res = await fetch("/api/train/smoke/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "running") {
      smokeRunning = true;
      setSmokeResult("running", "Smoke test running — 200 train / 50 eval samples.");
      scheduleSmokePoll();
      return;
    }
    smokeRunning = false;
    if (data.status === "passed") {
      setSmokeResult("pass", "Smoke test passed. Ready to start the full run.");
    } else if (data.status === "failed") {
      const exit = data.exit_code != null ? ` (exit ${data.exit_code})` : "";
      setSmokeResult("fail", `Smoke test failed${exit}.`, data.log_tail || []);
    } else {
      setSmokeResult("idle", "");
    }
    pollTelemetryFallback();
  } catch (e) {
    smokeRunning = false;
    setSmokeResult("fail", `Could not read smoke status: ${e}`);
    pollTelemetryFallback();
  }
}

async function startSmokeRun() {
  if (smokeRunning) return;
  smokeRunning = true;
  el.btnSmoke.disabled = true;
  el.btnStart.disabled = true;
  el.btnFresh.disabled = true;
  setSmokeResult("running", "Smoke test running — 200 train / 50 eval samples.");
  try {
    const res = await fetch("/api/train/smoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_size: parseInt(el.inputBatchSize.value, 10) || 8,
        grad_accum: parseInt(el.inputGradAccum.value, 10) || 4,
        learning_rate: parseFloat(el.inputLearningRate.value) || 1e-4,
        gradient_checkpointing: el.inputGradCheckpoint.value === "on",
      }),
    });
    if (!res.ok) {
      const text = await res.text();
      let detail = text;
      try {
        const err = JSON.parse(text);
        detail = err.detail || err.error || text || "Unknown error";
      } catch {
        // Keep raw text.
      }
      smokeRunning = false;
      setSmokeResult("fail", `Could not start smoke test: ${detail}`);
      pollTelemetryFallback();
      return;
    }
    scheduleSmokePoll(0);
  } catch (e) {
    smokeRunning = false;
    setSmokeResult("fail", `Could not start smoke test: ${e}`);
    pollTelemetryFallback();
  }
}
```

Add the listener near the existing control listeners:

```javascript
el.btnSmoke.addEventListener("click", startSmokeRun);
```

In the `DOMContentLoaded` handler, add:

```javascript
pollSmokeStatus();
```

- [ ] **Step 5: Add the status styles**

In `server/static/style.css`, after the `.btn-fresh:hover` rule:

```css
.smoke-result {
  margin-top: 10px;
  padding: 10px 12px;
  border: 1px solid var(--rule-soft);
  border-radius: var(--radius);
  background: var(--paper-deep);
  color: var(--ink-2);
  font-size: 13px;
}

.smoke-result.pass {
  color: var(--firoozeh);
  border-color: var(--firoozeh);
}

.smoke-result.fail {
  color: var(--brick);
  border-color: rgba(162, 64, 44, 0.45);
}

.smoke-result pre {
  max-height: 220px;
  margin: 8px 0 0;
  overflow: auto;
  white-space: pre-wrap;
  font: 12px var(--font-mono);
  color: var(--ink-2);
}
```

- [ ] **Step 6: Run the static UI test and JavaScript syntax check**

Run: `pytest tests/test_server_train.py::test_smoke_controls_are_present_in_static_ui -v`
Expected: PASS.

Run: `node --check server/static/app.js`
Expected: exit code 0.

- [ ] **Step 7: Run all server tests**

Run: `pytest tests/test_server_train.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/style.css tests/test_server_train.py
git commit -m "feat: add smoke-test button to training cockpit"
```

---

### Task 4: Regression verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run the wider test suite**

Run: `pytest tests/test_server_train.py tests/test_script_defaults.py -v`
Expected: PASS.

- [ ] **Step 2: Check the diff**

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 3: Verify no production paths appear in the smoke command test output**

Run: `pytest tests/test_server_train.py::test_smoke_start_builds_fast_isolated_command -v`
Expected: PASS; the test asserts both smoke paths end in `smoke-test`.

- [ ] **Step 4: Commit any test-only cleanup**

If Step 1–3 required changes:

```bash
git add server/app.py server/static/index.html server/static/app.js server/static/style.css tests/test_server_train.py
git commit -m "test: verify smoke-test isolation"
```

If no changes were needed, skip the commit.
