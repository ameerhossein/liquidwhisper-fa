# RunPod L40S Automated Watcher & WebUI Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, automated Python watcher script that monitors RunPod stock for an NVIDIA L40S GPU, automatically provisions an on-demand pod with 300 GB container disk and 200 GB persistent volume, bootstraps the Persian ASR environment and Web Cockpit over SSH with private git clone, enables a browser web terminal, and notifies the user with direct URLs upon readiness.

**Architecture:** A modular Python engine (`src/runpod_deploy/`) standard-library-first (zero external Python dependencies required on local machine), providing:
1. `client.py`: RunPod REST v2 / GraphQL catalog client for GPU availability polling and pod lifecycle management.
2. `ssh_bootstrap.py`: Dual-transport SSH connector (handling direct TCP and RunPod gateway proxy) that executes remote bootstrap commands (installing `ttyd` web terminal, cloning private repo via `gh` token, executing `./setup_server.sh` and `./runpod_web.sh -d`).
3. `notify.py`: Multimodal notification dispatcher (desktop alert via `notify-send`, audible bell chime, and formatted terminal connection banner).
4. `scripts/runpod_l40s_watch.py`: CLI driver with live spinner, retry countdown, stock monitoring, and progress reporting.
5. `runpod_watch.sh`: Quick launcher script.

**Tech Stack:** Python 3.10+ (standard library: `urllib.request`, `json`, `subprocess`, `time`, `shutil`), Bash, RunPod REST v2 / GraphQL API, `gh` CLI for token resolution.

**Spec:** `docs/superpowers/specs/2026-11-01-runpod-l40s-launcher-design.md`

## Global Constraints

- **Local Python Dependencies:** Stdlib only (`urllib.request`, `json`, `subprocess`, `argparse`, `shutil`, `dataclasses`, `time`). No mandatory pip packages required on the local workstation.
- **Base Container Image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (guarantees Python 3.11 and CUDA 12.4 audio stack matching repo requirements).
- **GPU Target:** `NVIDIA L40S` (count 1, 48 GB VRAM).
- **Default Storage Allocations:**
  - Ephemeral container root disk (`disk`): **300 GB** (enforces headroom for the ~170 GB paired shard cache, raw downloads, and the 180 GB `ensure_space()` guard).
  - Persistent volume (`mounts.persistent`): **200 GB** at `/workspace`.
- **Exposed Ports:** `["7860/http", "8080/http", "8888/http", "22/tcp"]`.
- **Service Ports:**
  - `7860/http`: Persian ASR Cockpit Web UI (`https://<pod-id>-7860.proxy.runpod.net`).
  - `8080/http`: Instant Browser Web Terminal powered by `ttyd` (`https://<pod-id>-8080.proxy.runpod.net`).
- **Private Repo Clone:** Retrieve GitHub token locally via `gh auth token` or `GITHUB_TOKEN` environment variable; clone via HTTPS `https://x-access-token:${TOKEN}@github.com/ameerhossein/liquidwhisper-fa.git` without printing token to logs.

## Review Focus

1. **Missing or Expired RUNPOD_API_KEY:** Script must validate the key before entering the polling loop and display a clear, actionable remediation message rather than crashing with an unhandled exception.
2. **Missing GitHub Token:** If neither `gh auth token` nor `GITHUB_TOKEN` is available, script must fail early before provisioning a billable pod.
3. **RunPod SSH Gateway vs. Direct TCP:** Pod metadata may report `ssh.direct: null` and assign `ssh.proxy` gateway (`host: "ssh.runpod.io", user: "<pod-id>"`). The SSH runner must cleanly handle both modes.
4. **Transient Network or Rate Limit Errors (429/500/502):** API requests in the polling loop must catch HTTP/network errors and backoff gracefully without terminating the watcher.
5. **Partial or Aborted Setup:** If SSH bootstrap fails during execution, script must print the failed command, pod ID, and instructions on how to inspect or terminate the pod to prevent unexpected billing.

---

### Task 1: RunPod API Client & Catalog Availability Checker

**Files:**
- Create: `src/runpod_deploy/__init__.py`
- Create: `src/runpod_deploy/client.py`
- Test: `tests/test_runpod_client.py`

**Interfaces:**
- Produces:
  - `RunPodClient(api_key: str, base_url: str = "https://api.runpod.io/v2")`
  - `RunPodClient.check_l40s_availability(cloud_type: str = "SECURE") -> tuple[bool, str, list[dict]]`
    (returns `(is_available, best_datacenter_id, available_dcs)`)
  - `RunPodClient.create_pod(...) -> dict`
  - `RunPodClient.get_pod(pod_id: str) -> dict`
  - `RunPodClient.terminate_pod(pod_id: str) -> bool`
  - `resolve_runpod_key() -> str`
  - `resolve_github_token() -> str`
  - `resolve_ssh_public_key() -> str`

- [ ] **Step 1: Write failing tests for credential resolution and catalog polling**

```python
# tests/test_runpod_client.py
import pytest
from unittest.mock import patch, MagicMock
from src.runpod_deploy.client import (
    resolve_runpod_key,
    resolve_github_token,
    resolve_ssh_public_key,
    RunPodClient,
)

def test_resolve_runpod_key_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key_123")
    assert resolve_runpod_key() == "test_key_123"

def test_resolve_runpod_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY is not set"):
        resolve_runpod_key(config_file=tmp_path / "empty.toml")

def test_resolve_github_token_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_mocktoken")
    assert resolve_github_token() == "ghp_mocktoken"

def test_resolve_github_token_gh_cli(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="gho_from_cli\n")
        assert resolve_github_token() == "gho_from_cli"

def test_check_l40s_availability_in_stock():
    client = RunPodClient(api_key="mock_key")
    mock_catalog = [
        {
            "id": "NVIDIA L40S",
            "availability": "HIGH",
            "dataCenters": [
                {"id": "US-NV-1", "availability": "HIGH"},
                {"id": "EU-IS-1", "availability": "NONE"},
            ]
        }
    ]
    with patch.object(client, "_request", return_value=mock_catalog):
        avail, dc_id, dcs = client.check_l40s_availability()
        assert avail is True
        assert dc_id == "US-NV-1"
        assert len(dcs) == 1

def test_check_l40s_availability_out_of_stock():
    client = RunPodClient(api_key="mock_key")
    mock_catalog = [
        {
            "id": "NVIDIA L40S",
            "availability": "NONE",
            "dataCenters": [
                {"id": "US-NV-1", "availability": "NONE"},
            ]
        }
    ]
    with patch.object(client, "_request", return_value=mock_catalog):
        avail, dc_id, dcs = client.check_l40s_availability()
        assert avail is False
        assert dc_id == ""
        assert len(dcs) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runpod_client.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.runpod_deploy')

- [ ] **Step 3: Implement `src/runpod_deploy/client.py`**

Implement:
- `resolve_runpod_key(config_file=None)`: Checks `os.environ["RUNPOD_API_KEY"]`, `.env`, and `~/.runpod/config.toml`.
- `resolve_github_token()`: Checks `os.environ["GITHUB_TOKEN"]`, `os.environ["GH_TOKEN"]`, `.env`, and executes `gh auth token`.
- `resolve_ssh_public_key(key_path=None)`: Checks common paths (`~/.ssh/id_ed25519.pub`, `~/.ssh/id_rsa.pub`) or user-supplied path.
- `RunPodClient`: Handles REST calls to `https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD`, `POST /v2/pods`, `GET /v2/pods/{id}`, `DELETE /v2/pods/{id}` using standard library `urllib.request`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runpod_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runpod_deploy/__init__.py src/runpod_deploy/client.py tests/test_runpod_client.py
git commit -m "feat(runpod): implement RunPod API client and catalog availability checker"
```

---

### Task 2: Dual-Transport SSH Connector & Remote Bootstrapper

**Files:**
- Create: `src/runpod_deploy/ssh_bootstrap.py`
- Test: `tests/test_runpod_ssh.py`

**Interfaces:**
- Consumes: Pod details dictionary from `RunPodClient.get_pod()`.
- Produces:
  - `SSHConnectionInfo(host: str, port: int, user: str, is_proxy: bool)`
  - `resolve_ssh_connection(pod_data: dict) -> SSHConnectionInfo`
  - `build_ssh_command(conn: SSHConnectionInfo, remote_cmd: str, identity_file: str | None = None) -> list[str]`
  - `SSHExecutor.run_streaming(conn: SSHConnectionInfo, remote_cmd: str, on_line: callable = None) -> int`
  - `SSHExecutor.bootstrap_pod(conn: SSHConnectionInfo, gh_token: str, repo_url: str, branch: str = "main") -> bool`
  - `SSHExecutor.wait_web_ready(conn: SSHConnectionInfo, timeout_sec: int = 120) -> bool`

- [ ] **Step 1: Write failing tests for SSH connection resolver and bootstrap commands**

```python
# tests/test_runpod_ssh.py
import pytest
from unittest.mock import patch, MagicMock
from src.runpod_deploy.ssh_bootstrap import (
    resolve_ssh_connection,
    build_ssh_command,
    SSHConnectionInfo,
    build_bootstrap_script,
)

def test_resolve_ssh_connection_direct():
    pod_data = {
        "id": "pod_xyz123",
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 14522, "ip": "194.26.1.10"}
            ]
        }
    }
    conn = resolve_ssh_connection(pod_data)
    assert conn.host == "194.26.1.10"
    assert conn.port == 14522
    assert conn.user == "root"
    assert conn.is_proxy is False

def test_resolve_ssh_connection_proxy():
    pod_data = {
        "id": "pod_xyz123",
        "runtime": {
            "ports": []  # No direct port 22 mapped
        },
        "ssh": {
            "proxy": {
                "host": "ssh.runpod.io",
                "port": 22,
                "user": "pod_xyz123-root"
            }
        }
    }
    conn = resolve_ssh_connection(pod_data)
    assert conn.host == "ssh.runpod.io"
    assert conn.port == 22
    assert conn.user == "pod_xyz123-root"
    assert conn.is_proxy is True

def test_build_ssh_command():
    conn = SSHConnectionInfo(host="194.26.1.10", port=14522, user="root", is_proxy=False)
    cmd = build_ssh_command(conn, "uptime", identity_file="/home/user/.ssh/id_ed25519")
    assert "ssh" in cmd[0]
    assert "-p" in cmd
    assert "14522" in cmd
    assert "root@194.26.1.10" in cmd
    assert "-i" in cmd
    assert "/home/user/.ssh/id_ed25519" in cmd
    assert "uptime" in cmd[-1]

def test_build_bootstrap_script_contains_ttyd_and_token():
    script = build_bootstrap_script(
        gh_token="secret_token",
        repo_url="https://github.com/ameerhossein/liquidwhisper-fa.git",
        branch="main"
    )
    assert "x-access-token" in script
    assert "ttyd" in script
    assert "8080" in script
    assert "setup_server.sh" in script
    assert "runpod_web.sh" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runpod_ssh.py -v`
Expected: FAIL (ImportError / NameError)

- [ ] **Step 3: Implement `src/runpod_deploy/ssh_bootstrap.py`**

Implement:
- `resolve_ssh_connection`: Looks at `runtime.ports` for private port 22, falling back to `ssh.proxy` or `ssh.direct` blocks.
- `build_ssh_command`: Assembles `ssh` arguments with `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10`.
- `build_bootstrap_script`: Returns remote bash script that:
  1. Installs static `ttyd` binary to `/usr/local/bin/ttyd` and runs `nohup ttyd -p 8080 -W bash > /workspace/ttyd.log 2>&1 &`.
  2. Clones or updates repository in `/workspace/stt` using `https://x-access-token:${GH_TOKEN}@...`.
  3. Copies `.env.example` to `.env` if `.env` does not exist.
  4. Runs `./setup_server.sh`.
  5. Runs `./runpod_web.sh -d` on port 7860.
- `SSHExecutor`: Runs commands over subprocess with live stdout line streaming and callback support.
- `wait_web_ready`: Probes `curl -s -f http://127.0.0.1:7860/` over SSH until HTTP 200.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runpod_ssh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runpod_deploy/ssh_bootstrap.py tests/test_runpod_ssh.py
git commit -m "feat(runpod): implement SSH dual-transport connector and remote bootstrapper"
```

---

### Task 3: Multimodal Notifications & Terminal UI Elements

**Files:**
- Create: `src/runpod_deploy/notify.py`
- Test: `tests/test_runpod_notify.py`

**Interfaces:**
- Produces:
  - `send_desktop_notification(title: str, message: str) -> bool`
  - `play_audio_chime() -> None`
  - `format_ready_banner(pod_id: str, web_cockpit_url: str, web_terminal_url: str, ssh_cmd: str) -> str`
  - `ProgressSpinner(label: str)` (context manager / interactive text updater)

- [ ] **Step 1: Write failing tests for notification and banner generation**

```python
# tests/test_runpod_notify.py
from unittest.mock import patch
from src.runpod_deploy.notify import (
    send_desktop_notification,
    play_audio_chime,
    format_ready_banner,
)

def test_send_desktop_notification_success():
    with patch("shutil.which", return_value="/usr/bin/notify-send"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert send_desktop_notification("Test Title", "Test Message") is True
        mock_run.assert_called_once()

def test_send_desktop_notification_missing_cli():
    with patch("shutil.which", return_value=None):
        assert send_desktop_notification("Test Title", "Test Message") is False

def test_format_ready_banner():
    banner = format_ready_banner(
        pod_id="pod_12345",
        web_cockpit_url="https://pod_12345-7860.proxy.runpod.net",
        web_terminal_url="https://pod_12345-8080.proxy.runpod.net",
        ssh_cmd="ssh root@1.2.3.4 -p 1234"
    )
    assert "https://pod_12345-7860.proxy.runpod.net" in banner
    assert "https://pod_12345-8080.proxy.runpod.net" in banner
    assert "ssh root@1.2.3.4 -p 1234" in banner
    assert "pod_12345" in banner
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runpod_notify.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `src/runpod_deploy/notify.py`**

Implement:
- `send_desktop_notification`: Calls `notify-send -u critical -i preferences-system "title" "message"` if `notify-send` is installed.
- `play_audio_chime`: Sends terminal bell `\a` to stdout and tries `paplay /usr/share/sounds/freedesktop/stereo/complete.oga` or `speaker-test` if available.
- `format_ready_banner`: Builds an ASCII border card containing Pod ID, Web Cockpit URL, Web Terminal URL, RunPod console URL, and SSH command.
- `ProgressSpinner`: Lightweight terminal spinner with elapsed timer and countdown for polling loops.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runpod_notify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runpod_deploy/notify.py tests/test_runpod_notify.py
git commit -m "feat(runpod): implement desktop notification, audio chime, and banner formatter"
```

---

### Task 4: Main CLI Watcher Script & Shell Launcher

**Files:**
- Create: `scripts/runpod_l40s_watch.py`
- Create: `runpod_watch.sh`
- Test: `tests/test_runpod_l40s_watch.py`

**Interfaces:**
- Consumes: `src.runpod_deploy.client`, `src.runpod_deploy.ssh_bootstrap`, `src.runpod_deploy.notify`.
- CLI Options:
  - `--interval`: Polling interval in seconds (default: 25).
  - `--cloud-type`: `SECURE` (default), `COMMUNITY`, or `ALL`.
  - `--container-disk`: Ephemeral root disk size in GB (default: 300).
  - `--volume-size`: Persistent volume size in GB (default: 200).
  - `--check-only`: Check L40S availability once and exit.
  - `--dry-run`: Poll and display provisioning parameters without creating real pod.
  - `--branch`: Git branch to clone (default: `main`).

- [ ] **Step 1: Write failing CLI integration tests**

```python
# tests/test_runpod_l40s_watch.py
import pytest
from unittest.mock import patch, MagicMock
from scripts.runpod_l40s_watch import main

def test_cli_check_only_available(capsys):
    with patch("src.runpod_deploy.client.resolve_runpod_key", return_value="mock_key"), \
         patch("src.runpod_deploy.client.RunPodClient.check_l40s_availability", return_value=(True, "US-NV-1", [{"id": "US-NV-1"}])):
        ret = main(["--check-only"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "L40S available in datacenter: US-NV-1" in captured.out

def test_cli_check_only_unavailable(capsys):
    with patch("src.runpod_deploy.client.resolve_runpod_key", return_value="mock_key"), \
         patch("src.runpod_deploy.client.RunPodClient.check_l40s_availability", return_value=(False, "", [])):
        ret = main(["--check-only"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "L40S is currently NOT available" in captured.out

def test_cli_dry_run(capsys):
    with patch("src.runpod_deploy.client.resolve_runpod_key", return_value="mock_key"), \
         patch("src.runpod_deploy.client.resolve_github_token", return_value="mock_gh_token"), \
         patch("src.runpod_deploy.client.resolve_ssh_public_key", return_value="ssh-ed25519 AAAAC3..."), \
         patch("src.runpod_deploy.client.RunPodClient.check_l40s_availability", return_value=(True, "US-NV-1", [{"id": "US-NV-1"}])):
        ret = main(["--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "300 GB" in captured.out
        assert "200 GB" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runpod_l40s_watch.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'scripts.runpod_l40s_watch')

- [ ] **Step 3: Implement `scripts/runpod_l40s_watch.py` and `runpod_watch.sh`**

Implement:
- CLI argument parsing (`argparse`).
- Early credentials pre-flight check (`RUNPOD_API_KEY`, `GITHUB_TOKEN`, SSH public key).
- Polling loop:
  - Prints current status, timestamp, and spinner.
  - Queries `client.check_l40s_availability()`.
  - If unavailable: sleeps `--interval` with interactive countdown.
  - If available: breaks loop and initiates deployment.
- Pod creation:
  - Invokes `client.create_pod` with GPU `NVIDIA L40S`, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, ports `["7860/http", "8080/http", "8888/http", "22/tcp"]`, container disk 300 GB, persistent volume 200 GB at `/workspace`.
- Readiness & Bootstrap:
  - Polls `client.get_pod` until running.
  - Waits for SSH connection.
  - Streams remote bootstrap execution.
  - Verifies Web Cockpit returns 200 at `http://127.0.0.1:7860/`.
- Notification & Banner:
  - Calls `send_desktop_notification`, `play_audio_chime`, and prints formatted banner.
- Shell launcher:
  - `runpod_watch.sh` chmod +x, sourcing `.venv` if available and passing through arguments to `python3 -m scripts.runpod_l40s_watch "$@"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runpod_l40s_watch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/runpod_l40s_watch.py runpod_watch.sh tests/test_runpod_l40s_watch.py
git commit -m "feat(runpod): implement CLI watcher script and shell launcher"
```

---

### Task 5: End-to-End Test Suite & Documentation

**Files:**
- Modify: `GUIDE.md`
- Modify: `README.md`
- Test: `tests/test_runpod_l40s_watch.py`

**Interfaces:**
- Comprehensive simulated end-to-end test verifying full lifecycle:
  `unavailable (poll 1) -> available (poll 2) -> pod create -> wait running -> ssh bootstrap -> health 200 -> notify & banner`.

- [ ] **Step 1: Add full lifecycle test to `tests/test_runpod_l40s_watch.py`**

```python
def test_full_lifecycle_simulation(capsys):
    poll_results = [
        (False, "", []),
        (True, "US-NV-1", [{"id": "US-NV-1"}])
    ]
    with patch("src.runpod_deploy.client.resolve_runpod_key", return_value="mock_key"), \
         patch("src.runpod_deploy.client.resolve_github_token", return_value="mock_gh_token"), \
         patch("src.runpod_deploy.client.resolve_ssh_public_key", return_value="ssh-ed25519 AAA..."), \
         patch("src.runpod_deploy.client.RunPodClient.check_l40s_availability", side_effect=poll_results), \
         patch("src.runpod_deploy.client.RunPodClient.create_pod", return_value={"id": "pod_live_123"}), \
         patch("src.runpod_deploy.client.RunPodClient.get_pod", return_value={
             "id": "pod_live_123",
             "desiredStatus": "RUNNING",
             "runtimeStatus": "running",
             "runtime": {"ports": [{"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4"}]}
         }), \
         patch("src.runpod_deploy.ssh_bootstrap.SSHExecutor.bootstrap_pod", return_value=True), \
         patch("src.runpod_deploy.ssh_bootstrap.SSHExecutor.wait_web_ready", return_value=True), \
         patch("src.runpod_deploy.notify.send_desktop_notification", return_value=True), \
         patch("src.runpod_deploy.notify.play_audio_chime"):
        
        from scripts.runpod_l40s_watch import main
        ret = main(["--interval", "0.01"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "RUNPOD L40S COCKPIT IS LIVE!" in captured.out
        assert "https://pod_live_123-7860.proxy.runpod.net" in captured.out
        assert "https://pod_live_123-8080.proxy.runpod.net" in captured.out
```

- [ ] **Step 2: Run test suite to verify all tests pass**

Run: `pytest tests/test_runpod_*.py -v`
Expected: ALL PASS

- [ ] **Step 3: Update `GUIDE.md` and `README.md`**

Document the new capability:
- Quick start with `./runpod_watch.sh`
- CLI options: `--interval`, `--check-only`, `--dry-run`, `--container-disk`, `--volume-size`
- Web terminal access via `https://<pod-id>-8080.proxy.runpod.net`
- Web Cockpit access via `https://<pod-id>-7860.proxy.runpod.net`
- Notes explaining the 300 GB container disk and 200 GB persistent storage rationale.

- [ ] **Step 4: Commit**

```bash
git add GUIDE.md README.md tests/test_runpod_l40s_watch.py
git commit -m "docs: add RunPod L40S automated watcher instructions and end-to-end tests"
```
