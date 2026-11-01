# RunPod L40S Automated Watcher & WebUI Launcher — Design

- Date: 2026-11-01
- Status: Draft for review
- Scope: GPU availability polling, RunPod API pod deployment, automated GitHub private clone, environment setup, Web Cockpit launch, desktop/audio notification

---

## 1. Intent

Provisioning high-demand GPUs like the NVIDIA L40S (48 GB VRAM) on RunPod often requires repeated manual checks across multiple datacenters due to intermittent availability. Once secured, setting up the repository, dependencies, and starting the web UI requires manual SSH steps.

This tool automates the entire lifecycle:
1. Continuously polls RunPod for **NVIDIA L40S** stock availability across datacenters.
2. The instant an L40S becomes available, automatically provisions a pod with exposed WebUI (`7860/http`) and SSH (`22/tcp`) ports.
3. Automatically authenticates and clones the private GitHub repository (`ameerhossein/liquidwhisper-fa`) on the remote pod using the local `gh` token.
4. Executes `./setup_server.sh` and boots the Persian ASR Web Cockpit via `./runpod_web.sh` in the background.
5. Verifies service readiness and immediately notifies the user via desktop alert (`notify-send`), terminal audio chime, and prints the direct HTTPS proxy link (`https://<pod-id>-7860.proxy.runpod.net`).

---

## 2. Goals

- **Automated Availability Watcher:** Poll RunPod REST v2 (`/v2/catalog/gpus?include=AVAILABILITY&product=POD`) and GraphQL fallback to check stock status for `NVIDIA L40S`.
- **Configurable Polling Interval:** Default to 25 seconds with clean progress output (spinner / timestamp / status message).
- **Zero-Manual-Intervention Provisioning:** Deploy an on-demand pod with:
  - GPU: `NVIDIA L40S` (count 1, 48 GB VRAM)
  - Base Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (matching Python 3.11 and CUDA 12.4 audio stack)
  - Exposed Ports:
    - `7860/http`: Persian ASR Web Cockpit
    - `8080/http`: Browser Web Terminal (`ttyd`)
    - `8888/http`: JupyterLab (optional)
    - `22/tcp`: Direct SSH access
  - Storage:
    - **Ephemeral container root disk (`disk`):** 300 GB (default, configurable via `--container-disk`). The codebase's paired training pipeline (`telephony_mode: paired`) builds a ~170 GB direct-shard Arrow cache (`src/prep_cache.py`) and downloads ~25–30 GB of raw audio/weights into the roomiest disk (`/tmp` or `/root/.cache/huggingface` via `src/cache.py`). The built-in `ensure_space()` guard enforces at least 180 GB free projected disk space before running `datasets.map`. 300 GB provides safe headroom for OS, CUDA, raw downloads, and the 170 GB shard cache without `OSError 28 (No space left on device)`.
    - **Persistent storage volume (`/workspace`):** 200 GB (default, configurable via `--volume-size`). Accommodates repository checkout, `.venv` with PyTorch cu124 (~8 GB), training checkpoints, LoRA adapters, merged standalone models (~3.1 GB each), and run ledgers across multiple runs. (Optionally allows pointing `ASR_PREP_CACHE_DIR=/workspace/stt-outputs/prep_cache` to preserve the 170 GB prep cache across pod teardowns).
- **Web Terminal Enablement:** Installs/runs lightweight `ttyd` daemon on port 8080 backgrounded, exposing an instant browser terminal at `https://<pod-id>-8080.proxy.runpod.net`.
- **Live Terminal Progress Reporting:** Real-time feedback in terminal:
  - Spinner, elapsed time, datacenter stock states, and retry countdown while watching for L40S.
  - Pod provisioning status & IP/port resolution.
  - SSH readiness ping attempts.
  - Remote setup progress streaming (git clone, virtualenv, dependencies, codec checks).
  - Health check verification loop polling `http://127.0.0.1:7860/`.
- **Seamless Private Repo Clone:** Seamlessly retrieve local GitHub credentials using `gh auth token` (falling back to `GITHUB_TOKEN` / `GH_TOKEN` environment variables) to clone `https://github.com/ameerhossein/liquidwhisper-fa.git` over HTTPS on the pod without exposing tokens in public logs.
- **Automated Bootstrap:** Execute `./setup_server.sh` remotely to build `.venv`, check telephony codecs, and install pinned dependencies, then invoke `./runpod_web.sh -d` to launch the detached background Uvicorn server.
- **Health Verification:** Probe the web server (`http://127.0.0.1:7860/`) over SSH or proxy URL until it returns HTTP 200.
- **Multimodal Notification:** Alert the user via:
  - System desktop notification (`notify-send` on Linux)
  - Audible bell (`\a` / `paplay` / `speaker-test` fallback)
  - Formatted terminal banner with direct WebUI proxy link and SSH command
- **CLI Options & Dry-Run:** Support `--check-only`, `--dry-run`, `--interval <sec>`, `--gpu <name>`, `--repo-url <url>`, `--branch <name>`.
- **Comprehensive Unit & Integration Tests:** Mocked RunPod API client, polling logic, SSH bootstrapping, and notification helpers.

---

## 3. Non-goals

- Launching serverless endpoints (this is a persistent pod service for the interactive web cockpit).
- Managing multi-pod clusters.
- Automatic shutdown/termination before the user uses the web cockpit (though optional `--terminate-after` or `--stop-after` flags can be supported).

---

## 4. Anti-goals

- Hardcoding API keys or GitHub tokens in script code or committed files.
- Hanging indefinitely on SSH connections without timeouts or retries.
- Claiming readiness before the Web UI actually answers HTTP requests.
- Swallowing RunPod API errors without printing clear remediation (e.g. insufficient funds, invalid API key).

---

## 5. Constraints

- **RunPod Authentication:** Requires `RUNPOD_API_KEY` in environment or in `.env` / `~/.runpod/config.toml`.
- **GitHub Authentication:** Requires active `gh auth status` or `GITHUB_TOKEN` in environment / `.env`.
- **SSH Key:** Local SSH public key (`~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`) must be registered on RunPod or injected during pod creation.
- **Python Standard Library:** Core launcher script should run with standard Python 3.10+ without requiring external third-party packages installed locally (optional `paramiko` fallback to standard system `ssh` command).
- **Cloud & Datacenter Targeting:** Catalog responses include datacenter breakdown. Provisioning targets the specific datacenter reporting stock (`dataCenterId`). Default cloud type is `SECURE` (enterprise), with `--cloud-type` option allowing `COMMUNITY` or `ALL`.
- **SSH Dual-Transport:** RunPod pods report either direct TCP connection (`ssh.direct: { ip, port }`) or proxied gateway (`ssh.proxy: { host, port, user }`). The SSH executor dynamically inspects `ssh` details from pod metadata and connects seamlessly to whichever transport is available.
- **Storage Sizing (Heavy Training Workload):**
  - **Ephemeral container root disk (`disk`):** Set to **300 GB**. The repository's paired training pipeline (`telephony_mode: paired`) generates a ~170 GB direct-shard Arrow dataset. `src/cache.py` directs HF downloads and `src/prep_cache.py` staging to the roomiest disk (`/tmp` or `/root`), and `ensure_space()` strictly aborts if free space is under 180 GB. 300 GB ensures full headroom during feature extraction and tokenization.
  - **Persistent Volume (`/workspace`):** Set to **200 GB** (`mounts.persistent.size = 200, path = "/workspace"`). Persists `.venv`, repository, multiple checkpoints, LoRA adapters, merged standalone models, and execution logs.
  - Both parameters are exposed as CLI flags (`--container-disk` and `--volume-size`).

---

## 6. Approaches Considered

### Approach 1: Self-Contained Python Watcher & Launcher (`scripts/runpod_l40s_watch.py`) — **Selected**
- **Pros:**
  - Robust structured JSON parsing from RunPod REST v2 and GraphQL endpoints.
  - Native integration with Python `.env` loaders and standard subprocess/HTTP libraries.
  - Testable with standard `pytest` unit test fixtures.
  - Graceful handling of SIGINT, retry backoff, and terminal UI formatting.
- **Cons:** None significant for this workflow.

### Approach 2: Bash Wrapper using `runpodctl` binary
- **Pros:** Uses official RunPod CLI.
- **Cons:**
  - `runpodctl` is not pre-installed in the user's local PATH.
  - Less portable across environments without installing additional binaries.
  - Brittle bash JSON parsing (`jq`) and complex retry loops.

---

## 7. Detailed Architecture & Workflow

```
[Start]
   │
   ▼
[Check Auth & Credentials]
   ├── RUNPOD_API_KEY (env or .env)
   ├── GITHUB_TOKEN (gh auth token or env)
   └── SSH Public Key (~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub)
   │
   ▼
[Poll RunPod Stock Loop]
   ├── GET /v2/catalog/gpus?include=AVAILABILITY&product=POD
   ├── Filter: id == "NVIDIA L40S"
   ├── Check: availability != "NONE" (HIGH, MEDIUM, LOW)
   ├── If NONE: sleep(--interval) -> retry with progress bar / countdown
   └── If Available: break loop!
   │
   ▼
[Provision Pod]
   ├── POST /v2/pods
   │     gpu: "NVIDIA L40S", count: 1
   │     image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
   │     ports: ["7860/http", "8080/http", "8888/http", "22/tcp"]
   │     disk: 300 GB
   │     mounts: { persistent: { size: 200, path: "/workspace" } }
   │     env: { PUBLIC_KEY: <ssh_pub_key> }
   └── Obtain pod ID & assigned host/ports
   │
   ▼
[Wait for Pod Readiness]
   ├── Poll GET /v2/pods/{id} until desiredStatus=RUNNING and runtimeStatus=running
   └── Test SSH connection (direct TCP or proxy gateway)
   │
   ▼
[Bootstrap over SSH]
   ├── Setup web terminal: install & launch ttyd on port 8080 (backgrounded)
   ├── Clone repo: git clone https://x-access-token:${GH_TOKEN}@github.com/ameerhossein/liquidwhisper-fa.git /workspace/stt
   ├── Run: cd /workspace/stt && ./setup_server.sh
   └── Start Cockpit: ./runpod_web.sh -d
   │
   ▼
[Verify Web Cockpit Response]
   └── Poll http://127.0.0.1:7860/ over SSH / curl until HTTP 200
   │
   ▼
[Notify User]
   ├── notify-send "RunPod L40S Ready" "Web Cockpit: https://<pod-id>-7860.proxy.runpod.net"
   ├── Terminal audio chime (\a)
   └── Display banner with Cockpit URL, Web Terminal URL, and SSH command
```

---

## 8. Notification & User Experience

When the pod is deployed and ready:
1. **Desktop notification**: Triggered via `notify-send -i computer -u critical "RunPod L40S WebUI Ready" "Direct URL: https://<pod-id>-7860.proxy.runpod.net"` (non-blocking if not on Linux or if display server not active).
2. **Audible alert**: Terminal bell `print("\a", flush=True)` plus `paplay /usr/share/sounds/freedesktop/stereo/complete.oga` if available.
3. **Console banner**:
```text
========================================================================
🚀 RUNPOD L40S COCKPIT IS LIVE!
========================================================================
Web Cockpit URL: https://<pod-id>-7860.proxy.runpod.net
Web Terminal:    https://<pod-id>-8080.proxy.runpod.net
RunPod Console:  https://console.runpod.io/pod/<pod-id>
SSH Access:      ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
Working Dir:     /workspace/stt
Logs:            /workspace/stt/logs/runpod_web.log
========================================================================
```

---

## 9. What's Left for the Builder

1. Implement `scripts/runpod_l40s_watch.py` with clean modular functions (`check_credentials`, `get_l40s_availability`, `create_pod`, `wait_pod_ready`, `ssh_exec`, `notify_user`).
2. Add standalone CLI runner / shell wrapper `./runpod_watch.sh`.
3. Add tests in `tests/test_runpod_l40s_watch.py` verifying stock parsing, API calls, error backoff, and command assembly.
4. Update `GUIDE.md` and `README.md` with instructions on running the watcher script.
