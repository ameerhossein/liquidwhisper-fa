# HF-Backed Persian ASR Training — Operator Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to work through this runbook step by step.
> Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For the operator:** this is an ops runbook, not a code change. Work top to
> bottom and stop at the first failed check. Every command is meant to be run
> from `/workspace/stt` on the GPU pod unless the step says "local machine".

**Spec:** `README.md` (training flow), `PRODUCTION.md` (prep cache and HF
mirror), `.env.example` (environment surface).

**Goal:** produce a trained Persian ASR LoRA adapter and a standalone merged
model, with the one-time prepared-feature cache stored in a Hugging Face
dataset repo and every training artifact mirrored to a Hugging Face model repo.

**Architecture:** the 168 GB paired prep cache is built **on the GPU pod**
(HF is storage, not compute), uploaded to a public HF dataset repo, and loaded
by the trainer on demand. Training uses `BuzzASR/persian` + `shekar-ai/Neyshekar`
with the L40S defaults (BF16 LoRA, paired clean+phone, 2 epochs). The run
mirror writes checkpoints, adapter snapshots, `best_adapter/`, reports, and the
merged model to a private HF model repo.

**Target HF repos:**

| Purpose | Repo | Visibility |
|---|---|---|
| Prep cache (`train/`, ~168 GB) | `ameerhossein/liquidwhisper-neyshekar-prep` | **public** (free-tier limit is 100 GB private) |
| Run mirror (checkpoints, adapter, reports, merged) | `ameerhossein/liquidwhisper-neyshekar` | private |
| Clean final merged model (optional, recommended) | `ameerhossein/liquidwhisper-neyshekar-merged` | private |

**Global constraints:**

- GPU: NVIDIA L40S 48 GB; pod disk: 300 GB container + 200 GB `/workspace`.
- Prep cache goes on the **container disk** (`/root/prep_cache`), never the
  200 GB persistent volume.
- Prep and training must use the same data settings: `paired`, profiles
  `customer`, seed `42`, ffmpeg on, no `--max_train_samples`.
- `HF_TOKEN` must have write access to all three repos.
- Current checkout has uncommitted changes. The RunPod watcher clones
  `origin/main`; either commit/push first or rsync the working tree.
- Do not put `ASR_COCKPIT_TOKEN` in `.env` if you will use the cockpit buttons:
  the current UI does not prompt for it and Start would return 401.

**Review focus (most likely to bite):**

1. The pod runs different code than the local working tree because the watcher
   clones GitHub — verify the prep-cache format and trainer version match.
2. The prep repo is public on the free tier; confirm that derived features and
   tokenized labels are acceptable before the first upload.
3. Prep fills the container disk, not `/workspace`; a wrong `ASR_PREP_CACHE_DIR`
   or `--prep-cache-dir` ends in `OSError 28`.
4. The trainer refuses a HF cache whose fingerprint differs from its own
   settings; prep and training must use the same telephony flags.
5. Setting `ASR_COCKPIT_TOKEN` locks the cockpit UI out of Start/Stop/Merge
   because the UI does not send the token.

---

## Phase 0 — Pre-flight (local machine)

- [ ] **Confirm the training code that will run on the pod.** The watcher clones
  GitHub, so uncommitted local files are not included.

```bash
cd /home/ameer/Projects/stt
git status --short
# If the current changes are intentional, commit and push:
git add -A && git commit -m "chore: prepare HF-backed training run"
git push origin main
```

  If you do not want to commit, use a manually created pod and rsync the
  working tree instead of the watcher (the watcher always clones GitHub).

- [ ] **Confirm the HF repos exist and the visibility is right.** In the HF UI:
  - `ameerhossein/liquidwhisper-neyshekar-prep` → **Dataset**, **public**.
  - `ameerhossein/liquidwhisper-neyshekar` → **Model**, private.
  - `ameerhossein/liquidwhisper-neyshekar-merged` → **Model**, private (optional).

  If the prep repo does not exist, the prep script can create it public because
  `ASR_PREP_HF_PUBLIC=1`; creating it by hand first avoids surprises.

- [ ] **Confirm the local `.env` has the RunPod key and GitHub auth works.**

```bash
cd /home/ameer/Projects/stt
cat .env                 # should contain RUNPOD_API_KEY=...
gh auth status           # watcher uses gh auth token
ls ~/.ssh/id_*.pub       # watcher injects the public key
```

- [ ] **Check L40S availability and the planned pod size.**

```bash
./runpod_watch.sh --check-only
./runpod_watch.sh --dry-run
```

  Defaults are 300 GB container disk and 200 GB `/workspace`. Do not shrink the
  container disk below 300 GB; prep writes ~170 GB of Arrow shards plus raw
  audio/model caches.

---

## Phase 1 — Launch the pod

- [ ] **Start the watcher.** It polls until an L40S is available, then clones
  the repo, runs `setup_server.sh`, and starts the cockpit.

```bash
cd /home/ameer/Projects/stt
./runpod_watch.sh
```

  Write down the three URLs and the SSH command it prints:

  - Cockpit: `https://<pod-id>-7860.proxy.runpod.net`
  - Web terminal: `https://<pod-id>-8080.proxy.runpod.net`
  - SSH: `ssh root@<host> -p <port> -i <key>`

- [ ] **Open a shell on the pod** (web terminal or SSH) and confirm the repo and
  GPU are present.

```bash
cd /workspace/stt
git log --oneline -1
nvidia-smi
df -h /root /workspace
```

  Expected: L40S 48 GB, `/root` ~300 GB free, `/workspace` ~200 GB.

---

## Phase 2 — Put the training `.env` on the pod

- [ ] **Create `/workspace/stt/.env`.** Use `.env.training.example` from this
  repo as the template and replace `HF_TOKEN` (and the repo names if you are
  not using `ameerhossein/...`). The full content is reproduced at the end of
  this plan. If the file is not on the pod yet (the watcher cloned the last
  pushed commit), create it from the block below.

```bash
cd /workspace/stt
nano .env          # or: cat > .env <<'EOF' ... EOF
chmod 600 .env
set -a; . ./.env; set +a   # load it into this shell for the snippets below
```

  Values that matter most:

  - `HF_TOKEN=hf_...`
  - `ASR_PREP_HF_REPO=ameerhossein/liquidwhisper-neyshekar-prep`
  - `ASR_PREP_HF_PUBLIC=1`
  - `ASR_PREP_CACHE_DIR=/root/prep_cache`
  - `ASR_HF_RUN_REPO=ameerhossein/liquidwhisper-neyshekar`

- [ ] **Verify HF authentication and write access.**

```bash
cd /workspace/stt
source .venv/bin/activate
set -a; . ./.env; set +a
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
print("authenticated as:", api.whoami()["name"])
for repo, kind in [
    ("ameerhossein/liquidwhisper-neyshekar-prep", "dataset"),
    ("ameerhossein/liquidwhisper-neyshekar", "model"),
]:
    try:
        info = api.repo_info(repo, repo_type=kind)
        print(f"{repo}: exists, private={info.private}")
    except Exception as exc:
        print(f"{repo}: not found yet ({exc}); it will be created")
PY
```

- [ ] **Restart the cockpit so it sees the new `.env`.** The watcher started it
  before this file existed.

```bash
cd /workspace/stt
./runpod_web.sh -d
curl -s localhost:7860/api/hf/config | python -m json.tool
```

  Expected: `"enabled": true`, `"repo_id": "ameerhossein/liquidwhisper-neyshekar"`,
  and `runs_root` under `/workspace/stt-outputs/results/runs`.

---

## Phase 3 — Smoke test (15 minutes, recommended)

This proves the audio stack, ffmpeg telephony, CUDA, and trainer wiring before
a multi-hour prep and a ~10-hour training run. It deliberately disables HF so it
cannot pollute the real repos.

- [ ] **Tiny prep of 32 rows.**

```bash
cd /workspace/stt
source .venv/bin/activate
python scripts/prepare_paired_cache.py clean \
  --prep-cache-dir /root/prep_smoke \
  --max-train-samples 32 \
  --num-proc 2 \
  --force
python scripts/prepare_paired_cache.py phone \
  --prep-cache-dir /root/prep_smoke \
  --max-train-samples 32 \
  --num-proc 2 \
  --force
rm -rf /root/prep_smoke
```

  Expected: `[clean] OK rows=32 ...` and `[phone] OK rows=32 ...`. This checks
  dataset download, audio decode, normalizer, label tokenization, and the
  ffmpeg telephony augmentation path.

- [ ] **Tiny train (32 rows, 20-row preflight/benchmark).**

```bash
cd /workspace/stt
source .venv/bin/activate
env ASR_PREP_HF_REPO=off ASR_HF_RUN_REPO=off \
python scripts/train_lora.py \
  --base_model BuzzASR/persian \
  --dataset_name shekar-ai/Neyshekar \
  --output_dir /root/smoke/output \
  --results_dir /root/smoke/results \
  --prep_cache_dir /root/smoke/prep \
  --max_train_samples 32 \
  --max_eval_samples 16 \
  --preflight_samples 20 \
  --max_benchmark_samples 20 \
  --epochs 1 \
  --eval_steps 50 \
  --save_steps 50 \
  --num_proc 2 \
  --no-gradient_checkpointing \
  --no-run_epoch_benchmark \
  --no-run_post_test_eval \
  --no_hf_sync
rm -rf /root/smoke
```

  Expected: preflight completes, one short epoch trains, adapter is written.
  This is also the only place where a one-epoch run is acceptable; the real run
  uses two paired epochs.

---

## Phase 4 — Build the paired prep cache and publish it to HF

This is the long one-time step: 116,488 paired rows, ~168 GB final cache.
The script prepares `clean` and `phone` halves in separate processes, evicts or
offloads the clean half when needed, assembles them with hardlinks, publishes
`/root/prep_cache/train`, and uploads `train/` to HF.

- [ ] **Start it in tmux so an SSH disconnect cannot kill it.**

```bash
cd /workspace/stt
tmux new -s prep
source .venv/bin/activate
mkdir -p /root/prep_cache
set -a; . ./.env; set +a
python scripts/prepare_paired_cache.py all \
  --prep-cache-dir /root/prep_cache \
  --num-proc 8 \
  2>&1 | tee /root/prep_cache/prep.log
```

  Detach with `Ctrl+B`, then `D`; reattach with `tmux attach -t prep`.

- [ ] **Monitor disk, memory, and progress.**

```bash
tail -f /root/prep_cache/prep.log
df -h /root /workspace
free -g
```

  What to expect in the log:

  - `[clean] ... preparing to /root/prep_cache/clean`
  - `[all] clean done; cgroup usage=...`
  - either `dropped local page cache` / `evicted ...` or `offloading to HF`
  - `[phone] ...`
  - `[all] assembling clean+phone`
  - `[all] published /root/prep_cache/train rows=116488`
  - `[hf] uploading .../train -> ameerhossein/liquidwhisper-neyshekar-prep/train`
  - `[hf] upload complete`

  If the pod is memory-limited, rerun with `--num-proc 4`. If a phase dies,
  recover with the explicit `clean`, `phone`, or `assemble` subcommands rather
  than `all`; `all` refuses an existing stage without `--force`.

- [ ] **Validate the local cache.** The trainer will reject a cache whose
  fingerprint does not match its own settings.

```bash
python - <<'PY'
import json
p = "/root/prep_cache/train/meta.json"
d = json.load(open(p))
print("format:", d["format"])
print("rows:", d["total_rows"])
print("shards:", len(d["shards"]))
print("fingerprint:", json.dumps(d["fingerprint"], indent=2))
PY
```

  Expected: `format=shards_v1`, `rows=116488`, and a fingerprint with
  `"telephony_mode": "paired"`, `"telephony_profiles": ["customer"]`,
  `"telephony_seed": 42`, `"telephony_use_ffmpeg": true`.

- [ ] **Verify the remote copy.**

```bash
cd /workspace/stt
source .venv/bin/activate
set -a; . ./.env; set +a
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "ameerhossein/liquidwhisper-neyshekar-prep"
info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
files = {s.rfilename: (s.size or 0) for s in info.siblings}
print("private:", info.private)
print("train/meta.json present:", "train/meta.json" in files)
print("train files:", sum(f.startswith("train/") for f in files))
print("train GB:", round(sum(v for k, v in files.items() if k.startswith("train/")) / 1e9, 1))
PY
```

  Expected: public repo, `train/meta.json` present, ~168 GB of `train/` files.

- [ ] **Choose the training load path.**

  - **Fast path (same pod):** keep `/root/prep_cache/train` and start training.
    The trainer loads the local cache; HF is the durable backup for future pods.
  - **True HF-load path (matches the "load from HF" requirement):** delete the
    local copy first, so the trainer downloads `train/` from HF exactly once.

```bash
# Only for the true HF-load path:
rm -rf /root/prep_cache/train /root/prep_cache/clean /root/prep_cache/phone
df -h /root
```

  Do not delete `/root/prep_cache/train` while training is running.

---

## Phase 5 — Train

Use the cockpit for the real run: it creates a self-contained run directory,
writes the ledger, and offers Resume/Merge. The CLI equivalent follows.

### Primary: cockpit

- [ ] **Open the cockpit URL** and confirm the HF line shows the run repo.
  If the page was already open before the `.env` restart, reload it.

- [ ] **Set the run configuration:**
  - Data: **Paired clean + phone**
  - Epochs: **2**
  - Batch size: **8**; grad accumulation: **4** (effective 32)
  - Learning rate: **1e-4**
  - Eval/save steps: **1000 / 1000**
  - Preflight samples: **1000**; max benchmark samples: **1000**
  - Gradient checkpointing: **off**

  The mandatory preflight is the baseline measurement; you do not need a
  separate `run_baseline.sh` run unless you want the full 2,149-row comparison
  outside training.

- [ ] **Click Start** and confirm the stages advance:
  `preflight → training → post_eval`; `merge` stays pending.

- [ ] **Monitor the run from the pod.**

```bash
RUN_DIR="$(find /workspace/stt-outputs/results/runs -maxdepth 1 -type d \
  -regextype posix-extended -regex '.*/[0-9]{8}-[0-9]{6}-[0-9a-f]{6}' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
echo "$RUN_DIR"
tail -f "${RUN_DIR}/training_live.log"
watch -n 30 'df -h /workspace; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

  Expected: `Prep Cache: downloading train/ from HF ...` if you chose the true
  HF-load path, then preflight, then optimizer steps. The validation split is
  small and is prepared locally on the first pod; that is expected.

### CLI equivalent

```bash
cd /workspace/stt
tmux new -s train
source .venv/bin/activate
./run_train.sh
```

  `run_train.sh` sources `.env` (so HF mirroring is on), uses
  `ASR_OUTPUT_DIR=/workspace/stt-outputs/buzzasr-neyshekar-lora`, and
  `ASR_HF_RUN_REPO` makes the trainer allocate a run directory under
  `ASR_RUNS_DIR` for the ledger and mirror. Attach with `tmux attach -t train`.

- [ ] **If the pod dies, resume from HF.** In the cockpit use
  **Recover → Restore & resume**. From the CLI:

```bash
cd /workspace/stt
source .venv/bin/activate
python scripts/hf_restore.py --run latest
./run_train.sh --hf_resume latest
```

  A run with a complete checkpoint resumes from the newest step. A completed
  run is restore-only.

---

## Phase 6 — Merge and publish the final model

Training completion uploads `best_adapter/` and the final reports. Merging is a
separate step.

- [ ] **Cockpit path:** after the training stage is completed and
  `best_adapter/adapter_config.json` exists, click **Merge**. It writes
  `<run>/merged/` and uploads it to the run mirror repo.

- [ ] **CLI path:** find the run id and merge with the mirror context.

```bash
cd /workspace/stt
source .venv/bin/activate
set -a; . ./.env; set +a
RUN_ID="$(python - <<'PY'
import json
from huggingface_hub import hf_hub_download
p = hf_hub_download("ameerhossein/liquidwhisper-neyshekar", "latest.json")
print(json.load(open(p))["run_id"])
PY
)"
RUN_DIR="/workspace/stt-outputs/results/runs/${RUN_ID}"

# Cockpit run: adapter under <run>/output. CLI run: adapter under ASR_OUTPUT_DIR.
ADAPTER="${RUN_DIR}/output/best_adapter"
[ -f "${ADAPTER}/adapter_config.json" ] || \
  ADAPTER="/workspace/stt-outputs/buzzasr-neyshekar-lora/best_adapter"

python scripts/merge_lora.py \
  --base_model BuzzASR/persian \
  --adapter_path "${ADAPTER}" \
  --output_dir "${RUN_DIR}/merged" \
  --run_dir "${RUN_DIR}" \
  --hf_run_repo ameerhossein/liquidwhisper-neyshekar
```

- [ ] **Publish a clean, directly loadable merged model repo.** The run mirror
  stores the merged model under `runs/<run_id>/merged/`; this step puts it at
  the root of a dedicated model repo.

```bash
python scripts/upload_to_hf.py \
  --artifact merged \
  --merged-path "${RUN_DIR}/merged" \
  --output-dir "${RUN_DIR}/output" \
  --results-dir "${RUN_DIR}" \
  --merged-repo ameerhossein/liquidwhisper-neyshekar-merged \
  --private
```

  For a CLI run, replace the `--output-dir`/`--results-dir` values with
  `/workspace/stt-outputs/buzzasr-neyshekar-lora` and
  `/workspace/stt-outputs/results`.

---

## Phase 7 — Verify the HF artifacts

- [ ] **List the run mirror.**

```bash
cd /workspace/stt
source .venv/bin/activate
set -a; . ./.env; set +a
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "ameerhossein/liquidwhisper-neyshekar"
files = api.list_repo_files(repo)
run_ids = sorted({f.split("/")[1] for f in files if f.startswith("runs/")})
print("runs:", run_ids[-5:])
run_id = run_ids[-1]
for name in ("run.json", "hf_state.json", "best_adapter/adapter_config.json",
             "merged/config.json", "reports/final_evaluation_report.json"):
    path = f"runs/{run_id}/{name}"
    print(("OK  " if path in files else "MISS"), path)
PY
```

- [ ] **Verify the final merged model repo loads.** On any machine with
  `transformers`:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
repo = "ameerhossein/liquidwhisper-neyshekar-merged"
processor = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo)
print(model.config._name_or_path, processor.tokenizer.vocab_size)
```

- [ ] **Download the final model locally.**

```bash
cd /workspace/stt
set -a; . ./.env; set +a
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ameerhossein/liquidwhisper-neyshekar-merged",
    local_dir="/workspace/models/buzzasr-persian-final",
)
PY
```

- [ ] **Record the run id and metrics** from
  `runs/<run_id>/reports/final_evaluation_report.json` (WER/CER, formal vs
  informal, phone vs clean).

---

## Phase 8 — Cleanup and cost control

- [ ] **Stop the pod** when training, merge, and download are done. The pod
  bills while it is running; the HF mirror and `/workspace` volume are the
  durable state.

- [ ] **Free the ephemeral prep cache** on the pod before stopping if you need
  disk; it can be re-downloaded from HF:

```bash
rm -rf /root/prep_cache /root/hf_cache
```

- [ ] **Remote retention is manual.** The run mirror never prunes. Delete old
  `runs/<run_id>/` folders, old checkpoints, and old adapter snapshots in the HF
  UI/CLI when the private repo approaches 100 GB. Keep the final merged repo
  separate so cleanup cannot delete it.

- [ ] **Privacy note:** the prep repo is public because the 168 GB cache exceeds
  the free private allowance. It contains derived features and tokenized labels,
  not raw audio; if that is not acceptable, use a Pro/private repo and set
  `ASR_PREP_HF_PUBLIC=0`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `private HF repo ... cannot hold ...G` | Prep repo is private and the cache is ~168 GB | Make the repo public in HF, or set `ASR_PREP_HF_PUBLIC=1` and delete the existing private repo so it is recreated public |
| `HF prep cache ... does not match this run's train fingerprint` | Prep and training data flags differ | Rerun prep with the same defaults (`paired`, customer, seed 42, ffmpeg on, no sample cap) or remove `ASR_PREP_HF_REPO` for a local re-prep |
| Training starts a full local prep despite the HF repo | `.env` not loaded, wrong repo id, or local cache is invalid | Check `echo $ASR_PREP_HF_REPO`; run Python entrypoints from `/workspace/stt`; look for `Prep cache: downloading train/ from HF` in the log |
| Cockpit HF line is empty | Server started before `.env` existed | `./runpod_web.sh -d` and reload the page |
| `Start failed: 401` | `ASR_COCKPIT_TOKEN` is set but the UI does not send it | Remove/comment `ASR_COCKPIT_TOKEN` and restart the cockpit, or drive the API with `X-Auth-Token` |
| `OSError 28` / disk full | Prep cache or HF cache on `/workspace` | Keep `ASR_PREP_CACHE_DIR=/root/prep_cache` and `ASR_HF_CACHE_DIR=/root/hf_cache`; check `df -h /root /workspace` |
| Prep killed during phone phase | cgroup memory pressure | Rerun with `--num-proc 4`; `auto` already evicts or offloads the clean half |
| Upload fails after assembly | Network/HF outage | The local `train/` is intact; re-upload it directly (below) |
| Pod clone lacks local changes | Watcher clones `origin/main` | Commit/push, or rsync the working tree to `/workspace/stt` |

Manual re-upload of an assembled cache:

```bash
cd /workspace/stt
set -a; . ./.env; set +a
python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id="ameerhossein/liquidwhisper-neyshekar-prep",
    repo_type="dataset",
    folder_path="/root/prep_cache/train",
    path_in_repo="train",
    commit_message="Re-upload assembled paired prep cache",
)
PY
```

---

## The required `.env`

This is also saved as `.env.training.example` in the repo. Put it at
`/workspace/stt/.env` on the pod with `chmod 600`.

```dotenv
# --- Hugging Face auth ------------------------------------------------------
HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN

# --- 1. Prepared-feature cache (HF dataset repo) ---------------------------
ASR_PREP_HF_REPO=ameerhossein/liquidwhisper-neyshekar-prep
ASR_PREP_HF_PUBLIC=1

# Prep work on the big container disk, not the 200 GB /workspace volume.
ASR_PREP_CACHE_DIR=/root/prep_cache
ASR_HF_CACHE_DIR=/root/hf_cache
HF_DATASETS_CACHE=/root/hf_cache/datasets

# --- 2. Durable run mirror (HF model repo) ---------------------------------
ASR_HF_RUN_REPO=ameerhossein/liquidwhisper-neyshekar

# --- Persistent paths on /workspace ----------------------------------------
ASR_OUTPUT_DIR=/workspace/stt-outputs/buzzasr-neyshekar-lora
ASR_RESULTS_DIR=/workspace/stt-outputs/results
ASR_RUNS_DIR=/workspace/stt-outputs/results/runs
ASR_MERGED_DIR=/workspace/stt-outputs/buzzasr-persian-merged

# --- Web cockpit ------------------------------------------------------------
PORT=7860
HOST=0.0.0.0
# Leave unset for the cockpit buttons; set only for curl/API with X-Auth-Token.
# ASR_COCKPIT_TOKEN=...

# --- Local machine only (not on the pod) -----------------------------------
# RUNPOD_API_KEY=rpa_REPLACE_WITH_YOUR_KEY
```

**Local machine `.env`:** only `RUNPOD_API_KEY` is required. The watcher does
not copy this file to the pod; create the pod `.env` separately as above.
