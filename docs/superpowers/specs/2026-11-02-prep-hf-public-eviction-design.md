# Fresh-pod paired prep: public HF repo, local-first eviction

## Intent

Run `scripts/prepare_paired_cache.py all` on a fresh pod and land the final
`train/` cache in a **public** Hugging Face dataset repo, without hitting the
free tier's 100 GB private quota and without transferring the 84 GB clean half
to HF when the local page cache can be evicted instead.

## Goals

- Support a public dataset repo explicitly, never by silently changing an
  existing repo's visibility.
- Fail fast when a private repo provably cannot hold the assembled cache,
  instead of failing after the final upload starts.
- Evict the clean half's page cache without root before falling back to HF.
- Delete `clean/` from the repo once assembly has succeeded locally.

## Non-goals

- No WebUI or cockpit fields. `.env` only.
- No change to the trainer's `ASR_PREP_HF_REPO` download path.
- No automatic flip of an existing repo between private and public.

## Anti-goals

- Publishing to a public repo because a default drifted.
- Losing the only copy of the clean half before assembly succeeds.
- A rerun silently doubling stored bytes through new LFS objects.

## Constraints

- `huggingface_hub` 0.36.2: `HfApi.delete_folder` and
  `os.posix_fadvise(os.POSIX_FADV_DONTNEED)` are available.
- Free HF private storage is 100 GB; PRO is 1 TB.
- Assembled `train/` is ~168 GB; each half is ~84 GB; cgroup is 188 GB.
- Container images deny writes to `/proc/sys/vm/drop_caches`, so the current
  `auto` path always falls through to a full HF round trip.

## Approaches

1. **PRO + private repo.** Meets the quota, costs $9/mo, keeps the derived
   cache private. Rejected: user chose the free public path.
2. **Public repo, keep uploading `clean/`.** Works within quota but moves
   ~168 GB of avoidable traffic per run and leaves 84 GB of dead weight.
3. **(Chosen) Public repo + per-file eviction + remote cleanup.** Repo holds
   only `train/` (~168 GB public); `clean/` leaves the pod only when per-file
   eviction demonstrably fails.

## Decisions

### Visibility

- `ASR_PREP_HF_PUBLIC=1` (or `--public-repo`) creates the dataset repo as
  public. Default remains private, so nothing is published by accident.
- An existing repo is never flipped. The helper reads the actual `private`
  flag, prints it, and warns on a mismatch.
- A private repo is refused before the clean offload when
  `2.1 x clean_size` exceeds the private limit (default 100 GB, override
  `ASR_PREP_HF_PRIVATE_LIMIT_GB`). Every upload re-checks the stage size
  against the same limit.

### Eviction ladder (`--clean-strategy auto`)

1. `sync` + global `drop_caches`.
2. If denied: `posix_fadvise(DONTNEED)` on every `.arrow` file in the clean
   stage.
3. Re-read cgroup usage. Keep the clean stage local when usage is below
   `--max-pre-phone-gb`; otherwise upload to HF, delete locally, run phone,
   download back, assemble.

Eviction is advisory, so the cgroup measurement, not the call's success,
decides. The clean stage is only deleted after a confirmed upload.

### Remote cleanup

- After `assemble_stages` succeeds and the final upload completes, delete
  `clean/` from the repo (default on; `--keep-hf-clean` opts out).
- A failed delete is a warning, not a failed run: the published cache is the
  deliverable.
- Documented: re-uploading the same path creates new LFS objects, so use a
  fresh repo or `super_squash_history` between attempts.

## Left to the builder

- Exact log wording.
- Whether to factor the cgroup re-read into its own helper.
- Test fixture shapes beyond the named behaviours.
