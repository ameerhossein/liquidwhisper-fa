import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset

from scripts import prepare_paired_cache as ppc
from src import prep_cache


def _stage(path: Path, labels, fingerprint=None, *, variant=None, source="test-source"):
    path.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict(
        {
            "input_features": [[float(i)] for i in labels],
            "labels": [[int(i)] for i in labels],
            "is_informal": [False for _ in labels],
        }
    )
    ds.save_to_disk(str(path))
    arrow = next(path.glob("*.arrow"))
    loaded = Dataset.from_file(str(arrow))
    ppc.write_stage_manifest(
        path, loaded, variant=variant or path.name,
        fingerprint=fingerprint if fingerprint is not None else {"x": 1},
        source_fingerprint=source,
    )
    return loaded


def test_assemble_stages_keeps_clean_then_phone_order(tmp_path):
    fingerprint = {"dataset_name": "test", "split": "train"}
    _stage(tmp_path / "clean", [1, 2], fingerprint)
    _stage(tmp_path / "phone", [1, 2], fingerprint)

    meta = ppc.assemble_stages(
        tmp_path / "clean",
        tmp_path / "phone",
        tmp_path / "prep" / "train",
        fingerprint,
        force=True,
    )

    assert meta["total_rows"] == 4
    loaded = prep_cache.load_prep_cache(tmp_path / "prep" / "train", fingerprint)
    assert loaded is not None
    assert loaded["labels"] == [[1], [2], [1], [2]]


def test_assemble_rejects_row_count_mismatch(tmp_path):
    _stage(tmp_path / "clean", [1, 2])
    _stage(tmp_path / "phone", [1])

    with pytest.raises(ValueError, match="row count"):
        ppc.assemble_stages(
            tmp_path / "clean",
            tmp_path / "phone",
            tmp_path / "prep" / "train",
            {"x": 1},
            force=True,
        )


def test_assemble_rejects_unpaired_rows(tmp_path):
    _stage(tmp_path / "clean", [1, 2])
    _stage(tmp_path / "phone", [1, 3])

    with pytest.raises(ValueError, match="labels"):
        ppc.assemble_stages(
            tmp_path / "clean",
            tmp_path / "phone",
            tmp_path / "prep" / "train",
            {"x": 1},
            force=True,
        )


def test_build_fingerprint_matches_paired_training_contract(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.train_lora._label_format",
        lambda _model: ((51908, 51968, 51972), 51865),
    )
    ns = SimpleNamespace(
        base_model="BuzzASR/persian",
        dataset_name="shekar-ai/Neyshekar",
        telephony_profiles=["customer"],
        telephony_seed=42,
        telephony_use_ffmpeg=True,
        max_train_samples=None,
    )
    assert ppc.build_fingerprint(ns) == {
        "dataset_name": "shekar-ai/Neyshekar",
        "base_model": "BuzzASR/persian",
        "informal_source": "MahtaFetrat/GPTInformal-Persian",
        "informal_mix_version": "gptinformal_v1",
        "split": "train",
        "max_samples": None,
        "telephony_mode": "paired",
        "telephony_prob": None,
        "telephony_profiles": ["customer"],
        "telephony_seed": 42,
        "telephony_use_ffmpeg": True,
        "telephony_rng": "per_row_index_v1",
        "label_ids": [51908, 51968, 51972, 51865],
    }


@pytest.mark.parametrize("changed", ["base_model", "dataset_name", "telephony_seed", "telephony_profiles"])
def test_assemble_rejects_half_prepared_with_different_settings(tmp_path, changed):
    fingerprint = {"base_model": "m", "dataset_name": "d", "telephony_seed": 42,
                   "telephony_profiles": ["customer"]}
    _stage(tmp_path / "clean", [1, 2], fingerprint)
    _stage(tmp_path / "phone", [1, 2], {**fingerprint, changed: "different"})
    with pytest.raises(ValueError, match="preparation fingerprint mismatch"):
        ppc.assemble_stages(tmp_path / "clean", tmp_path / "phone", tmp_path / "train", fingerprint)
    assert (tmp_path / "clean" / ppc.MANIFEST).exists()
    assert (tmp_path / "phone" / ppc.MANIFEST).exists()
    assert not (tmp_path / "train").exists()


def test_assemble_rejects_wrong_variant_and_changed_source(tmp_path):
    _stage(tmp_path / "clean", [1, 2])
    _stage(tmp_path / "phone", [1, 2], variant="clean")
    with pytest.raises(ValueError, match="phone stage has variant"):
        ppc.assemble_stages(tmp_path / "clean", tmp_path / "phone", tmp_path / "train", {"x": 1})

    import json
    path = tmp_path / "phone" / ppc.MANIFEST
    manifest = json.loads(path.read_text())
    manifest.update(variant="phone", source_fingerprint="different-source")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source dataset fingerprints differ"):
        ppc.assemble_stages(tmp_path / "clean", tmp_path / "phone", tmp_path / "train", {"x": 1})


def test_half_rejects_legacy_manifest_and_changed_preparation_code(tmp_path, monkeypatch):
    import json
    _stage(tmp_path / "clean", [1, 2])
    path = tmp_path / "clean" / ppc.MANIFEST
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "format": "prep_half_v1"}))
    with pytest.raises(ValueError, match="regenerate legacy halves"):
        ppc.load_stage_manifest(tmp_path / "clean")
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(ppc, "_preparation_hash", lambda: "changed-code")
    with pytest.raises(ValueError, match="preparation code changed"):
        ppc.load_stage_manifest(tmp_path / "clean")


@pytest.mark.parametrize("mismatch", [None, "labels", "is_informal"])
def test_recover_variant_dependent_source_hashes(tmp_path, mismatch):
    _stage(tmp_path / "clean", [1, 2], source="clean-hash")
    _stage(tmp_path / "phone", [1, 2], source="phone-hash")
    if mismatch:
        phone = ppc.load_stage_dataset(tmp_path / "phone")
        values = [[1], [3]] if mismatch == "labels" else [False, True]
        phone = phone.remove_columns(mismatch).add_column(mismatch, values)
        phone.save_to_disk(str(tmp_path / "replacement"))
        import shutil
        shutil.rmtree(tmp_path / "phone")
        (tmp_path / "replacement").rename(tmp_path / "phone")
        loaded = Dataset.from_file(str(next((tmp_path / "phone").glob("*.arrow"))))
        ppc.write_stage_manifest(
            tmp_path / "phone", loaded, variant="phone", fingerprint={"x": 1},
            source_fingerprint="phone-hash",
        )
    kwargs = dict(allow_source_fingerprint_mismatch=True)
    if mismatch:
        with pytest.raises(ValueError, match=mismatch):
            ppc.assemble_stages(tmp_path / "clean", tmp_path / "phone",
                                tmp_path / "train", {"x": 1}, **kwargs)
        assert (tmp_path / "clean" / ppc.MANIFEST).exists()
        assert (tmp_path / "phone" / ppc.MANIFEST).exists()
        assert not (tmp_path / "train").exists()
    else:
        meta = ppc.assemble_stages(tmp_path / "clean", tmp_path / "phone",
                                   tmp_path / "train", {"x": 1}, **kwargs)
        assert meta["total_rows"] == 4


def test_half_loads_source_with_identical_pipeline_state(tmp_path, monkeypatch):
    from datasets.fingerprint import Hasher
    from src.data import PersianASRDataPipeline

    source_hashes = []
    preparation_states = []

    class Raw:
        _fingerprint = "source"

        def __len__(self):
            return 2

    def load_raw(self, **kwargs):
        # The informal-data filter captures self through its normalizer closure.
        source_hashes.append(Hasher.hash(self))
        return Raw()

    def prepare(self, raw, **kwargs):
        preparation_states.append((self.telephony_prob, self.strict_telephony))
        return object()

    monkeypatch.setattr("transformers.WhisperProcessor.from_pretrained", lambda _: None)
    monkeypatch.setattr("scripts.train_lora._label_format", lambda _: (None, None))
    monkeypatch.setattr(PersianASRDataPipeline, "load_raw_dataset", load_raw)
    monkeypatch.setattr(PersianASRDataPipeline, "prepare_dataset_split", prepare)
    monkeypatch.setattr(ppc, "build_fingerprint", lambda _: {"x": 1})
    monkeypatch.setattr(ppc, "write_stage_manifest", lambda *a, **kw: {"rows": 2, "shards": []})
    for variant in ("clean", "phone"):
        ns = _all_ns(tmp_path)
        ns.stage = str(tmp_path / variant)
        ppc.prepare_half(ns, variant)
    assert source_hashes[0] == source_hashes[1]
    assert preparation_states == [(0.0, False), (1.0, True)]


def test_upload_final_defaults_to_configured_repo(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = {}
    monkeypatch.setattr(
        ppc,
        "upload_stage",
        lambda repo, stage, path, **kw: calls.update(repo=repo, stage=stage, path=path),
    )
    ns = SimpleNamespace(
        hf_train_repo=None,
        hf_repo="me/prep",
        offload_clean_to_hf=None,
        no_upload_final=False,
    )
    ppc._upload_final(ns, tmp_path / "train")
    assert calls["repo"] == "me/prep"
    assert calls["stage"] == tmp_path / "train"
    assert calls["path"] == "train"


def test_upload_final_respects_opt_out(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        ppc,
        "upload_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("uploaded")),
    )
    ns = SimpleNamespace(
        hf_train_repo="me/prep",
        hf_repo=None,
        offload_clean_to_hf=None,
        no_upload_final=True,
    )
    ppc._upload_final(ns, tmp_path / "train")


def _all_ns(tmp_path, **overrides):
    from types import SimpleNamespace

    ns = SimpleNamespace(
        prep_cache_dir=str(tmp_path),
        base_model="m", dataset_name="d", num_proc=1,
        telephony_profiles=["customer"], telephony_seed=42,
        telephony_use_ffmpeg=True, max_train_samples=None, force=True,
        clean_strategy="auto", max_pre_phone_gb=60.0, clean_up=True,
        no_upload_final=True, keep_hf_clean=False,
        hf_train_repo=None, hf_repo=None, offload_clean_to_hf=None,
        public_repo=False,
    )
    ns.__dict__.update(overrides)
    return ns


def test_ensure_repo_creates_public_when_requested(monkeypatch):
    from huggingface_hub.utils import RepositoryNotFoundError

    calls = {"repo_info": 0}

    class _Api:
        def repo_info(self, *a, **k):
            calls["repo_info"] += 1
            if calls["repo_info"] == 1:
                raise RepositoryNotFoundError("not found")
            return type("Info", (), {"private": False})()

        def create_repo(self, repo_id, **kw):
            calls.update(kw, repo_id=repo_id)

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    assert ppc._ensure_repo("me/prep", public=True) is False
    assert calls["private"] is False
    assert calls["repo_type"] == "dataset"
    assert calls["repo_info"] == 2


def test_ensure_repo_propagates_non_404_errors(monkeypatch):
    calls = {"created": False}

    class _Api:
        def repo_info(self, *a, **k):
            raise OSError("network down")

        def create_repo(self, *a, **k):
            calls["created"] = True

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    with pytest.raises(OSError, match="network"):
        ppc._ensure_repo("me/prep", public=True)
    assert calls["created"] is False


def test_ensure_repo_rereads_visibility_after_create(monkeypatch):
    from huggingface_hub.utils import RepositoryNotFoundError

    calls = {"repo_info": 0}

    class _Api:
        def repo_info(self, *a, **k):
            calls["repo_info"] += 1
            if calls["repo_info"] == 1:
                raise RepositoryNotFoundError("not found")
            return type("Info", (), {"private": True})()

        def create_repo(self, *a, **k):
            return None

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    assert ppc._ensure_repo("me/prep", public=True) is True


def test_ensure_repo_never_flips_existing_private_repo(monkeypatch):
    calls = {"created": False}

    class _Api:
        def repo_info(self, *a, **k):
            return type("Info", (), {"private": True})()

        def create_repo(self, *a, **k):
            calls["created"] = True

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    assert ppc._ensure_repo("me/prep", public=True) is True
    assert calls["created"] is False


def test_private_quota_guard_rejects_oversized_final_cache():
    with pytest.raises(RuntimeError, match="private"):
        ppc._assert_private_fits("me/prep", True, 168 * 10**9)
    ppc._assert_private_fits("me/prep", False, 168 * 10**9)


def test_guard_projected_train_skips_when_no_final_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ppc, "_ensure_repo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("checked")),
    )
    ppc._guard_projected_train(_all_ns(tmp_path, no_upload_final=True), tmp_path)


def test_private_limit_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("ASR_PREP_HF_PRIVATE_LIMIT_GB", "garbage")
    assert ppc._private_limit_bytes() == 100.0 * 1e9

    monkeypatch.setenv("ASR_PREP_HF_PRIVATE_LIMIT_GB", "2048")
    assert ppc._private_limit_bytes() == 2048.0 * 1e9


def test_evict_stage_files_advises_every_arrow(tmp_path, monkeypatch):
    (tmp_path / "a.arrow").write_bytes(b"x" * 10)
    (tmp_path / "b.arrow").write_bytes(b"y" * 20)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        ppc.os,
        "posix_fadvise",
        lambda fd, off, length, advice: seen.append(advice),
    )
    assert ppc._evict_stage_files(tmp_path) == 30
    assert seen == [ppc.os.POSIX_FADV_DONTNEED, ppc.os.POSIX_FADV_DONTNEED]
    assert ppc._evict_stage_files(tmp_path / "empty") == 0


def test_auto_keeps_clean_when_fadvise_frees_memory(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(
        ppc, "_drop_caches",
        lambda: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    monkeypatch.setattr(
        ppc, "_evict_stage_files",
        lambda stage: events.append("fadvise") or 84_000_000_000,
    )
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 20.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: None)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)

    ppc.run_all(_all_ns(tmp_path))

    assert "fadvise" in events
    assert "offload" not in events


def test_auto_offloads_when_memory_stays_high_after_fadvise(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(
        ppc, "_drop_caches",
        lambda: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    monkeypatch.setattr(ppc, "_evict_stage_files", lambda stage: 1)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 150.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: None)

    ppc.run_all(_all_ns(tmp_path))

    assert "offload" in events


def test_run_all_deletes_remote_clean_after_assembling(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_drop_caches", lambda: events.append("drop"))
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: events.append("download"))
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append(f"delete:{path}"))

    ppc.run_all(_all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep"))

    assert events[-1] == "delete:clean"


def test_run_all_keeps_remote_clean_with_flag(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append("delete"))

    ppc.run_all(
        _all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep",
                keep_hf_clean=True)
    )

    assert "delete" not in events


def test_delete_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        ppc, "delete_stage",
        lambda repo, path: (_ for _ in ()).throw(RuntimeError("403")),
    )
    assert ppc.run_all(_all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep")) == 0


def test_repo_public_prefers_flag_then_env(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("ASR_PREP_HF_PUBLIC", raising=False)
    assert ppc._repo_public(SimpleNamespace(public_repo=True)) is True
    assert ppc._repo_public(SimpleNamespace(public_repo=False)) is False
    monkeypatch.setenv("ASR_PREP_HF_PUBLIC", "1")
    assert ppc._repo_public(SimpleNamespace(public_repo=False)) is True


def test_guard_projected_train_counts_offloaded_clean(monkeypatch):
    monkeypatch.setattr(ppc, "_ensure_repo", lambda repo, public: True)
    monkeypatch.setattr(ppc, "_repo_storage_bytes", lambda repo: 0)
    monkeypatch.setattr(ppc.prep_cache, "dir_size", lambda path: 40 * 10**9)

    ns = _all_ns("/tmp", no_upload_final=False, hf_repo="me/prep",
                 clean_strategy="auto")
    with pytest.raises(RuntimeError, match="private"):
        ppc._guard_projected_train(ns, "/tmp/clean")


def test_guard_projected_train_keeps_two_x_when_clean_stays_local(monkeypatch):
    monkeypatch.setattr(ppc, "_ensure_repo", lambda repo, public: True)
    monkeypatch.setattr(ppc, "_repo_storage_bytes", lambda repo: 0)
    monkeypatch.setattr(ppc.prep_cache, "dir_size", lambda path: 40 * 10**9)

    ns = _all_ns("/tmp", no_upload_final=False, hf_repo="me/prep",
                 clean_strategy="keep")
    ppc._guard_projected_train(ns, "/tmp/clean")


def test_guard_projected_train_counts_existing_repo_bytes(monkeypatch):
    monkeypatch.setattr(ppc, "_ensure_repo", lambda repo, public: True)
    monkeypatch.setattr(ppc, "_repo_storage_bytes", lambda repo: 50 * 10**9)
    monkeypatch.setattr(ppc.prep_cache, "dir_size", lambda path: 40 * 10**9)

    ns = _all_ns("/tmp", no_upload_final=False, hf_repo="me/prep",
                 clean_strategy="keep")
    with pytest.raises(RuntimeError, match="private"):
        ppc._guard_projected_train(ns, "/tmp/clean")


def test_hf_off_sentinel_is_not_a_repo(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(ppc, "_auto_hf_repo", lambda: None)
    ns = SimpleNamespace(
        hf_train_repo=None, hf_repo="off", offload_clean_to_hf=None
    )
    assert ppc._final_hf_repo(ns) is None
    assert ppc._clean_hf_repo(ns) is None


def test_upload_stage_enforces_private_quota(monkeypatch):
    uploaded = {"called": False}

    class _Api:
        def upload_folder(self, *a, **k):
            uploaded["called"] = True

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    monkeypatch.setattr(ppc, "_ensure_repo", lambda repo, public: True)
    monkeypatch.setattr(ppc.prep_cache, "dir_size", lambda path: 200 * 10**9)

    with pytest.raises(RuntimeError, match="private"):
        ppc.upload_stage("me/prep", Path("/tmp/train"), "train")
    assert uploaded["called"] is False


def test_delete_stage_passes_dataset_folder(monkeypatch):
    calls = {}

    class _Api:
        def delete_folder(self, repo_id, **kw):
            calls.update(kw, repo_id=repo_id)

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    ppc.delete_stage("me/prep", "clean")

    assert calls["repo_id"] == "me/prep"
    assert calls["path_in_repo"] == "clean"
    assert calls["repo_type"] == "dataset"


def test_run_all_invokes_projection_guard(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(
        ppc, "_guard_projected_train",
        lambda ns, stage: called.append(stage),
    )
    monkeypatch.setattr(ppc, "_drop_caches", lambda: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})

    ppc.run_all(_all_ns(tmp_path, clean_strategy="keep"))

    assert called == [tmp_path / "clean"]


def test_run_all_uploads_before_deleting_remote_clean(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(ppc, "_upload_final", lambda ns, train: events.append("upload"))
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append("delete"))

    ppc.run_all(
        _all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep",
                no_upload_final=False)
    )

    assert events == ["upload", "delete"]


def test_failed_final_upload_keeps_remote_clean(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        ppc, "_upload_final",
        lambda ns, train: (_ for _ in ()).throw(RuntimeError("quota")),
    )
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append("delete"))

    with pytest.raises(RuntimeError, match="quota"):
        ppc.run_all(
            _all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep",
                    no_upload_final=False)
        )

    assert "delete" not in events
