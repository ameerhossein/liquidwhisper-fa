"""Contract tests for the zero-dependency local demo baselines."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_demo_module():
    path = ROOT / "scripts" / "demo_local.py"
    spec = importlib.util.spec_from_file_location("demo_local_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_baselines_support_phone_deltas_and_reference_tag():
    demo = _load_demo_module()
    whisper, buzzasr = demo.sim_benchmarks[:2]

    assert whisper["is_baseline"] is True
    assert whisper["is_reference"] is True
    assert buzzasr["is_baseline"] is True
    assert buzzasr["is_reference"] is False

    for row in (whisper, buzzasr):
        assert row["total_samples"] == 2149
        assert isinstance(row["phone_normalized_wer"], (int, float))

    assert len(demo.DEMO_EPOCH_PHONE_WERS) == 4
    assert all(
        isinstance(value, (int, float)) for value in demo.DEMO_EPOCH_PHONE_WERS
    )
