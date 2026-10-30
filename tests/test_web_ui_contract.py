from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_columns_use_measured_deltas():
    html = (ROOT / "server/static/index.html").read_text(encoding="utf-8")
    assert "Δ clean vs base" in html
    assert "Δ phone vs base" in html
    assert "vs Whisper v3" not in html
    assert 'id="benchmarkSummary"' in html
    assert "/static/benchmark_view.js" in html


def test_ui_has_no_hardcoded_baseline_fallbacks():
    js = (ROOT / "server/static/app.js").read_text(encoding="utf-8")
    assert "WHISPER_WER_FALLBACK" not in js
    assert "BUZZASR_WER_FALLBACK" not in js
    assert "No measured baseline yet" in js


def test_benchmark_best_is_gated_to_comparable_rows():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "test_benchmark_view.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
