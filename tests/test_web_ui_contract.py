from pathlib import Path
import json
import re
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


def test_stateful_run_controls_are_present():
    html = (ROOT / "server/static/index.html").read_text()
    js = (ROOT / "server/static/app.js").read_text()
    for element in (
        "runSelect",
        "runIdentity",
        "runCreated",
        "stageStepper",
        "stepSaved",
        "storageWarning",
        "btnResume",
        "btnMerge",
        "mergeResult",
        "mergeLog",
    ):
        assert f'id="{element}"' in html
    assert "/static/run_view.js" in html
    assert html.index("/static/run_view.js") < html.index("/static/app.js")
    assert "/api/runs/latest" in js
    assert "stageStepper" in js
    assert "stageMap(" in js
    assert "Runs are not on persistent storage (" in js
    assert "Set ASR_RUNS_DIR to a /workspace path." in js
    assert "Dev simulation" in html


def test_run_actions_are_wired():
    js = (ROOT / "server/static/app.js").read_text()
    assert "/resume" in js
    assert "/merge" in js
    assert "/api/runs?limit=" in js
    assert "storageWarning" in js
    assert "Dev simulation" in js


def test_run_history_and_actions_are_wired():
    js = (ROOT / "server/static/app.js").read_text()
    for function in (
        "refreshRunList",
        "selectRun",
        "resumeRun",
        "mergeRun",
        "renderMergeStatus",
    ):
        assert f"function {function}(" in js, f"missing {function}()"
    assert "viewingHistorical" in js
    assert '"/api/runs?limit=20"' in js
    assert "runSelect" in js
    assert "adapter_exists" in js
    assert 'run.source === "fake"' in js
    assert "formRestored = false" in js
    assert "scheduleMergePoll(runId, 2000)" in js
    assert "mergeLog" in js
    assert "runPicker" in (ROOT / "server/static/index.html").read_text()


def test_stepper_markup_matches_stage_order():
    html = (ROOT / "server/static/index.html").read_text()
    stages = re.findall(r'data-stage="([^"]+)"', html)
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [
            node,
            "-e",
            "process.stdout.write(JSON.stringify("
            "require('./server/static/run_view.js').STAGE_ORDER))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert stages == json.loads(result.stdout)


def test_stepper_styles_every_stage_state():
    css = (ROOT / "server/static/style.css").read_text()
    for status in (
        "pending",
        "running",
        "completed",
        "stopped",
        "failed",
        "skipped",
        "interrupted",
        "unknown",
    ):
        assert f".stage.is-{status}" in css
    assert ".storage-warning" in css
    assert ".stage-dot" in css
    assert ".stage-label" in css
    assert ".stage-detail" in css
    assert "prefers-reduced-motion" in css


def test_run_view_behavior():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "test_run_view.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
