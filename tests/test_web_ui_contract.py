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
    assert "scheduleMergePoll(runId, 2000, generation)" in js
    assert "mergeLog" in js
    assert "runPicker" in (ROOT / "server/static/index.html").read_text()


def test_historical_run_content_is_scoped():
    js = (ROOT / "server/static/app.js").read_text()
    server = (ROOT / "server/app.py").read_text()
    assert "runLogUrl(selectedRunId, viewingHistorical)" in js
    assert "updateTelemetry(payload.telemetry || {}, viewingHistorical)" in js
    assert "mergePollGeneration" in js
    assert "stage.dataset.status = status;" in js
    assert "if (detail) detail.textContent = detailText;" in js
    assert '@app.get("/api/runs/{run_id}/logs")' in server


def test_storage_warning_is_outside_run_head():
    html = (ROOT / "server/static/index.html").read_text()
    run_head = html.index('id="runHead"')
    section_end = html.index("</section>", run_head)
    warning = html.index('id="storageWarning"')
    assert warning > section_end, "the warning must render before the first run"


def test_historical_view_keeps_live_controls_and_picker_failure_local():
    js = (ROOT / "server/static/app.js").read_text()
    assert "async function pollLiveControls(" in js
    assert "pollLiveControls();" in js
    assert "interrupted: \"Run interrupted\"" in js
    refresh = js.index("async function refreshRunList(")
    select = js.index("async function selectRun(")
    refresh_body = js[refresh:select]
    catch_body = refresh_body.split("catch (e)")[-1]
    assert "viewingHistorical = false" not in catch_body
    assert "loadRunState()" not in catch_body


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


def _reduced_motion_block(css):
    start = css.index("prefers-reduced-motion")
    open_brace = css.index("{", start)
    depth = 0
    for i in range(open_brace, len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace : i + 1]
    raise AssertionError("unbalanced braces in prefers-reduced-motion block")


def test_reduced_motion_covers_stepper_and_merge_transitions():
    css = (ROOT / "server/static/style.css").read_text()
    assert "prefers-reduced-motion" in css
    block = _reduced_motion_block(css)
    # The global rule must actually switch transitions off, not merely exist.
    assert re.search(r"transition\s*:\s*none", block), (
        "the reduced-motion rule must disable transitions"
    )
    # Universal coverage is what reaches the stepper dots, progress/bar fills,
    # buttons, and merge states without enumerating each selector.
    assert "*" in block, "the reduced-motion rule must apply globally"
    # The transitions being covered must really exist, or the test is vacuous.
    for selector in (".stage-dot", ".progress-fill", ".bar-fill", ".bar-ref", ".btn"):
        assert selector in css, f"missing animated selector {selector}"
        assert re.search(
            rf"{re.escape(selector)}[^{{]*\{{[^}}]*transition\s*:",
            css,
        ), f"{selector} must declare a transition for reduced-motion to cover"


def test_picker_refresh_guards_inflight_selection():
    js = (ROOT / "server/static/app.js").read_text()
    refresh = js.index("async function refreshRunList(")
    select = js.index("async function selectRun(")
    body = js[refresh:select]
    assert "selectionAtStart" in body
    assert "selectedIdAtStart" in body
    assert "shouldApplyPickerRefresh" in body
    assert "resolvePickerSelection" in body
    # A stale response must bail before touching the picker or reloading state.
    bail = body.index("shouldApplyPickerRefresh")
    assert "return;" in body[bail:]
    assert "loadRunState()" in body


def test_cockpit_exposes_paired_training_default():
    html = (ROOT / "server" / "static" / "index.html").read_text()
    js = (ROOT / "server" / "static" / "app.js").read_text()
    view = (ROOT / "server" / "static" / "run_view.js").read_text()
    assert 'id="inputTelephonyMode"' in html
    assert 'value="paired" selected' in html
    assert 'id="inputEpochs"' in html and 'value="2"' in html
    assert "telephony_mode" in js
    assert "epochsForTelephonyMode" in js
    assert "epochsForTelephonyMode" in view


def test_pure_run_helpers_are_wired():
    js = (ROOT / "server/static/app.js").read_text()
    assert "formValuesFromConfig(" in js
    assert "runViewState(" in js
    restore = js.index("function restoreRunConfig(")
    render = js.index("function renderRun(")
    assert "formValuesFromConfig" in js[restore:render]
    assert "runViewState" in js[render:]


def test_hf_recover_ui_is_present_and_wired():
    html = (ROOT / "server/static/index.html").read_text()
    js = (ROOT / "server/static/app.js").read_text()
    for element in (
        "btnRecover",
        "hfRecoverDialog",
        "inputHfRepo",
        "inputHfRunId",
        "btnHfList",
        "hfRunList",
        "btnHfRestore",
        "btnHfRestoreResume",
        "hfRestoreStatus",
        "hfRestoreLog",
        "btnHfClose",
        "hfSyncBadge",
    ):
        assert f'id="{element}"' in html
    for function in (
        "loadHfConfig",
        "openRecoverDialog",
        "closeRecoverDialog",
        "loadRemoteRuns",
        "renderRemoteRuns",
        "startRestore",
        "pollRestore",
        "renderHfBadge",
    ):
        assert f"function {function}(" in js
    for endpoint in (
        "/api/hf/config",
        "/api/hf/runs",
        "/api/hf/restore",
        "/api/hf/restore/status",
    ):
        assert endpoint in js
    assert "hfSyncLabel" in js
    assert "recoverActionState" in js


def test_hf_recover_ui_is_present_and_wired():
    html = (ROOT / "server/static/index.html").read_text()
    js = (ROOT / "server/static/app.js").read_text()
    for element in (
        "btnRecover",
        "hfRecoverDialog",
        "inputHfRepo",
        "inputHfRunId",
        "btnHfList",
        "hfRunList",
        "btnHfRestore",
        "btnHfRestoreResume",
        "hfRestoreStatus",
        "hfRestoreLog",
        "btnHfClose",
        "hfSyncBadge",
    ):
        assert f'id="{element}"' in html
    for function in (
        "loadHfConfig",
        "openRecoverDialog",
        "closeRecoverDialog",
        "loadRemoteRuns",
        "renderRemoteRuns",
        "startRestore",
        "pollRestore",
        "renderHfBadge",
    ):
        assert f"function {function}(" in js
    for endpoint in (
        "/api/hf/config",
        "/api/hf/runs",
        "/api/hf/restore",
        "/api/hf/restore/status",
    ):
        assert endpoint in js
    assert "hfSyncLabel" in js
    assert "recoverActionState" in js
