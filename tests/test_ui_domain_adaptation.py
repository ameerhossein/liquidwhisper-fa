from pathlib import Path
import re


def test_index_html_has_domain_adaptation_elements():
    html_file = Path("server/static/index.html")
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")

    assert 'id="chkDomainAdaptation"' in content
    assert 'id="domainSection"' in content
    assert 'id="inputCustomDataPath"' in content
    assert 'id="inputMergedModelPath"' in content
    assert 'id="inputReplaySamples"' in content

    # Check labels and default values
    assert "Domain Adaptation (2–3h custom data)" in content
    assert "Custom audio path (on disk)" in content
    assert "Merged base model path" in content
    assert "Replay samples (Neyshekar)" in content
    assert 'value="models/buzzasr-neyshekar-standalone"' in content
    assert 'value="1000"' in content
    assert 'min="100"' in content
    assert 'max="5000"' in content

    # Check container is hidden by default
    assert re.search(r'<div[^>]*id="domainSection"[^>]*hidden', content) is not None


def test_app_js_includes_domain_adaptation_in_payload():
    js_file = Path("server/static/app.js")
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")

    # Check DOM element mapping in el
    assert "chkDomainAdaptation: document.getElementById(\"chkDomainAdaptation\")" in content
    assert "domainSection: document.getElementById(\"domainSection\")" in content
    assert "inputCustomDataPath: document.getElementById(\"inputCustomDataPath\")" in content
    assert "inputMergedModelPath: document.getElementById(\"inputMergedModelPath\")" in content
    assert "inputReplaySamples: document.getElementById(\"inputReplaySamples\")" in content
    assert "inputSaveSteps: document.getElementById(\"inputSaveSteps\")" in content

    # Check payload keys
    assert "domain_adaptation" in content
    assert "custom_data_path" in content
    assert "base_model" in content
    assert "replay_samples" in content

    # Check toggling behavior updates LR and epochs
    assert "0.00002" in content
    assert '"3"' in content
    assert "0.0001" in content
    assert '"2"' in content

    # Check toggling updates eval_steps and save_steps
    assert '"50"' in content
    assert '"1000"' in content
