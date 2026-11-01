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


def test_full_lifecycle_simulation(capsys):
    poll_results = [
        (False, "", []),
        (True, "US-NV-1", [{"id": "US-NV-1"}]),
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
             "runtime": {"ports": [{"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4"}]},
         }), \
         patch("src.runpod_deploy.ssh_bootstrap.SSHExecutor.bootstrap_pod", return_value=True), \
         patch("src.runpod_deploy.ssh_bootstrap.SSHExecutor.wait_web_ready", return_value=True), \
         patch("src.runpod_deploy.notify.send_desktop_notification", return_value=True), \
         patch("src.runpod_deploy.notify.play_audio_chime"):

        ret = main(["--interval", "0.01"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "RUNPOD L40S COCKPIT IS LIVE!" in captured.out
        assert "https://pod_live_123-7860.proxy.runpod.net" in captured.out
        assert "https://pod_live_123-8080.proxy.runpod.net" in captured.out
        assert "ssh root@1.2.3.4 -p 12345" in captured.out

