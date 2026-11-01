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
