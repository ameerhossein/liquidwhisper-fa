import pytest
from unittest.mock import patch, MagicMock
from src.runpod_deploy.client import (
    resolve_runpod_key,
    resolve_github_token,
    resolve_ssh_public_key,
    RunPodClient,
)


def test_resolve_runpod_key_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key_123")
    assert resolve_runpod_key() == "test_key_123"


def test_resolve_runpod_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY is not set"):
        resolve_runpod_key(config_file=tmp_path / "empty.toml")


def test_resolve_github_token_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_mocktoken")
    assert resolve_github_token() == "ghp_mocktoken"


def test_resolve_github_token_gh_cli(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="gho_from_cli\n")
        assert resolve_github_token() == "gho_from_cli"


def test_resolve_ssh_public_key(tmp_path):
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@test")
    assert resolve_ssh_public_key(key_path=str(pub_file)) == "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@test"


def test_check_l40s_availability_in_stock():
    client = RunPodClient(api_key="mock_key")
    mock_catalog = [
        {
            "id": "NVIDIA L40S",
            "availability": "HIGH",
            "dataCenters": [
                {"id": "US-NV-1", "availability": "HIGH"},
                {"id": "EU-IS-1", "availability": "NONE"},
            ],
        }
    ]
    with patch.object(client, "_request", return_value=mock_catalog):
        avail, dc_id, dcs = client.check_l40s_availability()
        assert avail is True
        assert dc_id == "US-NV-1"
        assert len(dcs) == 1


def test_check_l40s_availability_out_of_stock():
    client = RunPodClient(api_key="mock_key")
    mock_catalog = [
        {
            "id": "NVIDIA L40S",
            "availability": "NONE",
            "dataCenters": [
                {"id": "US-NV-1", "availability": "NONE"},
            ],
        }
    ]
    with patch.object(client, "_request", return_value=mock_catalog):
        avail, dc_id, dcs = client.check_l40s_availability()
        assert avail is False
        assert dc_id == ""
        assert len(dcs) == 0


def test_create_pod_payload():
    client = RunPodClient(api_key="mock_key")
    with patch.object(client, "_request") as mock_req:
        mock_req.return_value = {"id": "pod_new_123"}
        pod = client.create_pod(
            name="test-pod",
            gpu_id="NVIDIA L40S",
            data_center_id="US-NV-1",
            cloud_type="SECURE",
            container_disk_gb=300,
            volume_size_gb=200,
            public_key="ssh-ed25519 AAA...",
        )
        assert pod["id"] == "pod_new_123"
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        payload = kwargs.get("data")
        assert payload["gpu"]["id"] == "NVIDIA L40S"
        assert payload["disk"] == 300
        assert payload["mounts"]["persistent"]["size"] == 200
        assert payload["mounts"]["persistent"]["path"] == "/workspace"
        assert "7860/http" in payload["ports"]
        assert "8080/http" in payload["ports"]
        assert "22/tcp" in payload["ports"]
        assert payload["dataCenterId"] == "US-NV-1"
