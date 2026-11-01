import pytest
from unittest.mock import patch, MagicMock
from src.runpod_deploy.ssh_bootstrap import (
    resolve_ssh_connection,
    build_ssh_command,
    SSHConnectionInfo,
    build_bootstrap_script,
    SSHExecutor,
)


def test_resolve_ssh_connection_direct():
    pod_data = {
        "id": "pod_xyz123",
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 14522, "ip": "194.26.1.10"}
            ]
        },
    }
    conn = resolve_ssh_connection(pod_data)
    assert conn.host == "194.26.1.10"
    assert conn.port == 14522
    assert conn.user == "root"
    assert conn.is_proxy is False


def test_resolve_ssh_connection_proxy():
    pod_data = {
        "id": "pod_xyz123",
        "runtime": {
            "ports": []  # No direct port 22 mapped
        },
        "ssh": {
            "proxy": {
                "host": "ssh.runpod.io",
                "port": 22,
                "user": "pod_xyz123-root",
            }
        },
    }
    conn = resolve_ssh_connection(pod_data)
    assert conn.host == "ssh.runpod.io"
    assert conn.port == 22
    assert conn.user == "pod_xyz123-root"
    assert conn.is_proxy is True


def test_build_ssh_command():
    conn = SSHConnectionInfo(host="194.26.1.10", port=14522, user="root", is_proxy=False)
    cmd = build_ssh_command(conn, "uptime", identity_file="/home/user/.ssh/id_ed25519")
    assert "ssh" in cmd[0]
    assert "-p" in cmd
    assert "14522" in cmd
    assert "root@194.26.1.10" in cmd
    assert "-i" in cmd
    assert "/home/user/.ssh/id_ed25519" in cmd
    assert "uptime" in cmd[-1]


def test_build_bootstrap_script_contains_ttyd_and_token():
    script = build_bootstrap_script(
        gh_token="secret_token",
        terminal_user="cockpit",
        terminal_pass="secret_pass_123",
        repo_url="https://github.com/ameerhossein/liquidwhisper-fa.git",
        branch="main",
    )
    assert "x-access-token" in script
    assert "secret_token" in script
    assert "ttyd" in script
    assert "8080" in script
    assert "-c cockpit:secret_pass_123" in script
    assert "setup_server.sh" in script
    assert "runpod_web.sh" in script


def test_wait_ssh_ready():
    conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
    with patch.object(SSHExecutor, "run_command", side_effect=[(1, "Connection refused"), (0, "")]):
        assert SSHExecutor.wait_ssh_ready(conn, timeout_sec=5, poll_interval=0.05) is True

    with patch.object(SSHExecutor, "run_command", return_value=(1, "Connection refused")):
        assert SSHExecutor.wait_ssh_ready(conn, timeout_sec=0.1, poll_interval=0.05) is False


def test_wait_web_ready():
    conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
    with patch.object(SSHExecutor, "run_command", return_value=(0, "HTTP/1.1 200 OK")):
        assert SSHExecutor.wait_web_ready(conn, timeout_sec=5, poll_interval=0.1) is True

    with patch.object(SSHExecutor, "run_command", return_value=(1, "Connection refused")):
        assert SSHExecutor.wait_web_ready(conn, timeout_sec=0.2, poll_interval=0.1) is False
