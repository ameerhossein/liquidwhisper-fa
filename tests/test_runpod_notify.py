import shutil
from unittest.mock import patch, MagicMock
from src.runpod_deploy.notify import (
    send_desktop_notification,
    play_audio_chime,
    format_ready_banner,
    ProgressSpinner,
)


def test_send_desktop_notification_success():
    with patch("shutil.which", return_value="/usr/bin/notify-send"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert send_desktop_notification("Test Title", "Test Message") is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "notify-send" in cmd[0]
        assert "Test Title" in cmd
        assert "Test Message" in cmd


def test_send_desktop_notification_missing_cli():
    with patch("shutil.which", return_value=None):
        assert send_desktop_notification("Test Title", "Test Message") is False


def test_play_audio_chime(capsys):
    with patch("shutil.which", return_value=None):
        play_audio_chime()
        captured = capsys.readouterr()
        assert "\a" in captured.out


def test_format_ready_banner():
    banner = format_ready_banner(
        pod_id="pod_12345",
        web_cockpit_url="https://pod_12345-7860.proxy.runpod.net",
        web_terminal_url="https://pod_12345-8080.proxy.runpod.net",
        ssh_cmd="ssh root@1.2.3.4 -p 1234",
    )
    assert "https://pod_12345-7860.proxy.runpod.net" in banner
    assert "https://pod_12345-8080.proxy.runpod.net" in banner
    assert "ssh root@1.2.3.4 -p 1234" in banner
    assert "pod_12345" in banner


def test_progress_spinner_render():
    spinner = ProgressSpinner(label="Checking stock")
    line = spinner.render(frame_idx=2, elapsed_sec=12.5, countdown_sec=15)
    assert "Checking stock" in line
    assert "12.5s" in line
    assert "15s" in line
