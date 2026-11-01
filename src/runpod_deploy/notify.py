"""Notification helpers and terminal UI elements."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def send_desktop_notification(title: str, message: str) -> bool:
    """Send a desktop notification using notify-send if available."""
    notify_bin = shutil.which("notify-send")
    if not notify_bin:
        return False
    try:
        res = subprocess.run(
            [notify_bin, "-u", "critical", "-i", "computer", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def play_audio_chime() -> None:
    """Trigger an audible terminal bell and play a system sound if available."""
    # Terminal bell
    sys.stdout.write("\a")
    sys.stdout.flush()

    # Desktop audio sound
    sound_files = [
        "/usr/share/sounds/freedesktop/stereo/complete.oga",
        "/usr/share/sounds/gnome/default/alerts/glass.ogg",
        "/usr/share/sounds/sound-icons/prompt.wav",
    ]
    sound_file = next((f for f in sound_files if os.path.isfile(f)), None)

    if sound_file:
        for player in ("paplay", "aplay", "play", "ffplay"):
            p_bin = shutil.which(player)
            if p_bin:
                try:
                    args = [p_bin]
                    if player == "ffplay":
                        args.extend(["-nodisp", "-autoexit", "-loglevel", "quiet"])
                    args.append(sound_file)
                    subprocess.Popen(
                        args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    break
                except Exception:
                    pass


def format_ready_banner(
    pod_id: str,
    web_cockpit_url: str,
    web_terminal_url: str,
    ssh_cmd: str,
    terminal_user: str = "",
    terminal_pass: str = "",
) -> str:
    """Format an ASCII banner announcing deployment readiness."""
    console_url = f"https://console.runpod.io/pod/{pod_id}"
    term_line = f"  Web Terminal    : {web_terminal_url}"
    if terminal_user and terminal_pass:
        term_line += f"  (login: {terminal_user} / password: {terminal_pass})"

    lines = [
        "=" * 72,
        "  🚀 RUNPOD L40S COCKPIT IS LIVE!",
        "=" * 72,
        f"  Web Cockpit URL : {web_cockpit_url}",
        term_line,
        f"  RunPod Console  : {console_url}",
        f"  SSH Access      : {ssh_cmd}",
        "  Working Dir     : /workspace/stt",
        "  Cockpit Logs    : /workspace/stt/logs/runpod_web.log",
        "=" * 72,
    ]
    return "\n".join(lines)


class ProgressSpinner:
    """Terminal progress spinner with elapsed time and next-poll countdown."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Checking stock") -> None:
        self.label = label
        self.start_time = time.time()

    def render(self, frame_idx: int, elapsed_sec: float, countdown_sec: int) -> str:
        frame = self.FRAMES[frame_idx % len(self.FRAMES)]
        return f"{frame} {self.label}... (elapsed: {elapsed_sec:.1f}s | next retry in: {countdown_sec}s)"

    def tick(self, frame_idx: int, countdown_sec: int) -> None:
        elapsed = time.time() - self.start_time
        line = self.render(frame_idx, elapsed, countdown_sec)
        sys.stdout.write(f"\r\033[K{line}")
        sys.stdout.flush()

    def clear(self) -> None:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
