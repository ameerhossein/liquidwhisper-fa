"""SSH connection and remote bootstrapping for RunPod pods."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SSHConnectionInfo:
    host: str
    port: int
    user: str = "root"
    is_proxy: bool = False


def resolve_ssh_connection(pod_data: Dict[str, Any]) -> SSHConnectionInfo:
    """Resolve SSH connection parameters from pod metadata.

    Supports direct TCP mapped ports and RunPod SSH proxy gateways.
    """
    runtime = pod_data.get("runtime") or {}
    ports = runtime.get("ports") or []

    # 1. Direct TCP port mapped for container port 22
    for p in ports:
        if p.get("privatePort") == 22 and p.get("publicPort") and p.get("ip"):
            return SSHConnectionInfo(
                host=str(p["ip"]),
                port=int(p["publicPort"]),
                user="root",
                is_proxy=False,
            )

    # 2. Proxy Gateway info
    ssh_block = pod_data.get("ssh") or {}
    proxy_info = ssh_block.get("proxy")
    if proxy_info and proxy_info.get("host") and proxy_info.get("user"):
        return SSHConnectionInfo(
            host=str(proxy_info["host"]),
            port=int(proxy_info.get("port", 22)),
            user=str(proxy_info["user"]),
            is_proxy=True,
        )

    # 3. Direct SSH block info
    direct_info = ssh_block.get("direct")
    if direct_info and direct_info.get("ip") and direct_info.get("port"):
        return SSHConnectionInfo(
            host=str(direct_info["ip"]),
            port=int(direct_info["port"]),
            user=str(direct_info.get("user", "root")),
            is_proxy=False,
        )

    pod_id = pod_data.get("id", "unknown")
    raise RuntimeError(
        f"Could not resolve reachable SSH endpoint for pod {pod_id}.\n"
        "Pod may still be initializing its runtime network or lacks SSH mapping."
    )


def build_ssh_command(
    conn: SSHConnectionInfo,
    remote_cmd: str,
    identity_file: Optional[str] = None,
) -> List[str]:
    """Assemble an ssh command list for subprocess execution."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(conn.port),
    ]
    if identity_file:
        cmd.extend(["-i", identity_file])

    target = f"{conn.user}@{conn.host}"
    cmd.extend([target, remote_cmd])
    return cmd


def build_bootstrap_script(
    gh_token: str,
    terminal_user: str = "cockpit",
    terminal_pass: str = "",
    repo_url: str = "https://github.com/ameerhossein/liquidwhisper-fa.git",
    branch: str = "main",
) -> str:
    """Generate the remote bootstrap script."""
    # Convert repo_url to authenticated clone URL
    auth_repo_url = repo_url
    if "github.com" in repo_url:
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        if "@" in clean_url:
            clean_url = clean_url.split("@", 1)[1]
        auth_repo_url = f"https://x-access-token:{gh_token}@{clean_url}"

    ttyd_auth_flag = f"-c {terminal_user}:{terminal_pass}" if terminal_pass else ""

    script = f"""#!/usr/bin/env bash
set -e

echo "=== [1/4] Ensuring Web Terminal (ttyd on port 8080) ==="
if ! command -v ttyd >/dev/null 2>&1; then
    echo "Downloading static ttyd binary..."
    curl -sLo /usr/local/bin/ttyd https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 || true
    chmod +x /usr/local/bin/ttyd || true
fi

if command -v ttyd >/dev/null 2>&1 && ! pgrep -x "ttyd" >/dev/null 2>&1; then
    echo "Starting ttyd on port 8080..."
    nohup ttyd -p 8080 {ttyd_auth_flag} -W bash > /workspace/ttyd.log 2>&1 &
    sleep 1
fi

echo "=== [2/4] Cloning or updating repository ==="
mkdir -p /workspace
cd /workspace
if [ -d "/workspace/stt/.git" ]; then
    echo "Existing git repo found in /workspace/stt, updating..."
    cd /workspace/stt
    git fetch {auth_repo_url} {branch} || true
    git checkout {branch} || true
    git pull {auth_repo_url} {branch} || true
else
    echo "Cloning {repo_url} into /workspace/stt..."
    git clone --branch {branch} {auth_repo_url} /workspace/stt
    cd /workspace/stt
    # Sanitize remote origin so personal access token is not retained on persistent volume
    git remote set-url origin {repo_url} 2>/dev/null || true
fi

echo "=== [3/4] Running setup_server.sh ==="
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi
chmod +x setup_server.sh runpod_web.sh || true
./setup_server.sh

echo "=== [4/4] Starting Persian ASR Web Cockpit ==="
./runpod_web.sh -d

echo "=== Remote Bootstrap Complete ==="
"""
    return script


class SSHExecutor:
    """Executes remote commands over SSH."""

    @staticmethod
    def run_command(
        conn: SSHConnectionInfo,
        cmd: str,
        identity_file: Optional[str] = None,
        timeout: int = 60,
    ) -> Tuple[int, str]:
        ssh_cmd = build_ssh_command(conn, cmd, identity_file=identity_file)
        try:
            res = subprocess.run(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            return res.returncode, res.stdout
        except subprocess.TimeoutExpired:
            return -1, "SSH command timed out"
        except Exception as e:
            return -1, f"SSH execution failed: {e}"

    @classmethod
    def run_streaming(
        cls,
        conn: SSHConnectionInfo,
        cmd: str,
        identity_file: Optional[str] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> int:
        ssh_cmd = build_ssh_command(conn, cmd, identity_file=identity_file)
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        if proc.stdout:
            for line in proc.stdout:
                line_str = line.rstrip()
                if on_line:
                    on_line(line_str)
                else:
                    print(line_str, flush=True)

        proc.wait()
        return proc.returncode

    @classmethod
    def wait_ssh_ready(
        cls,
        conn: SSHConnectionInfo,
        timeout_sec: int = 120,
        poll_interval: float = 3.0,
        identity_file: Optional[str] = None,
    ) -> bool:
        """Poll SSH port until SSH accepts a command."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rc, _ = cls.run_command(conn, "true", identity_file=identity_file, timeout=8)
            if rc == 0:
                return True
            time.sleep(poll_interval)
        return False

    @classmethod
    def bootstrap_pod(
        cls,
        conn: SSHConnectionInfo,
        gh_token: str,
        terminal_user: str = "cockpit",
        terminal_pass: str = "",
        repo_url: str = "https://github.com/ameerhossein/liquidwhisper-fa.git",
        branch: str = "main",
        identity_file: Optional[str] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> bool:
        script = build_bootstrap_script(
            gh_token=gh_token,
            terminal_user=terminal_user,
            terminal_pass=terminal_pass,
            repo_url=repo_url,
            branch=branch,
        )
        # Pass script via base64 or inline bash -c
        import base64
        encoded_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
        remote_cmd = f"echo {encoded_script} | base64 -d | bash"
        rc = cls.run_streaming(conn, remote_cmd, identity_file=identity_file, on_line=on_line)
        return rc == 0

    @classmethod
    def wait_web_ready(
        cls,
        conn: SSHConnectionInfo,
        timeout_sec: int = 120,
        poll_interval: float = 3.0,
        identity_file: Optional[str] = None,
    ) -> bool:
        """Poll the Web Cockpit at http://127.0.0.1:7860/ over SSH until HTTP 200."""
        deadline = time.time() + timeout_sec
        check_cmd = "curl -s -i http://127.0.0.1:7860/ | head -n 1"
        while time.time() < deadline:
            rc, out = cls.run_command(conn, check_cmd, identity_file=identity_file, timeout=10)
            if rc == 0 and ("200 OK" in out or "307 Temporary Redirect" in out):
                return True
            time.sleep(poll_interval)
        return False
