"""RunPod REST v2 client and credential resolution helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_env_file(path: Path) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if not path.is_file():
        return vals
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass
    return vals


def _read_toml_key(path: Path, section: str, key: str) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        cur_sec = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    cur_sec = line[1:-1].strip()
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if (not section or cur_sec == section) and k == key:
                        return v
    except Exception:
        pass
    return None


def resolve_runpod_key(config_file: Optional[Path] = None) -> str:
    """Resolve RunPod API key from env, .env, or ~/.runpod/config.toml."""
    # 1. Environment variable
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key

    # 2. Local .env file
    env_file = Path(".env")
    if env_file.is_file():
        env_dict = _read_env_file(env_file)
        if env_dict.get("RUNPOD_API_KEY"):
            return env_dict["RUNPOD_API_KEY"]

    # 3. ~/.runpod/config.toml
    config_path = config_file or (Path.home() / ".runpod" / "config.toml")
    toml_key = _read_toml_key(config_path, "", "api_key")
    if toml_key:
        return toml_key

    raise RuntimeError(
        "RUNPOD_API_KEY is not set.\n"
        "Remediation: Set it in your environment: export RUNPOD_API_KEY=your_key\n"
        "Or add it to .env: RUNPOD_API_KEY=your_key\n"
        "Get your key at: https://console.runpod.io/user/settings"
    )


def resolve_github_token() -> str:
    """Resolve GitHub token from env, .env, or via `gh auth token`."""
    token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if token:
        return token

    env_file = Path(".env")
    if env_file.is_file():
        env_dict = _read_env_file(env_file)
        if env_dict.get("GITHUB_TOKEN"):
            return env_dict["GITHUB_TOKEN"]
        if env_dict.get("GH_TOKEN"):
            return env_dict["GH_TOKEN"]

    # Try gh auth token CLI
    if shutil.which("gh"):
        try:
            res = subprocess.run(
                ["gh", "auth", "token"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            if res.returncode == 0 and out:
                return out
        except Exception:
            pass

    raise RuntimeError(
        "GitHub token not found.\n"
        "Remediation: Run `gh auth login` with the GitHub CLI, or set GITHUB_TOKEN in your environment or .env."
    )


def resolve_ssh_key_pair(key_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Read SSH public key and locate its corresponding private key file."""
    candidates = []
    if key_path:
        candidates.append(Path(key_path))
    else:
        ssh_dir = Path.home() / ".ssh"
        candidates.extend([
            ssh_dir / "id_ed25519.pub",
            ssh_dir / "id_rsa.pub",
            ssh_dir / "id_ecdsa.pub",
        ])

    for p in candidates:
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    # Look for corresponding private key
                    priv_candidate = None
                    if p.name.endswith(".pub"):
                        priv_file = p.parent / p.name[:-4]
                        if priv_file.is_file():
                            priv_candidate = str(priv_file)
                    return content, priv_candidate
            except Exception:
                pass

    raise RuntimeError(
        f"No SSH public key found. Checked: {[str(c) for c in candidates]}\n"
        "Remediation: Generate one with `ssh-keygen -t ed25519` or pass --ssh-key-path."
    )


def resolve_ssh_public_key(key_path: Optional[str] = None) -> str:
    """Read SSH public key from given path or auto-detect from ~/.ssh/."""
    pub, _ = resolve_ssh_key_pair(key_path)
    return pub


class RunPodClient:
    """RunPod REST v2 API Client."""

    def __init__(self, api_key: str, base_url: str = "https://api.runpod.io/v2") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        body = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if "Cloudflare" in err_body or "cf-error-details" in err_body:
                raise RuntimeError(
                    f"RunPod API blocked by Cloudflare (HTTP {e.code} {e.reason}).\n"
                    "Your IP address or network route is being blocked by Cloudflare bot protection.\n"
                    "Remediation: Configure an HTTP/HTTPS proxy in your terminal (e.g., export https_proxy=http://127.0.0.1:port) "
                    "or VPN, or create the pod manually in the RunPod web console."
                ) from e
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("error") or err_json.get("message") or err_body
            except Exception:
                msg = err_body
            raise RuntimeError(f"RunPod API error ({e.code} {e.reason}): {msg}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"RunPod network error: {e.reason}") from e

    def check_l40s_availability(
        self,
        cloud_type: str = "SECURE",
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Check availability of NVIDIA L40S.

        Returns (is_available, best_datacenter_id, available_datacenters).
        """
        resp = self._request(
            "GET",
            "catalog/gpus",
            params={"include": "AVAILABILITY", "product": "POD"},
        )
        if isinstance(resp, dict):
            catalog = resp.get("gpus", [])
        elif isinstance(resp, list):
            catalog = resp
        else:
            raise RuntimeError(f"Unexpected catalog response type from RunPod API: {type(resp).__name__} ({resp})")

        l40s_info = None
        for item in catalog:
            if not isinstance(item, dict):
                continue
            if item.get("id") == "NVIDIA L40S":
                l40s_info = item
                break

        if not l40s_info:
            return False, "", []

        total_avail = l40s_info.get("availability", "NONE").upper()
        if total_avail == "NONE":
            return False, "", []

        available_dcs: List[Dict[str, Any]] = []
        for dc in l40s_info.get("dataCenters", []):
            dc_avail = dc.get("availability", "NONE").upper()
            if dc_avail != "NONE":
                available_dcs.append(dc)

        best_dc = available_dcs[0]["id"] if available_dcs else ""
        return bool(available_dcs), best_dc, available_dcs

    def create_pod(
        self,
        name: str = "persian-asr-cockpit",
        gpu_id: str = "NVIDIA L40S",
        data_center_id: str = "",
        cloud_type: str = "SECURE",
        container_disk_gb: int = 300,
        volume_size_gb: int = 200,
        image_name: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        public_key: str = "",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create on-demand GPU pod with specified specs."""
        env_map = {}
        if public_key:
            env_map["PUBLIC_KEY"] = public_key
        if extra_env:
            env_map.update(extra_env)

        payload: Dict[str, Any] = {
            "name": name,
            "image": image_name,
            "cloud": cloud_type,
            "gpu": {"id": gpu_id, "count": 1},
            "disk": container_disk_gb,
            "mounts": {
                "persistent": {
                    "size": volume_size_gb,
                    "path": "/workspace",
                }
            },
            "ports": [
                "7860/http",  # Persian ASR Cockpit
                "8080/http",  # Web Terminal (ttyd)
                "8888/http",  # JupyterLab (optional)
                "22/tcp",     # SSH
            ],
            "env": env_map,
        }
        # RunPod v2 REST API does not allow 'dataCenterId' at the root level;
        # the scheduler places the pod in an available datacenter for the requested cloud & GPU.

        return self._request("POST", "pods", data=payload)

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        """Retrieve live metadata for a pod."""
        return self._request("GET", f"pods/{pod_id}")

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod by ID."""
        try:
            self._request("DELETE", f"pods/{pod_id}")
            return True
        except Exception:
            return False
