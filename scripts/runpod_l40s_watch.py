#!/usr/bin/env python3
"""RunPod NVIDIA L40S Watcher & Persian ASR Cockpit Auto-Launcher.

Polls RunPod for L40S GPU capacity across datacenters. When available:
1. Provisions pod with 300 GB container disk + 200 GB persistent volume.
2. Exposes ports 7860 (Web Cockpit), 8080 (Web Terminal), 8888 (JupyterLab), 22 (SSH).
3. Connects over SSH (direct TCP or proxy gateway).
4. Launches ttyd web terminal daemon on 8080 with session credentials.
5. Clones private repo via GitHub token and executes setup_server.sh.
6. Launches ./runpod_web.sh -d and verifies HTTP 200 on port 7860.
7. Dispatches desktop notification, audio chime, and prints ready banner.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path for direct CLI script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.runpod_deploy.client as rp_client
import src.runpod_deploy.notify as rp_notify
import src.runpod_deploy.ssh_bootstrap as rp_ssh


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch RunPod for NVIDIA L40S and auto-deploy Persian ASR Web Cockpit."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=25.0,
        help="Polling interval in seconds while checking stock (default: 25.0)",
    )
    parser.add_argument(
        "--cloud-type",
        type=str,
        default="SECURE",
        choices=["SECURE", "COMMUNITY", "ALL"],
        help="RunPod cloud type to target (default: SECURE)",
    )
    parser.add_argument(
        "--container-disk",
        type=int,
        default=300,
        help="Ephemeral container root disk in GB (default: 300)",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=200,
        help="Persistent volume size in GB mounted at /workspace (default: 200)",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/ameerhossein/liquidwhisper-fa.git",
        help="Git repository URL to clone on the pod",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch to clone (default: main)",
    )
    parser.add_argument(
        "--pod-name",
        type=str,
        default="persian-asr-cockpit",
        help="Name of the pod to create (default: persian-asr-cockpit)",
    )
    parser.add_argument(
        "--ssh-key-path",
        type=str,
        default=None,
        help="Path to local SSH public key file (default: auto-detect from ~/.ssh/)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check L40S availability once and exit (0 if available, 1 if none)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check credentials and availability, display planned deployment without creating pod",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("==========================================================")
    print(" RunPod NVIDIA L40S Automated Watcher & Launcher")
    print("==========================================================")

    # 1. Resolve credentials
    try:
        api_key = rp_client.resolve_runpod_key()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    client = rp_client.RunPodClient(api_key=api_key)

    if args.check_only:
        try:
            avail, best_dc, dcs = client.check_l40s_availability(cloud_type=args.cloud_type)
        except Exception as e:
            print(f"Error querying catalog: {e}", file=sys.stderr)
            return 1

        if avail:
            dc_names = ", ".join(d["id"] for d in dcs)
            print(f"L40S available in datacenter: {best_dc} (all: {dc_names})")
            return 0
        else:
            print("L40S is currently NOT available in any datacenter.")
            return 1

    # Pre-flight check for GitHub token and SSH key
    try:
        gh_token = rp_client.resolve_github_token()
        pub_key, priv_key = rp_client.resolve_ssh_key_pair(args.ssh_key_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        avail, best_dc, dcs = client.check_l40s_availability(cloud_type=args.cloud_type)
        print("[DRY-RUN] Credentials and API connection verified.")
        print(f"[DRY-RUN] Current L40S availability: {'YES (' + best_dc + ')' if avail else 'NO'}")
        print(f"[DRY-RUN] Planned Container Disk: {args.container_disk} GB")
        print(f"[DRY-RUN] Planned Persistent Volume: {args.volume_size} GB")
        print(f"[DRY-RUN] Target Repository: {args.repo_url} (branch: {args.branch})")
        if priv_key:
            print(f"[DRY-RUN] Using SSH private key: {priv_key}")
        print("[DRY-RUN] Dry run complete. No pod created.")
        return 0

    # 2. Polling loop with exponential backoff on transient errors
    spinner = rp_notify.ProgressSpinner(label="Watching for NVIDIA L40S availability")
    poll_count = 0
    error_streak = 0
    target_dc = ""
    print(f"Starting watcher loop (polling every {args.interval}s)... Press Ctrl+C to stop.")

    try:
        while True:
            poll_count += 1
            try:
                avail, best_dc, dcs = client.check_l40s_availability(cloud_type=args.cloud_type)
                error_streak = 0  # reset on successful query
                if avail:
                    spinner.clear()
                    target_dc = best_dc
                    dc_list = ", ".join(d["id"] for d in dcs)
                    print(f"\n🎉 [Poll #{poll_count}] L40S GPU IS AVAILABLE! (Selected DC: {target_dc} | Available: {dc_list})")
                    break
            except Exception as e:
                err_str = str(e)
                # Fatal: unauthorized or forbidden key
                if "401" in err_str or "403" in err_str or "Unauthorized" in err_str or "Forbidden" in err_str:
                    spinner.clear()
                    print(f"\nError: Authentication failed (401/403): {e}", file=sys.stderr)
                    print("Remediation: Verify RUNPOD_API_KEY at https://console.runpod.io/user/settings", file=sys.stderr)
                    return 1

                error_streak += 1
                backoff_wait = min(120.0, float(args.interval) * (1.5 ** min(error_streak, 6)))
                spinner.clear()
                print(f"\n[Warning] API poll #{poll_count} error (streak {error_streak}): {e}. Backing off {backoff_wait:.1f}s...", flush=True)
                time.sleep(backoff_wait)
                continue

            # Wait countdown with spinner
            countdown = int(args.interval)
            step = 0.5
            start_wait = time.time()
            frame_idx = 0
            while (time.time() - start_wait) < args.interval:
                rem = max(0, int(args.interval - (time.time() - start_wait)))
                spinner.tick(frame_idx, rem)
                frame_idx += 1
                time.sleep(step)
    except KeyboardInterrupt:
        spinner.clear()
        print("\nWatcher stopped by user.")
        return 130

    # 3. Provision Pod with safe cleanup handling
    print(f"\n[Provisioning] Creating pod '{args.pod_name}' in {target_dc}...")
    print(f"  GPU            : NVIDIA L40S (count 1)")
    print(f"  Container Disk : {args.container_disk} GB")
    print(f"  Persistent Vol : {args.volume_size} GB at /workspace")
    print(f"  Exposed Ports  : 7860/http (Cockpit), 8080/http (Terminal), 8888/http, 22/tcp")

    pod_id = None
    try:
        pod = client.create_pod(
            name=args.pod_name,
            gpu_id="NVIDIA L40S",
            data_center_id=target_dc,
            cloud_type=args.cloud_type,
            container_disk_gb=args.container_disk,
            volume_size_gb=args.volume_size,
            public_key=pub_key,
        )
        pod_id = pod.get("id")
        if not pod_id:
            print(f"Error: Pod creation response missing ID: {pod}", file=sys.stderr)
            return 1
        print(f"Pod allocated successfully! Pod ID: {pod_id}")
    except Exception as e:
        print(f"Error creating pod: {e}", file=sys.stderr)
        return 1

    # Protected block: ensures user is alerted if script is interrupted while pod is alive
    try:
        # 4. Wait for pod running & resolve SSH
        print("\nWaiting for container initialization and network ports...")
        deadline = time.time() + 600  # 10 min timeout
        conn = None

        while time.time() < deadline:
            try:
                pod_data = client.get_pod(pod_id)
                desired = pod_data.get("desiredStatus", "").upper()
                runtime = (pod_data.get("runtimeStatus") or pod_data.get("status") or "").lower()

                # Print periodic progress dots / status info
                sys.stdout.write(f"\rWaiting for container initialization... status={desired}/{runtime or 'initializing'} ")
                sys.stdout.flush()

                if (desired == "RUNNING" or not desired) and runtime == "running":
                    try:
                        conn = rp_ssh.resolve_ssh_connection(pod_data)
                        print(f"\nContainer is running! SSH endpoint: {conn.user}@{conn.host}:{conn.port} (proxy={conn.is_proxy})")
                        break
                    except Exception:
                        pass
            except Exception as e:
                print(f"\nNote: waiting for pod status ({e})...")

            time.sleep(5)

        if not conn:
            print(f"Error: Timed out waiting for pod {pod_id} network endpoint.", file=sys.stderr)
            print(f"Inspect or terminate manually: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
            return 1

        # 5. Wait for SSH daemon to respond
        print("Pinging SSH daemon until port 22 responds...")
        ssh_ready = rp_ssh.SSHExecutor.wait_ssh_ready(conn=conn, timeout_sec=120, identity_file=priv_key)
        if not ssh_ready:
            print(f"Error: Timed out waiting for SSH daemon to become reachable on pod {pod_id}.", file=sys.stderr)
            print(f"Console URL: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
            return 1
        print("SSH connection confirmed!")

        # 6. Remote bootstrap
        print("\n=== Commencing Remote Bootstrap over SSH ===")
        print("This will setup ttyd (web terminal), clone repo, install dependencies, and launch Persian ASR Web Cockpit.")
        
        term_user = "cockpit"
        term_pass = secrets.token_hex(6)
        
        success = rp_ssh.SSHExecutor.bootstrap_pod(
            conn=conn,
            gh_token=gh_token,
            terminal_user=term_user,
            terminal_pass=term_pass,
            repo_url=args.repo_url,
            branch=args.branch,
            identity_file=priv_key,
        )
        if not success:
            print(f"\n❌ Error: Remote bootstrap script encountered an error.", file=sys.stderr)
            print(f"Pod ID: {pod_id} (STILL RUNNING AND BILLING)", file=sys.stderr)
            print(f"Console: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
            print(f"SSH Inspect: ssh {conn.user}@{conn.host} -p {conn.port}" + (f" -i {priv_key}" if priv_key else ""), file=sys.stderr)
            return 1

        # 7. Verify Web Cockpit ready
        print("\nVerifying Web Cockpit health (http://127.0.0.1:7860/)...")
        ready = rp_ssh.SSHExecutor.wait_web_ready(conn=conn, timeout_sec=90, identity_file=priv_key)
        if not ready:
            print(f"\n❌ Error: Web Cockpit did not return HTTP 200 within timeout.", file=sys.stderr)
            print(f"Pod ID: {pod_id} (STILL RUNNING AND BILLING)", file=sys.stderr)
            print(f"Console: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
            print(f"Inspect server logs on pod: /workspace/stt/logs/runpod_web.log", file=sys.stderr)
            return 1

        # 8. Formulate URLs and notify
        web_cockpit_url = f"https://{pod_id}-7860.proxy.runpod.net"
        web_terminal_url = f"https://{pod_id}-8080.proxy.runpod.net"
        ssh_cmd = f"ssh {conn.user}@{conn.host} -p {conn.port}"
        if priv_key:
            ssh_cmd += f" -i {priv_key}"

        banner = rp_notify.format_ready_banner(
            pod_id=pod_id,
            web_cockpit_url=web_cockpit_url,
            web_terminal_url=web_terminal_url,
            ssh_cmd=ssh_cmd,
            terminal_user=term_user,
            terminal_pass=term_pass,
        )
        print("\n" + banner + "\n")

        # Send notifications
        rp_notify.send_desktop_notification("RunPod L40S Web Cockpit Live", f"Direct URL: {web_cockpit_url}")
        rp_notify.play_audio_chime()
        return 0

    except KeyboardInterrupt:
        print(f"\n\n⚠️  ABORTED BY USER (Ctrl+C).", file=sys.stderr)
        print(f"⚠️  WARNING: Pod {pod_id} was provisioned and is STILL RUNNING on your account!", file=sys.stderr)
        print(f"⚠️  Manage or terminate it at: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error during pod setup: {e}", file=sys.stderr)
        if pod_id:
            print(f"⚠️  Pod {pod_id} is STILL RUNNING on your account!", file=sys.stderr)
            print(f"⚠️  Manage or terminate it at: https://console.runpod.io/pod/{pod_id}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
