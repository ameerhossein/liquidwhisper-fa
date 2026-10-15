#!/usr/bin/env bash
set -e

# Load .env if present (repo root next to this script, else CWD).
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# RunPod default proxy port is 7860 or 8000
PORT=${PORT:-7860}
HOST=${HOST:-0.0.0.0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/runpod_web.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/runpod_web.pid}"

FOLLOW_LOGS=1
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -d|--detach|--no-follow) FOLLOW_LOGS=0 ;;
        -f|--foreground|--no-background) FOREGROUND=1 ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "=========================================================="
echo " Launching Persian ASR RunPod Cockpit"
echo " Web UI & Live Telemetry Server"
echo " Host: ${HOST} | Port: ${PORT}"
echo "=========================================================="

# Foreground mode (legacy behavior, dies with SSH).
if [ "${FOREGROUND:-0}" = "1" ]; then
    python3 -m uvicorn server.app:app --host "${HOST}" --port "${PORT}" "${EXTRA_ARGS[@]}"
    exit 0
fi

mkdir -p "${LOG_DIR}"

# Stop previous instance if PID file points at a live server.
if [ -f "${PID_FILE}" ]; then
    OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Stopping previous instance (PID ${OLD_PID})..."
        kill "${OLD_PID}" 2>/dev/null || true
        sleep 2
    fi
    rm -f "${PID_FILE}"
fi

echo "Starting in background (survives SSH disconnect)..."
echo "Logs: ${LOG_FILE}"
# setsid detaches the server into its own session/process group so that
# Ctrl+C (SIGINT to the log tail) and SSH hangup (SIGHUP) never reach it.
# nohup is kept as a belt-and-braces guard on systems without setsid.
if command -v setsid >/dev/null 2>&1; then
    setsid nohup python3 -m uvicorn server.app:app --host "${HOST}" --port "${PORT}" "${EXTRA_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
    nohup python3 -m uvicorn server.app:app --host "${HOST}" --port "${PORT}" "${EXTRA_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
echo "$!" > "${PID_FILE}"
disown 2>/dev/null || true

PID="$(cat "${PID_FILE}")"
echo "PID: ${PID} (saved to ${PID_FILE})"
echo "Re-attach logs: tail -F ${LOG_FILE}"
echo "Stop server:    ./kill_web.sh (or: kill \$(cat ${PID_FILE}) && rm ${PID_FILE})"
echo "----------------------------------------------------------"
echo "Streaming logs (Ctrl+C detaches, server keeps running):"
echo "----------------------------------------------------------"

if [ "${FOLLOW_LOGS}" = "1" ]; then
    # Give uvicorn a moment so tail has something to show immediately.
    sleep 1
    tail -F "${LOG_FILE}"
fi
