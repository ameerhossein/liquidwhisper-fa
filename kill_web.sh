#!/usr/bin/env bash
set -e

# Load .env if present (repo root next to this script, else CWD).
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/runpod_web.pid}"
PORT=${PORT:-7860}

FORCE=0
for arg in "$@"; do
    case "$arg" in
        -f|--force|-9|--kill) FORCE=1 ;;
        -h|--help) echo "Usage: ./kill_web.sh [-f|--force]"; exit 0 ;;
    esac
done

# Never kill ourselves or any ancestor shell: pgrep -f can match a parent
# whose command line happens to contain the search pattern (e.g. wrappers).
PROTECTED=" $$ $PPID "
_p="$PPID"
while true; do
    _p="$(ps -o ppid= -p "${_p}" 2>/dev/null | tr -d '[:space:]' || true)"
    [ -z "${_p}" ] || [ "${_p}" = "0" ] || [ "${_p}" = "1" ] && break
    case "${PROTECTED}" in *" ${_p} "*) break ;; esac
    PROTECTED="${PROTECTED}${_p} "
done

is_protected() {
    case "${PROTECTED}" in *" $1 "*) return 0 ;; esac
    # Extra guard: anything running this script.
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q "kill_web" || return 1
}

# Echo candidate PIDs minus protected ones.
unprotected() {
    local out="" pid
    for pid in "$@"; do
        is_protected "${pid}" || out="${out} ${pid}"
    done
    echo "${out}"
}

stop_pid() {
    local pid="$1" sig="$2"
    kill "-${sig}" "${pid}" 2>/dev/null || return 0
    for _ in $(seq 1 10); do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

STOPPED=0

# 1) PID file (authoritative).
if [ -f "${PID_FILE}" ]; then
    PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
        if [ "${FORCE}" = "1" ]; then
            echo "Force-killing web server (PID ${PID})..."
            kill -KILL "${PID}" 2>/dev/null || true
        else
            echo "Stopping web server (PID ${PID})..."
            stop_pid "${PID}" TERM || {
                echo "Still alive, force-killing (PID ${PID})..."
                kill -KILL "${PID}" 2>/dev/null || true
            }
        fi
        sleep 1
        kill -0 "${PID}" 2>/dev/null && echo "WARNING: PID ${PID} still alive." || echo "Stopped (PID ${PID})."
        STOPPED=1
    else
        echo "Stale PID file (no process ${PID:-?}), removing."
    fi
    rm -f "${PID_FILE}"
fi

# 2) Fallback: any leftover uvicorn server.app process (e.g. PID file lost).
if command -v pgrep >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    LEFT="$(unprotected $(pgrep -f 'uvicorn server\.app:app' || true))"
    if [ -n "${LEFT}" ]; then
        if [ "${FORCE}" = "1" ]; then
            echo "Force-killing leftover server process(es):${LEFT}"
            # shellcheck disable=SC2086
            kill -KILL ${LEFT} 2>/dev/null || true
        else
            echo "Stopping leftover server process(es):${LEFT}"
            # shellcheck disable=SC2086
            kill ${LEFT} 2>/dev/null || true
            sleep 2
            # shellcheck disable=SC2086
            REMAIN="$(unprotected $(pgrep -f 'uvicorn server\.app:app' || true))"
            if [ -n "${REMAIN}" ]; then
                echo "Force-killing:${REMAIN}"
                # shellcheck disable=SC2086
                kill -KILL ${REMAIN} 2>/dev/null || true
            fi
        fi
        STOPPED=1
    fi
fi

# 3) Fallback: anything still listening on the web port.
if command -v ss >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    PIDS="$(unprotected $(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true))"
    if [ -n "${PIDS}" ]; then
        echo "Stopping process(es) on port ${PORT}:${PIDS}"
        # shellcheck disable=SC2086
        kill ${PIDS} 2>/dev/null || true
        STOPPED=1
    fi
fi

if [ "${STOPPED}" = "0" ]; then
    echo "Web server is not running."
fi
