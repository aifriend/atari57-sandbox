#!/usr/bin/env bash
# Launch the Atari57 research console.
#
# What this does, in order:
#   1. Verifies the venv exists (errors out with a hint if not).
#   2. Frees the port (SIGKILLs anything currently holding 127.0.0.1:8000).
#   3. Cleans stale training-job logs from previous sessions.
#   4. Spawns a background poller that opens the browser as soon as the server
#      reports healthy on /api/health.
#   5. Runs uvicorn in the foreground with --reload, so you see live logs and
#      Ctrl+C tears everything down.
#
# Override defaults via env vars:
#   PORT=8123 HOST=0.0.0.0 NO_BROWSER=1 ./start.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}"

echo "┌──────────────────────────────────────────────"
echo "│ ATARI57 // research console launcher"
echo "│ repo: $ROOT"
echo "│ url:  $URL"
echo "└──────────────────────────────────────────────"

# 1. Venv sanity.
if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv not found at $ROOT/.venv" >&2
  echo "       Run ./setup.sh first." >&2
  exit 1
fi
if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "ERROR: uvicorn not installed in .venv" >&2
  echo "       Run: VIRTUAL_ENV=.venv uv pip install -r requirements-relaxed.txt" >&2
  exit 1
fi

# 2. Free the port. lsof is on macOS, most modern Linux distros, and BSDs.
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -ti:"$PORT" 2>/dev/null || true)"
  if [[ -n "$PIDS" ]]; then
    echo "==> port $PORT in use by pid(s): $(echo "$PIDS" | tr '\n' ' ')— killing"
    # shellcheck disable=SC2086
    kill -9 $PIDS 2>/dev/null || true
    sleep 1
  else
    echo "==> port $PORT free"
  fi
elif command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1
fi

# 3. Clean stale training-job logs (each /api/training/start creates one).
COUNT=$(find runs -maxdepth 1 -name "_training_*.log" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$COUNT" != "0" ]]; then
  echo "==> cleaning $COUNT stale training-job log file(s)"
  rm -f runs/_training_*.log
fi

# 4. Background browser-opener. Polls /api/health and opens the browser when it
#    succeeds. Quits on its own if the server doesn't come up within ~12s.
if [[ -z "${NO_BROWSER:-}" ]]; then
  (
    for _ in $(seq 1 40); do
      if curl -sf "${URL}/api/health" >/dev/null 2>&1; then
        echo "==> server up — opening $URL"
        echo
        echo "    Try this once the page loads:"
        echo "      • click ▶ on the transport to watch a real agent play"
        echo "      • click ▶ TRAIN 5K to spawn a real training subprocess"
        echo "      • click ▶ COMPARE to run parallel evals on every bundled checkpoint"
        echo "      • click ▶ REPLAY to play the most recent self-play MP4"
        echo
        if [[ "$(uname -s)" == "Darwin" ]]; then
          open "$URL"
        elif command -v xdg-open >/dev/null 2>&1; then
          xdg-open "$URL"
        elif command -v wslview >/dev/null 2>&1; then
          wslview "$URL"
        else
          echo "    (no browser opener found — open $URL manually)"
        fi
        exit 0
      fi
      sleep 0.3
    done
    echo "WARN: server didn't reach /api/health in 12s — open $URL manually" >&2
  ) &
fi

# 5. Run uvicorn in the foreground. --reload watches frontend/*.py and reloads
#    on edit. info-level logs include each request line + status code.
echo "==> starting FastAPI sidecar (Ctrl+C to stop)"
echo "──────────────────────────────────────────────"

exec env PYTHONPATH="$ROOT" \
  .venv/bin/uvicorn frontend.server:app \
  --host "$HOST" --port "$PORT" \
  --reload \
  --reload-dir frontend \
  --log-level info
