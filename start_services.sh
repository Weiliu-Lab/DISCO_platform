#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HTC_URL_PREFIX="${HTC_URL_PREFIX:-/htc/}"
export ML_URL_PREFIX="${ML_URL_PREFIX:-/ml/}"
export DISCOPILOT_URL_PREFIX="${DISCOPILOT_URL_PREFIX:-/DISCOpilot/}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

pids=()

start() {
  local name="$1"
  shift
  echo "[start] ${name}: $*"
  "$@" &
  pids+=("$!")
}

stop_all() {
  echo "[stop] stopping services"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

trap stop_all SIGINT SIGTERM

if command -v gunicorn >/dev/null 2>&1; then
  start "web" gunicorn --chdir backend -b 0.0.0.0:5000 platform_web.wsgi:application
  start "htc" gunicorn --chdir high-throught-calcultion -b 0.0.0.0:8051 app:server
  start "ml" gunicorn --chdir ml_prediction -b 0.0.0.0:8050 app:server
  start "discopilot" gunicorn --chdir DISCOpilot -b 0.0.0.0:8054 chat_ui:server
else
  echo "[warn] gunicorn not found, falling back to dev servers"
  start "web" python backend/manage.py runserver 0.0.0.0:5000
  start "htc" python high-throught-calcultion/app.py
  start "ml" python ml_prediction/app.py
  start "discopilot" python DISCOpilot/chat_ui.py
fi

wait
stop_all
wait
