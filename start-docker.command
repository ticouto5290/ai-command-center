#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_DIR"

if ! docker info >/dev/null 2>&1; then
  open -a Docker >/dev/null 2>&1 || true
  attempt=0
  while [ "$attempt" -lt 45 ]; do
    if docker info >/dev/null 2>&1; then
      break
    fi
    attempt=$((attempt + 1))
    sleep 2
  done
  if ! docker info >/dev/null 2>&1; then
    echo "Docker Desktop did not become ready. Open it, wait until it says it is running, then open this file again."
    exit 1
  fi
fi

cat > .env.docker <<EOF
AICC_UID=$(id -u)
AICC_GID=$(id -g)
EOF

docker compose --env-file .env.docker up --build -d
open "http://127.0.0.1:8765/"
echo "AI Command Center is running in Docker at http://127.0.0.1:8765/"
echo "To stop it, run: docker compose --env-file .env.docker down"
