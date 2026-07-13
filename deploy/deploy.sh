#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build & (re)start the stack. Use for the first deploy AND for every update.
# Run from anywhere inside the repo, as the `deploy` user:
#     deploy/deploy.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# cd to the repo root (this script lives in deploy/)
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $(pwd)." >&2
    echo "  cp .env.example .env   # then set APP_API_KEY, API_FOOTBALL_KEY, DOMAIN" >&2
    exit 1
fi

# Fetch latest code if this is a git checkout with a remote (skip on first run
# where you may have just cloned).
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "==> Pulling latest code"
    git pull --ff-only || echo "  (skipped: no fast-forward / no upstream)"
fi

echo "==> Building images and starting containers"
docker compose up -d --build

echo "==> Waiting for the API to report healthy"
ok=""
for _ in $(seq 1 45); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
done
[ -n "$ok" ] && echo "  API is healthy." || echo "  WARNING: API not healthy yet — check: docker compose logs api"

echo "==> Reclaiming space from old image layers"
docker image prune -f >/dev/null 2>&1 || true

echo
echo "==> Current status:"
docker compose ps
DOMAIN=$(grep -E '^DOMAIN=' .env | cut -d= -f2- || true)
echo
echo "Done. App: https://${DOMAIN:-<your-domain>}"
echo "The API is internal-only (not publicly exposed). To view its docs, SSH-tunnel:"
echo "  ssh -L 8000:127.0.0.1:8000 deploy@<server-ip>   then open http://localhost:8000/docs"
