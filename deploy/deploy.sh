#!/usr/bin/env bash
# Build locally and push to the competition server over SSH (key auth only;
# configure ~/.ssh/config with a Host entry named in STM_HOST). The server's
# share/cache database and compiled-kernel cache are never overwritten.
#   STM_HOST=stm-server STM_SUBPATH=/stm ./deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STM_HOST:?set STM_HOST (an ~/.ssh/config host)}"
SUB="${STM_SUBPATH:-}"
( cd web && STM_BASE="${SUB%/}/" npm run build )
rsync -az --delete --info=progress2 \
  --exclude 'data/frozen' --exclude 'data/build/parts' --exclude 'web/node_modules' --exclude '.venv' \
  --exclude 'data/build/numba_cache' --exclude 'data/build/cache.sqlite*' --exclude '__pycache__' \
  ./ "$STM_HOST:/srv/stm/"
ssh "$STM_HOST" 'cd /srv/stm && (test -d .venv || python3 -m venv .venv) && .venv/bin/pip install -q -r requirements.txt && sudo systemctl restart stm && sudo systemctl reload caddy && curl -fsS localhost:8787/api/health'
echo "deployed"
