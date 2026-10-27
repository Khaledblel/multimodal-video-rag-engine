#!/usr/bin/env bash
# Usage: bash deploy/digitalocean/sync.sh root@DROPLET_IP
# Uploads the repo to /opt/vrag on the droplet and (re)starts the stack.
set -euo pipefail

TARGET="${1:?usage: sync.sh user@host}"
cd "$(dirname "$0")/../.."

rsync -az --delete \
  --exclude /.git --exclude /data --exclude /.env \
  --exclude node_modules --exclude .next --exclude __pycache__ \
  ./ "$TARGET:/opt/vrag/"
ssh "$TARGET" "bash /opt/vrag/deploy/digitalocean/bootstrap.sh"
