#!/usr/bin/env bash
# Usage: bash deploy/vm/sync.sh USER@VM_IP
# Uploads the repo to /opt/vrag on the VM and (re)starts the stack.
set -euo pipefail

TARGET="${1:?usage: sync.sh user@host}"
cd "$(dirname "$0")/../.."

ssh "$TARGET" 'mkdir -p /opt/vrag 2>/dev/null || sudo install -d -o "$(id -u)" -g "$(id -g)" /opt/vrag'
rsync -az --delete \
  --exclude /.git --exclude /data --exclude /.env --exclude /misc \
  --exclude node_modules --exclude .next --exclude __pycache__ \
  ./ "$TARGET:/opt/vrag/"
ssh "$TARGET" "bash /opt/vrag/deploy/vm/bootstrap.sh"
