#!/usr/bin/env bash
# Idempotent: installs Docker if missing, then builds and starts the stack. Run as root on the droplet.
set -euo pipefail

export HOME="${HOME:-/root}"
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

cd "$(dirname "$0")/../.."

apt_get() { apt-get -o DPkg::Lock::Timeout=300 "$@"; }

if ! command -v docker >/dev/null 2>&1; then
  apt_get update
  apt_get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt_get update
  apt_get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

[ -f .env ] || cp .env.example .env
install -d -m 0775 -o 1000 -g 1000 data data/media data/artifacts

docker compose up -d --build
docker compose ps
