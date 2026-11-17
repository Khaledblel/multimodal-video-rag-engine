#!/usr/bin/env bash
# Idempotent: installs Docker if missing, then builds and starts the stack.
set -euo pipefail

# The steps below need root, so a non-root user gets the script re-run under sudo.
[ "$(id -u)" -eq 0 ] || exec sudo bash "$0" "$@"

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

plugin() {
  docker "$1" version >/dev/null 2>&1 && return
  apt_get update
  apt_get install -y "$2" || apt_get install -y "$3"
}
plugin compose docker-compose-plugin docker-compose-v2
plugin buildx docker-buildx-plugin docker-buildx

[ -f .env ] || cp .env.example .env
if nvidia-smi >/dev/null 2>&1 && ! grep -q '^COMPOSE_FILE=' .env; then
  echo 'COMPOSE_FILE=compose.yaml:compose.gpu.yaml' >> .env
fi
install -d -m 0775 -o 1000 -g 1000 data data/media data/artifacts data/jobs

docker compose up -d --build qdrant ollama embedder api worker
# The web UI is built after the rest, so a failure in its build cannot keep the stack from starting.
docker compose up -d --build web \
  || echo "WARNING: the web UI did not build or start. Everything else is running." >&2
docker compose ps
if grep -q '^COMPOSE_FILE=.*compose\.gpu\.yaml' .env; then
  docker compose exec -T worker python -m vrag.tools.gpu_check
fi
docker compose exec -T worker python -m vrag.tools.ollama_check --pull
