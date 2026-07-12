#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time provisioning for a fresh Ubuntu 22.04/24.04 Hetzner server.
# Installs Docker + Compose, sets up a firewall, adds swap (so image builds
# don't OOM), and creates a non-root `deploy` user with Docker access.
#
# Run ONCE, as root, on the new server:
#     ssh root@<server-ip>
#     bash bootstrap.sh
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> Updating base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get upgrade -y
apt-get install -y ca-certificates curl git ufw

echo "==> Installing Docker Engine + Compose plugin (official script)"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

echo "==> Firewall: allow SSH + HTTP + HTTPS only, deny the rest"
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> Adding 2G swap (insurance against build-time OOM)"
if ! swapon --show | grep -q '/swapfile'; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Creating non-root 'deploy' user with Docker access"
if ! id deploy >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" deploy
    usermod -aG docker deploy
    # Reuse root's SSH key so you can immediately: ssh deploy@<ip>
    if [ -f /root/.ssh/authorized_keys ]; then
        install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
        install -m 600 -o deploy -g deploy /root/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
    fi
fi

echo
echo "==> Bootstrap complete."
docker --version
docker compose version
echo
echo "Next steps:"
echo "  1) log in as the deploy user:   ssh deploy@<server-ip>"
echo "  2) clone the repo, create .env, then run: deploy/deploy.sh"
