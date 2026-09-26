#!/usr/bin/env bash

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    exec sudo -- "$0" "$@"
fi

target_user=${1:-${SUDO_USER:-cc}}

if ! getent passwd "$target_user" >/dev/null; then
    echo "Unknown target user: $target_user" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    apt-transport-https \
    autoconf \
    automake \
    bison \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    docker.io \
    docker-buildx \
    flex \
    g++ \
    gcc \
    gfortran \
    git \
    git-lfs \
    gnupg \
    libboost-all-dev \
    libedit-dev \
    libffi-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libgmp-dev \
    libmpc-dev \
    libmpfr-dev \
    libncurses-dev \
    libntl-dev \
    libssl-dev \
    libtinfo-dev \
    libtool \
    libxml2-dev \
    libzstd-dev \
    linux-tools-common \
    linux-tools-generic \
    hwloc \
    m4 \
    make \
    msr-tools \
    ninja-build \
    numactl \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    xz-utils \
    zlib1g-dev

# Prefer matching kernel tools when Chameleon's Ubuntu repository provides
# them. The generic tools above remain available when it does not.
kernel_tools="linux-tools-$(uname -r)"
if apt-cache show "$kernel_tools" >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends "$kernel_tools"
else
    echo "Notice: $kernel_tools is unavailable; using linux-tools-generic." >&2
fi

systemctl enable --now docker
usermod -aG docker "$target_user"

# SALT's optional fresh-measurement path uses perf_event_open(2). This setting
# applies only to the leased bare-metal node and disappears when it is released.
cat >/etc/sysctl.d/99-salt-perf.conf <<'EOF'
kernel.perf_event_paranoid = -1
EOF
sysctl --system >/dev/null

echo "Node setup complete."
echo "Docker: $(docker --version)"
docker buildx version
echo "Perf: $(perf --version)"
echo "perf_event_paranoid: $(sysctl -n kernel.perf_event_paranoid)"
echo "User '$target_user' is in groups: $(id -nG "$target_user")"
echo "A new login is required before '$target_user' receives new docker-group membership."
