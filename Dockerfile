FROM ubuntu:24.04 AS artifact-base

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_TOOLCHAIN=nightly-2026-08-07

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    gnupg \
    git \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    autoconf \
    automake \
    libtool \
    m4 \
    pkg-config \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    libntl-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libzstd-dev \
    zlib1g-dev \
    valgrind \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Add the official LLVM 21 repository for Ubuntu 24.04 (Noble).
RUN wget -qO /usr/share/keyrings/apt.llvm.org.asc \
        https://apt.llvm.org/llvm-snapshot.gpg.key \
    && echo \
        "deb [signed-by=/usr/share/keyrings/apt.llvm.org.asc] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
        > /etc/apt/sources.list.d/llvm-21.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-21 \
        lld-21 \
        llvm-21 \
        llvm-21-dev \
        libmlir-21-dev \
        mlir-21-tools \
        libpolly-21-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/lib/llvm-21/bin:/root/.cargo/bin:/opt/venv/bin:${PATH}"
ENV MLIR_SYS_210_PREFIX="/usr/lib/llvm-21"
ENV TABLEGEN_210_PREFIX="/usr/lib/llvm-21"
ENV LIBCLANG_PATH="/usr/lib/llvm-21/lib"
ENV BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/llvm-21/lib/clang/21/include"
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV SYMBOLICA_HIDE_BANNER="1"

# Install the pinned Rust toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- \
        -y \
        --profile minimal \
        --default-toolchain "${RUST_TOOLCHAIN}" \
        --component rustfmt

# Install Python dependencies in a virtual environment.
RUN python3 -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt

WORKDIR /artifact
COPY .cargo/ .cargo/
COPY Cargo.toml Cargo.lock rust-toolchain ./
COPY analyzer/ analyzer/
COPY cachegrind-runner/ cachegrind-runner/
COPY denning/ denning/
COPY raffine/ raffine/

FROM artifact-base AS artifact

# Barvinok's pinned nested submodules still declare repo.or.cz URLs using the
# legacy git:// transport. Fetch the same repositories and revisions over HTTPS
# for better firewall compatibility and more reliable evaluator builds.
RUN git config --global \
    url."https://repo.or.cz/".insteadOf "git://repo.or.cz/"

# Compile both the evaluator-facing binaries and test harnesses during image
# construction so the first smoke/reproduction run does not rebuild them.
RUN cargo test --release --locked --workspace --no-run \
    && cargo build --release --locked --workspace

ARG ARTIFACT_REVISION=unknown
ENV ARTIFACT_REVISION="${ARTIFACT_REVISION}"

COPY benchmarks/ benchmarks/
COPY scripts/ scripts/
COPY salt_vs_hw_misses_package/ salt_vs_hw_misses_package/
COPY artifact/ artifact/
COPY README.md LICENSE requirements.txt .envrc.example ./
COPY --chmod=0755 artifact/entrypoint.sh /usr/local/bin/salt-artifact

ENTRYPOINT ["salt-artifact"]
CMD ["help"]
