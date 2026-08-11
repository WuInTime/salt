# AutoLALA

AUTOmatic Loop Asympotic Locality Analysis

## How to Compile

(assume ubuntu)
```bash
# install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

# install build tools
sudo apt install build-essential cmake autoconf libtool

# set environment variables
export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21
export TABLEGEN_210_PREFIX=/usr/lib/llvm-21

# build and test
cargo build --release
cargo test --release
```



## Personal Setup Note

sudo dnf install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
cd /home/ywu/code/out_source/autolala-25.10.30
direnv allow

For my machine, keep these variables local to the repo by using `.envrc` and the workspace VS Code settings below. On Fedora, install `direnv` first, enable the shell hook, then run `direnv allow` once in the project root so it loads automatically whenever you enter this workspace.

```bash
export PATH="/home/ywu/.local/llvm-21-mlir/bin:$PATH"; export MLIR_SYS_210_PREFIX=/home/ywu/.local/llvm-21-mlir; export TABLEGEN_210_PREFIX=/home/ywu/.local/llvm-21-mlir; export SYMBOLICA_HIDE_BANNER=1
```

Build the analyzer:

```bash
cargo build --release --bin analyzer
```

Run SALT analysis:

```bash
./target/release/analyzer -i /path/to/input.mlir --json -o /path/to/output.json salt --block-size 8
```

Notes:

- `--block-size` is optional for `salt`.
- Omit `--block-size` to keep block size symbolic.
- You can run without prebuilding:

```bash
cargo run -r -b analyzer -- -i /path/to/input.mlir --json -o /path/to/output.json salt --block-size 8
```

cargo run -r -b analyzer -- -i benchmarks/examples/sym_heat_center_only.mlir -m test.svg --json salt --block-size=8

target/release/analyzer -i benchmarks/examples/sym_heat_center_only.mlir -m test.svg --json salt --block-size=8


## Recommended development setup (for VSCode)

- Install DirEnv
  - [Installation](https://direnv.net/docs/installation.html)
  - [Setup](https://direnv.net/docs/hook.html)

- Use the repo-root `.envrc` to load the local LLVM toolchain automatically
  ```bash
  # .envrc
  export PATH="/home/ywu/.local/llvm-21-mlir/bin:$PATH"
  export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21
  export TABLEGEN_210_PREFIX=/usr/lib/llvm-21
  export SYMBOLICA_HIDE_BANNER=1
  ```

- Install `rust-analyzer` extension for VSCode
  - [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer)

  The repo already includes matching settings in `.vscode/settings.json` so rust-analyzer picks up the same environment automatically.
  ```json
  <!-- {
    "rust-analyzer.cargo.extraEnv": {
        "MLIR_SYS_210_PREFIX" : "/usr/lib/llvm-21",
        "TABLEGEN_210_PREFIX" : "/usr/lib/llvm-21",
    },
    "rust-analyzer.check.extraEnv": {
        "MLIR_SYS_210_PREFIX" : "/usr/lib/llvm-21",
        "TABLEGEN_210_PREFIX" : "/usr/lib/llvm-21",
    },
    "rust-analyzer.server.extraEnv": {
        "MLIR_SYS_210_PREFIX" : "/usr/lib/llvm-21",
        "TABLEGEN_210_PREFIX" : "/usr/lib/llvm-21",
    },
    "rust-analyzer.runnables.extraEnv": {
        "MLIR_SYS_210_PREFIX" : "/usr/lib/llvm-21",
        "TABLEGEN_210_PREFIX" : "/usr/lib/llvm-21",
    }, -->
    "editor.formatOnSave": true,
    "files.insertFinalNewline": true,
  }
  ```
