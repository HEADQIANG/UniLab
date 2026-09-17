#!/usr/bin/env bash
# Task-local Drake bootstrap that preserves the other simulation extras.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${UNILAB_DRAKE_SETUP_HOME:-$(dirname "$PROJECT_ROOT")/unilab-runtimes/drake}"
UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

usage() {
  printf '%s\n' \
    'Usage: bash scripts/diagnostics/setup_drake_mixed.sh [--run command ...]' \
    'Without arguments: download local dependencies, build, and install DrakeUni.' \
    'With --run: launch a command with the existing local Drake library paths.' \
    'Requires Ubuntu 24.04 x86_64, C++20, and the project Python 3.12 environment.' \
    'UNILAB_DRAKE_SETUP_HOME overrides the runtime directory; UV_BIN selects uv.'
}

DRAKE_PREFIX="$RUNTIME_ROOT/drake-1.56.0-noble"
DEPS_ROOT="$RUNTIME_ROOT/deps"
SOURCE_ROOT="$RUNTIME_ROOT/unilabsim-drake_uni-4cdc9ba"
export LD_LIBRARY_PATH="$DEPS_ROOT/usr/lib/x86_64-linux-gnu:$DRAKE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DRAKE_HOME="$DRAKE_PREFIX"
export UNILAB_DRAKE_HOME="$DRAKE_PREFIX"
export UNILAB_DRAKE_UNI_SOURCE="$SOURCE_ROOT"

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --run)
    shift
    [ "$#" -gt 0 ] || { usage >&2; exit 2; }
    [ -f "$DRAKE_PREFIX/lib/libdrake.so" ] || {
      printf '%s\n' 'Run setup_drake_mixed.sh without --run first.' >&2
      exit 1
    }
    exec "$@"
    ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac

[ "$(uname -s)/$(uname -m)" = Linux/x86_64 ] || {
  printf '%s\n' 'This bootstrap is for Ubuntu noble x86_64 only.' >&2
  exit 1
}
# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}/${VERSION_CODENAME:-}" = ubuntu/noble ] || {
  printf '%s\n' 'Use the platform-specific Drake setup for other distributions.' >&2
  exit 1
}
for tool in "$UV_BIN" c++ curl sha256sum tar apt-get dpkg-deb pkg-config; do
  command -v "$tool" >/dev/null || { printf 'Missing tool: %s\n' "$tool" >&2; exit 1; }
done
[ -x "$PROJECT_ROOT/.venv/bin/python" ] || {
  printf '%s\n' 'Create the project environment before adding Drake.' >&2
  exit 1
}
"$UV_BIN" --directory "$PROJECT_ROOT" run --no-sync python -c \
  'import sys; assert sys.version_info[:2] == (3, 12), "This bootstrap requires Python 3.12"'

DOWNLOADS="$RUNTIME_ROOT/downloads"
mkdir -p "$DOWNLOADS" "$DEPS_ROOT"

fetch_checked() {
  local url="$1" target="$2" digest="$3"
  if [ ! -f "$target" ]; then
    curl -fLsS -C - --retry 2 --connect-timeout 15 -o "$target.part" "$url"
    mv "$target.part" "$target"
  fi
  printf '%s  %s\n' "$digest" "$target" | sha256sum --check --status
}

fetch_checked \
  https://drake-packages.csail.mit.edu/drake/release/drake-1.56.0-noble.tar.gz \
  "$DOWNLOADS/drake-1.56.0-noble.tar.gz" \
  23b7a255f338893b6131ecfbd06b2eb9b7d7e4420f5d3fc056f9467e3aea1cea
if [ ! -f "$DRAKE_PREFIX/lib/libdrake.so" ]; then
  mkdir -p "$DRAKE_PREFIX"
  tar -xzf "$DOWNLOADS/drake-1.56.0-noble.tar.gz" --strip-components=1 -C "$DRAKE_PREFIX"
fi
fetch_checked \
  https://api.github.com/repos/unilabsim/drake_uni/tarball/4cdc9ba4c9b1a7542755631afe0d57dbb54cdb63 \
  "$DOWNLOADS/drake_uni-4cdc9ba.tar.gz" \
  aaa027f7363bf3f18caa99e1aab4e536a06f7160247bbbce70365e8b1502dc69
if [ ! -f "$SOURCE_ROOT/src/drake_uni/compiled/drake_env_pool.cc" ]; then
  tar -xzf "$DOWNLOADS/drake_uni-4cdc9ba.tar.gz" -C "$RUNTIME_ROOT"
fi
# This upstream 0.1.0 commit includes the named actuator/body metadata needed
# by current unisim-core; the earlier PyPI 0.1.0 artifact omits those fields.
BUILD_HELPER="$SOURCE_ROOT/scripts/build_drake_batch.py"
printf '%s  %s\n' \
  58d4b2b86b8d88a54f948a5add984e97f9927cd854c0adc8b90199d81a53575c \
  "$BUILD_HELPER" | sha256sum --check --status

# Download and unpack only: no sudo, apt install, or system package changes.
(
  cd "$DOWNLOADS"
  apt-get download libeigen3-dev libfmt-dev libfmt9 libspdlog-dev libspdlog1.12 libpython3.12-dev
  for package in ./*.deb; do
    dpkg-deb -x "$package" "$DEPS_ROOT"
  done
)
export EIGEN3_INCLUDE_DIR="$DEPS_ROOT/usr/include/eigen3"
export FMT_INCLUDE_DIR="$DEPS_ROOT/usr/include"
export FMT_LIB_DIR="$DEPS_ROOT/usr/lib/x86_64-linux-gnu"
export CPLUS_INCLUDE_PATH="$DEPS_ROOT/usr/include/python3.12:$DEPS_ROOT/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PKG_CONFIG_SYSROOT_DIR="$DEPS_ROOT"
export PKG_CONFIG_PATH="$DEPS_ROOT/usr/lib/x86_64-linux-gnu/pkgconfig:$DEPS_ROOT/usr/share/pkgconfig"

"$UV_BIN" --directory "$PROJECT_ROOT" run --no-sync python "$BUILD_HELPER" --drake-home "$DRAKE_PREFIX"
"$UV_BIN" --no-config pip install --python "$PROJECT_ROOT/.venv/bin/python" \
  --no-deps --no-build-isolation --index-url https://pypi.org/simple -e "$SOURCE_ROOT"
"$UV_BIN" --directory "$PROJECT_ROOT" run --no-sync python -c \
  'from drake_uni.runtime import batch_diagnostics; d = batch_diagnostics(); print(d); assert d.batch_available, d.batch_import_error'
printf '%s\n' 'Drake ready. Use setup_drake_mixed.sh --run to launch a mixed experiment.'
