#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Portable ACADOS setup (repo-local)
# - Skrypt może leżeć w mpc_solver/
# - External/ jest obok mpc_solver/  ->  ../External
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # repo_root
EXTERNAL_DIR="$ROOT_DIR/External"

ACADOS_REPO_DIR="$EXTERNAL_DIR/acados"
ACADOS_INSTALL_DIR="$ACADOS_REPO_DIR/install"

SYSROOT_DIR="$EXTERNAL_DIR/acados_sysroot"
VENV_DIR="$EXTERNAL_DIR/acados_venv"

echo "== Portable acados setup =="
echo "SCRIPT:  $SCRIPT_DIR"
echo "ROOT:    $ROOT_DIR"
echo "EXTERNAL:$EXTERNAL_DIR"
echo "ACADOS:  $ACADOS_REPO_DIR"
echo "INSTALL: $ACADOS_INSTALL_DIR"
echo "SYSROOT: $SYSROOT_DIR"
echo "VENV:    $VENV_DIR"
echo

mkdir -p "$EXTERNAL_DIR"

echo "[1/6] System deps (build tools, Eigen, JSON)..."
sudo apt update
sudo apt install -y \
  build-essential git cmake pkg-config \
  libeigen3-dev nlohmann-json3-dev \
  python3-venv python3-pip

echo "[2/6] Clone / update acados..."
if [[ ! -d "$ACADOS_REPO_DIR/.git" ]]; then
  git clone https://github.com/acados/acados.git --recursive "$ACADOS_REPO_DIR"
else
  (cd "$ACADOS_REPO_DIR" && git pull && git submodule update --init --recursive)
fi

echo "[3/6] Build + install acados -> $ACADOS_INSTALL_DIR ..."
# Stabilnie: zawsze czyść build, bo CMakeCache potrafi się gryźć po przeniesieniu ścieżek
rm -rf "$ACADOS_REPO_DIR/build"
mkdir -p "$ACADOS_REPO_DIR/build"
cd "$ACADOS_REPO_DIR/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$ACADOS_INSTALL_DIR" \
  -DACADOS_WITH_QPOASES=ON \
  -DACADOS_WITH_OSQP=ON \
  -DACADOS_WITH_DAQP=ON

make -j"$(nproc)"
make install

echo "[4/6] Create sysroot adapter (include/lib/bin)..."
mkdir -p "$SYSROOT_DIR"
rm -f "$SYSROOT_DIR/include" "$SYSROOT_DIR/lib" "$SYSROOT_DIR/bin"

ln -s "$ACADOS_INSTALL_DIR/include" "$SYSROOT_DIR/include"
ln -s "$ACADOS_INSTALL_DIR/lib"     "$SYSROOT_DIR/lib"
ln -s "$ACADOS_REPO_DIR/bin"        "$SYSROOT_DIR/bin"

# acados_template (Twoja wersja) potrzebuje link_libs.json w $ACADOS_SOURCE_DIR/lib
# a sysroot/lib jest symlinkiem -> install/lib, więc kopiujemy tam:
if [[ -f "$ACADOS_REPO_DIR/lib/link_libs.json" ]]; then
  cp -f "$ACADOS_REPO_DIR/lib/link_libs.json" "$ACADOS_INSTALL_DIR/lib/link_libs.json"
fi

echo "[5/6] Python venv + deps..."
cd "$ROOT_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install numpy casadi
python -m pip install -e "$ACADOS_REPO_DIR/interfaces/acados_template"

echo "[6/6] Quick sanity checks..."
test -f "$SYSROOT_DIR/include/acados_c/ocp_nlp_interface.h"
test -f "$SYSROOT_DIR/lib/libacados.so"
test -f "$SYSROOT_DIR/lib/link_libs.json"

mkdir -p "$ACADOS_REPO_DIR/bin"


echo
echo "OK."
echo "Use ACADOS_SOURCE_DIR=$SYSROOT_DIR"
echo "Use venv: source $VENV_DIR/bin/activate"

