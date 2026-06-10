#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

CPPAD_SRC="${EXTERNAL_DIR}/cppad"
CPPAD_BUILD="${EXTERNAL_DIR}/build/cppad"
CPPAD_INSTALL="${EXTERNAL_DIR}/install"

echo "[setup_deps] ROOT_DIR      = ${ROOT_DIR}"
echo "[setup_deps] EXTERNAL_DIR  = ${EXTERNAL_DIR}"

# -----------------------------
# APT deps
# -----------------------------
echo "[setup_deps] Installing apt packages..."
sudo apt-get update -y
sudo apt-get install -y \
  build-essential cmake pkg-config git \
  libeigen3-dev nlohmann-json3-dev \
  coinor-libipopt-dev coinor-libipopt1v5

# -----------------------------
# Fetch CppAD
# -----------------------------
mkdir -p "${EXTERNAL_DIR}"

if [[ ! -d "${CPPAD_SRC}/.git" ]]; then
  echo "[CppAD] Cloning..."
  git clone https://github.com/coin-or/CppAD.git "${CPPAD_SRC}"
else
  echo "[CppAD] Updating..."
  git -C "${CPPAD_SRC}" fetch --all --tags
fi

# wybór wersji:
# - najprościej: bierzemy master
echo "[CppAD] Checkout master"
git -C "${CPPAD_SRC}" checkout master

# (opcjonalnie) jeśli chcesz “najświeższy tag” zamiast master:
# TAG="$(git -C "${CPPAD_SRC}" tag --list | sort -V | tail -n 1)"
# echo "[CppAD] Checkout tag: ${TAG}"
# git -C "${CPPAD_SRC}" checkout "${TAG}"

# -----------------------------
# Build + install CppAD
# -----------------------------
mkdir -p "${CPPAD_BUILD}"
cmake -S "${CPPAD_SRC}" -B "${CPPAD_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="${CPPAD_INSTALL}"

cmake --build "${CPPAD_BUILD}" -j"$(nproc)"
cmake --install "${CPPAD_BUILD}"

# -----------------------------
# Sanity check
# -----------------------------
CFG="${CPPAD_INSTALL}/include/cppad/configure.hpp"
if [[ ! -f "${CFG}" ]]; then
  echo "[CppAD][ERROR] Missing: ${CFG}"
  exit 1
fi

echo "[setup_deps] OK: ${CFG}"
echo "[setup_deps] Done."