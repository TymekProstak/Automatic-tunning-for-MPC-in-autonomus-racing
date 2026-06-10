#!/usr/bin/env bash
set -euo pipefail

# katalog, gdzie leży ten skrypt = mpc_solver/
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$THIS_DIR/.." && pwd)"   # repo_root/

SYSROOT_DIR="$ROOT_DIR/External/acados_sysroot"
VENV_ACTIVATE="$ROOT_DIR/External/acados_venv/bin/activate"

if [[ ! -d "$SYSROOT_DIR" ]]; then
  echo "[ERR] Brak $SYSROOT_DIR. Najpierw odpal instalację acadosa."
  exit 1
fi
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[ERR] Brak venv: $VENV_ACTIVATE. Najpierw odpal instalację acadosa."
  exit 1
fi

source "$VENV_ACTIVATE"

export ACADOS_SOURCE_DIR="$SYSROOT_DIR"
export PATH="$ACADOS_SOURCE_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"

unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH

# uruchamiamy generator z katalogu mpc_solver (żeby relative paths były przewidywalne)
cd "$THIS_DIR"
python3 solver_generator.py
