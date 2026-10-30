#!/usr/bin/env bash
# Creates .venv (Python 3.13) and installs pinned packages. Does nothing if already set up.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.13}
command -v "$PY" >/dev/null || PY=python3
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
REQ=requirements.txt
[ "$(uname)" = "Darwin" ] && REQ=requirements-mac.txt
if ! .venv/bin/python -c "import torch, xgboost, chronos, statsmodels, arch, optuna, sklearn, pandas, matplotlib, tabulate, ruptures" 2>/dev/null; then
  .venv/bin/pip install -r "$REQ"
fi
echo "environment ok: $(.venv/bin/python --version)"
