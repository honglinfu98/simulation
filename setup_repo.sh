#!/usr/bin/env bash
# One-shot environment setup: create a local venv, install dependencies, and make
# the flat `volume_set_mtpp` package importable (no pyproject — a .pth on the repo
# root does the job). Mirrors the copytrading-style workflow.
#
#   chmod +x setup_repo.sh
#   ./setup_repo.sh
#   . venv/bin/activate
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "==> creating venv at $ROOT/venv"
"$PYTHON" -m venv "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --upgrade pip >/dev/null
echo "==> installing requirements"
"$ROOT/venv/bin/python" -m pip install -r "$ROOT/requirements.txt"

# Put the repo root on the venv's import path so `import volume_set_mtpp` works
# from anywhere once the venv is active (the package lives at the repo root).
SITE="$("$ROOT/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT" > "$SITE/volume_set_mtpp_repo.pth"

echo "==> done. Activate with:  . venv/bin/activate"
echo "    then:  cp .env.example .env  (fill in HPC_* and SMTP_*)"
