#!/usr/bin/env bash
# Compile the paper, from anywhere. Always runs latexmk INSIDE paper/ (so
# aaai2026.sty and the section/ assets resolve) and puts TinyTeX on PATH.
#
#   bash scripts/build_paper.sh            # -> paper/build/main.pdf
#   bash scripts/build_paper.sh -C         # clean build artifacts first, then build
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
cd "$ROOT/paper"
if [[ "${1:-}" == "-C" ]]; then latexmk -C >/dev/null 2>&1 || true; fi
latexmk -pdf -interaction=nonstopmode main.tex
echo "PDF: $ROOT/paper/build/main.pdf"
