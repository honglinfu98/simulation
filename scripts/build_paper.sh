#!/usr/bin/env bash
# Compile the paper from anywhere (handles cd + PATH + clean rebuild).
#   ./scripts/build_paper.sh          # -> paper/build/main.pdf
#   ./scripts/build_paper.sh -C       # clean first, then build
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
cd "$ROOT/paper"
[ "${1:-}" = "-C" ] && latexmk -C >/dev/null 2>&1
latexmk -pdf -interaction=nonstopmode main.tex
echo "PDF: $ROOT/paper/build/main.pdf"
