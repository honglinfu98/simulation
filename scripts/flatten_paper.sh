#!/usr/bin/env bash
# Produce the AAAI single-file submission source.
#
# AAAI forbids \input in the submitted source ("every section must be in the
# single source file") and wants the bibliography inlined as .bbl. We WRITE the
# paper modularly (paper/main.tex + \input{section/...}); this script flattens it
# into paper/main_flat.tex for submission. Only the PDF is needed at the review
# stage, so run this only when AAAI asks for source (camera-ready).
#
#   ./scripts/flatten_paper.sh          # -> paper/main_flat.tex (single file)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/paper"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"

# 1) build once so build/main.bbl exists (bibliography to inline)
latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

# 2) expand every \input recursively AND inline the .bbl -> one self-contained file
latexpand --expand-bbl build/main.bbl main.tex > main_flat.tex

echo "wrote paper/main_flat.tex ($(wc -l < main_flat.tex) lines)"
echo "remaining \\input lines (should be 0): $(grep -c '\\\\input' main_flat.tex || true)"
echo "Submit: main_flat.tex + aaai2026.sty + aaai2026.bst + figures (figs/, img/, exhibits/, generated/, diagram/)."
