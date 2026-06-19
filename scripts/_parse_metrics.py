#!/usr/bin/env python3
"""Parse a finished run's pulled-back artifacts into a one-line summary.

Usage: _parse_metrics.py <outdir> <tag> <outcome>
Prints e.g.  rho=0.860 genacc=0.290 ppl=20.42 Fano=6.0
Tolerates missing files (a crashed run may have none).
"""
import glob
import json
import os
import re
import sys


def main() -> int:
    outdir, tag, outcome = sys.argv[1], sys.argv[2], sys.argv[3]
    parts = []

    mlog = os.path.join(outdir, "master.log")
    if os.path.exists(mlog):
        txt = open(mlog, errors="replace").read()
        m = re.findall(r"closed_form_rho=([0-9.]+)", txt)
        if m:
            parts.append(f"rho={float(m[-1]):.3f}")

    gj = os.path.join(outdir, f"genuine_{tag}.json")
    if os.path.exists(gj):
        try:
            d = json.load(open(gj))
            if "genuine_mark_accuracy" in d:
                parts.append(f"genacc={float(d['genuine_mark_accuracy']):.3f}")
            if "genuine_mark_perplexity" in d:
                parts.append(f"ppl={float(d['genuine_mark_perplexity']):.2f}")
        except Exception:
            pass

    for sf in glob.glob(os.path.join(outdir, "stylized_facts", "*.json")):
        try:
            d = json.load(open(sf))
            fano = d.get("headline", {}).get("F5 Fano at scales", {}).get("model")
            if fano:
                parts.append(f"Fano={float(fano[0]):.1f}")
                break
        except Exception:
            pass

    print(" ".join(parts) if parts else f"outcome={outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
