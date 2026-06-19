#!/usr/bin/env python3
"""Evaluate a trained checkpoint (evaluation stage).

    python scripts/evaluate.py genuine  --checkpoint <ckpt> --data-dir <events> ...   # acc + perplexity
    python scripts/evaluate.py facts     --checkpoint <ckpt> --data-dir <events> ...   # stylized facts (free rollout)
    python scripts/evaluate.py thinning  --checkpoint <ckpt> --v2-dir <data> ...        # exact Ogata thinning
    python scripts/evaluate.py table                                                    # rebuild comparison table
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_TASKS = {
    "genuine": "volume_set_mtpp.evaluation.genuine_eval",
    "facts": "volume_set_mtpp.evaluation.stylized_facts",
    "thinning": "volume_set_mtpp.evaluation.nmh_thinning",
    "table": "volume_set_mtpp.evaluation.build_comparison_table",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _TASKS:
        sys.exit(f"usage: evaluate.py {{{'|'.join(_TASKS)}}} [args...]")
    task = sys.argv.pop(1)  # strip the subcommand so the target sees its own args
    import importlib

    importlib.import_module(_TASKS[task]).main()


if __name__ == "__main__":
    main()
