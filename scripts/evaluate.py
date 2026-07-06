#!/usr/bin/env python3
"""Evaluate a trained checkpoint (evaluation stage).

    python scripts/evaluate.py genuine  --checkpoint <ckpt> --data-dir <events> ...   # acc + perplexity
    python scripts/evaluate.py facts     --checkpoint <ckpt> --data-dir <events> ...   # stylized facts (free rollout)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_TASKS = {
    "genuine": "volume_set_mtpp.evaluation.genuine_eval",
    "facts": "volume_set_mtpp.evaluation.stylized_facts",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _TASKS:
        sys.exit(f"usage: evaluate.py {{{'|'.join(_TASKS)}}} [args...]")
    task = sys.argv.pop(1)  # strip the subcommand so the target sees its own args
    import importlib

    importlib.import_module(_TASKS[task]).main()


if __name__ == "__main__":
    main()
