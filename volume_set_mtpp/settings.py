"""Project-wide paths.

`PROJECT_ROOT` is the repository root (two levels up from this file:
volume_set_mtpp/settings.py -> volume_set_mtpp -> <repo root>).
Used by the data-processing driver to resolve default input/output locations.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
