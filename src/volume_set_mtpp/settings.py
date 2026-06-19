"""Project-wide paths.

`PROJECT_ROOT` is the repository root (three levels up from this file:
src/volume_set_mtpp/settings.py -> volume_set_mtpp -> src -> <repo root>).
Used by the data-processing driver to resolve default input/output locations.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
