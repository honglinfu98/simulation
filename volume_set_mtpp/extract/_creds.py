"""Shared credential check for the extraction stage.

Extraction needs provider credentials that are never committed to the repo
(see extract/README.md). This helper fails fast with an actionable message
when they're absent.
"""

import os
import sys


def require_credentials() -> str:
    """Return the name of the credential found, or exit with guidance.

    Looks for GOOGLE_APPLICATION_CREDENTIALS (GCS service account) or
    KAIKO_API_KEY (Kaiko REST). Extraction runs on the UCL HPC cluster against
    /SAN/medic/TFOW/...; the real downloader implementation is dropped into the
    download_* modules there.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return "GOOGLE_APPLICATION_CREDENTIALS"
    if os.environ.get("KAIKO_API_KEY"):
        return "KAIKO_API_KEY"
    sys.exit(
        "extract: no data-provider credentials found.\n"
        "Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json "
        "(GCS) or KAIKO_API_KEY=<key> (Kaiko), then re-run on a host with the "
        "extractor available. See volume_set_mtpp/extract/README.md."
    )
