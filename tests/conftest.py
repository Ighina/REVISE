"""Point the test suite at a writable scratch root.

``revise.config`` reads ``REVISE_SCRATCH`` at import time and creates the run
directories under it, so the variable has to be set before any ``revise``
module is imported.  An explicit setting is respected -- do that when running
the ``slow`` tests, so that the model they download is cached between runs --
and otherwise a fixed directory in the system temp is used, so that the suite
runs on a machine with no scratch filesystem without leaving a new directory
behind on every run.
"""
import os
import tempfile
from pathlib import Path

if not os.environ.get("REVISE_SCRATCH"):
    os.environ["REVISE_SCRATCH"] = str(Path(tempfile.gettempdir()) / "revise-tests")
