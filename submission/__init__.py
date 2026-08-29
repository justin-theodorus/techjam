"""TechJam conversational shopping agent.

The official harness may import this package either as ``submission`` from the
repository root or as ``techjam.submission`` from its parent directory.  The
source modules use the latter canonical name, so register a small compatibility
alias when the former entry point is used.  Both paths then share the same
module objects rather than loading duplicate enums and dataclasses.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


if __name__ == "submission" and "techjam.submission" not in sys.modules:
    techjam = sys.modules.get("techjam")
    if techjam is None:
        techjam = types.ModuleType("techjam")
        techjam.__path__ = [str(Path(__file__).resolve().parent.parent)]
        sys.modules["techjam"] = techjam
    sys.modules["techjam.submission"] = sys.modules[__name__]
