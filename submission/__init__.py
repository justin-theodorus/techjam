"""TechJam conversational shopping agent.

The official harness may import this package either as ``submission`` from the
repository root or as ``techjam.submission`` from its parent directory. Register
a small compatibility alias in both directions so source modules can use the
local package names while callers still get the same module objects.
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

if __name__ == "techjam.submission" and "submission" not in sys.modules:
    sys.modules["submission"] = sys.modules[__name__]
