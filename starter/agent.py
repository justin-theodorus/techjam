"""Local compatibility shim. Not part of the submission bundle.

Re-exports the submitted agent under the path `evaluator/local_evaluator.py`
imports it from. The evaluator is read-only, so the shim lives here rather than
the import being changed there.
"""

from __future__ import annotations

from submission.agent import Agent

__all__ = ["Agent"]
