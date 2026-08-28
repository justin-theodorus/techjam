"""Submission entry point.

Exports `Agent` as the challenge contract requires. Everything else lives in
`submission/src/`; this file exists so the class is importable from one stable
path regardless of how the package is laid out underneath.

    from submission.agent import Agent

Requires network: No. Bundled assets: none. Third-party packages: none.
"""

from __future__ import annotations

from submission.src import agent as _agent

Agent = _agent.Agent

__all__ = ["Agent"]
