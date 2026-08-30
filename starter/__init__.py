"""Local compatibility shim. Not part of the submission bundle.

`evaluator/local_evaluator.py` imports `starter.agent.Agent` and the evaluator
is read-only, so this package keeps that entry point working locally. The agent
itself lives in `submission/`.
"""
