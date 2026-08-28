"""Measurement harness: scoring, tracing, and run-to-run comparison.

Read-only with respect to `evaluator/`. The harness never re-implements the
dialogue loop; it wraps the agent and lets `evaluate()` drive.
"""
