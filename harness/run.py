"""One-command evaluation: score, trace, and diff against the previous run.

    python3 -m harness.run                      # score submission.agent:Agent
    python3 -m harness.run --agent pkg.mod:Agent
    python3 -m harness.run --check-baseline     # assert the frozen reference
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from harness import diff as diff_module
from harness.analysis import analyze, health_summary, latency_summary
from harness.record import RecordingAgent
from harness.report import baseline, render

DEFAULT_AGENT = "submission.agent:Agent"
BASELINE_FIELDS = (
    ("hit_rate_at_10", "hit_rate_at_10"),
    ("mrr", "mrr"),
    ("mttc", "mttc"),
    ("efficiency", "efficiency"),
    ("recommended_technical_score", "technical_score"),
)


def load_agent_class(spec: str) -> type:
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise SystemExit(f"--agent must look like 'module:Class', got {spec!r}")
    return getattr(importlib.import_module(module_name), class_name)


def require_catalog(path: Path) -> None:
    if path.exists():
        return
    raise SystemExit(
        f"catalog not found at {path}. Unpack it first:\n"
        "  gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl"
    )


def build_agent(agent_class: type, catalog_path: Path, no_fast_path: bool) -> object:
    """Constructs the agent, asking it to skip its fast path when told to.

    An agent that does not accept `fast_path` simply does not have one, so the
    flag is a no-op rather than an error.
    """
    if not no_fast_path:
        return agent_class(str(catalog_path))
    try:
        return agent_class(str(catalog_path), fast_path=False)
    except TypeError:
        raise SystemExit(f"{agent_class.__name__} has no fast path to disable")


def build_artifact(args, agent_spec: str, result: dict, sessions: list[dict], timings: dict) -> dict:
    metrics = {key: value for key, value in result.items() if key != "sessions"}
    return {
        "schema": 1,
        "agent": agent_spec,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "sample_limit": args.limit,
        "build_s": timings["build_s"],
        "duration_s": timings["duration_s"],
        "metrics": metrics,
        "latency": latency_summary(sessions),
        "health": health_summary(sessions),
        "sessions": sessions,
    }


def check_baseline(metrics: dict) -> int:
    reference = baseline()
    if reference is None:
        print("baseline reference missing at docs/baseline_results.json", file=sys.stderr)
        return 1
    failures = [
        f"  {ours}: got {metrics[ours]!r}, reference {reference[theirs]!r}"
        for ours, theirs in BASELINE_FIELDS
        if metrics[ours] != reference[theirs]
    ]
    if failures:
        print("BASELINE MISMATCH (the harness is wrong, not the reference):", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"baseline reproduced exactly: score {metrics['recommended_technical_score']}")
    return 0


def split_samples(samples: list[dict], split: str) -> list[dict]:
    """Returns a stratified half of `samples`, or all of them.

    Alternates within each scenario group, so both halves keep the dataset's
    80/80/30/10 mix. Deterministic and order-independent: it keys on each
    scenario's own running count, never on a hash or a seed. Constants tuned on
    "dev" must be reported on "held".
    """
    if split == "all":
        return samples
    wanted = 0 if split == "dev" else 1
    seen: dict[str, int] = {}
    kept = []
    for sample in samples:
        scenario = str(sample.get("scenario_type"))
        position = seen.get(scenario, 0)
        seen[scenario] = position + 1
        if position % 2 == wanted:
            kept.append(sample)
    return kept


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="TechJam measurement harness")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="module:Class of the agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--label", default="latest", help="run artifact name under --runs-dir")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N sessions")
    parser.add_argument("--split", choices=("all", "dev", "held"), default="all",
                        help="score a stratified half of the dataset")
    parser.add_argument("--compare-to", default=None, help="artifact to diff against")
    parser.add_argument("--no-diff", action="store_true")
    parser.add_argument("--check-baseline", action="store_true",
                        help="assert the run reproduces docs/baseline_results.json")
    parser.add_argument("--no-fast-path", action="store_true",
                        help="disable the agent's template shortcut, so the general "
                             "understanding path is scored on its own")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    require_catalog(catalog_path)

    samples = split_samples(load_jsonl(args.dataset), args.split)
    if args.limit is not None:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(catalog_path)

    started = perf_counter()
    agent = build_agent(load_agent_class(args.agent), catalog_path, args.no_fast_path)
    build_s = round(perf_counter() - started, 2)

    recorder = RecordingAgent(agent)
    started = perf_counter()
    result = evaluate(recorder, samples, catalog_ids, categories, products)
    duration_s = round(perf_counter() - started, 2)

    sessions = analyze(recorder.sessions, samples, result, catalog_ids)
    artifact = build_artifact(
        args, args.agent, result, sessions, {"build_s": build_s, "duration_s": duration_s}
    )

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    current = runs_dir / f"{args.label}.json"
    previous = runs_dir / f"{args.label}.previous.json"
    if current.exists():
        previous.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
    current.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(render(artifact))
    print(f"\nindex build {build_s}s, written to {current}")

    comparison = Path(args.compare_to) if args.compare_to else previous
    if not args.no_diff and comparison.exists():
        print()
        print(diff_module.render(
            diff_module.compare(diff_module.load(comparison), artifact),
            str(comparison), str(current),
        ))

    if args.check_baseline:
        if args.limit is not None or args.split != "all":
            print("--check-baseline requires the full dataset; drop --limit/--split", file=sys.stderr)
            return 1
        print()
        return check_baseline(artifact["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
