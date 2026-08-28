PYTHON ?= python3
export PYTHONPATH := .

.PHONY: eval quick baseline split risk paraphrase trace misses diff test data clean

# Score the agent, write runs/latest.json, diff against the previous run.
eval:
	$(PYTHON) -m harness.run

# Fast iteration loop: first 40 sessions, no artifact rotation.
quick:
	$(PYTHON) -m harness.run --limit 40 --label quick

# The organizer's shipped agent, preserved verbatim in starter/baseline_agent.py.
# The harness is wrong if this fails, not the reference.
baseline:
	$(PYTHON) -m harness.run --agent starter.baseline_agent:Agent --label baseline --check-baseline

# Fit on dev, report on held. A constant tuned on the full 200 is unguarded.
split:
	$(PYTHON) -m harness.run --split dev --label split_dev --no-diff
	$(PYTHON) -m harness.run --split held --label split_held --no-diff

# Risk B: does the score survive a different target distribution?
risk:
	$(PYTHON) -m harness.counterfactual

# Risk A: does the score survive the customer saying the same thing differently?
paraphrase:
	$(PYTHON) -m harness.paraphrase

trace:
	$(PYTHON) -m harness.trace --limit 5 --full

misses:
	$(PYTHON) -m harness.trace --misses --limit 10

diff:
	$(PYTHON) -m harness.diff runs/latest.previous.json runs/latest.json

test:
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) -m unittest discover -s harness/tests -t .
	$(PYTHON) -m unittest discover -s submission/src/tests -t .

data: data/catalog.jsonl
data/catalog.jsonl:
	gzip -dkc catalog.jsonl.gz > data/catalog.jsonl

clean:
	rm -rf runs __pycache__ */__pycache__
