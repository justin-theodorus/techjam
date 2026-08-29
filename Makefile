PYTHON ?= python3
export PYTHONPATH := .

.PHONY: eval quick baseline split risk paraphrase sessions deviations dense llm trace misses diff test data clean

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

# The instrument, not a result: how hard is each frozen synthetic set, and
# where does the shipped agent's rank land on it?
sessions:
	$(PYTHON) -m harness.sessions

# Every component with a live switch, re-swept against sets that still have
# headroom for them to move. Ten of them since Phase 6U.
deviations:
	$(PYTHON) -m harness.deviations

# Tier 1, switched off in the shipped configuration: the full battery with the
# dense track's one non-negative setting turned on. Reported as a column, never
# as a headline (findings 3.35).
dense:
	$(PYTHON) -m harness.deviations --component dense,dense_route,dense_negation
	$(PYTHON) -c "from submission.src import routing; \
	  routing.DISCOVERY_REACH = 100; \
	  from harness import counterfactual; counterfactual.main()"
	$(PYTHON) -c "from submission.src import routing; \
	  routing.DISCOVERY_REACH = 100; \
	  from harness import paraphrase; paraphrase.main()"

# Tier 2, switched off in the shipped configuration: the model rerank, on the
# one gate that reports tokens and latency and on the readable sets that still
# have rank headroom for a permutation to move. Requires network, credentials
# and real money, roughly 2,500 model calls, and is never reached by
# `make deviations` (findings 3.36).
llm:
	USE_LLM=1 $(PYTHON) -m harness.run --llm --label llm --no-diff
	USE_LLM=1 $(PYTHON) -m harness.deviations --component llm_rerank --set \
	  twin_cards,comparative_constraints,unstated_constraints,reworded_constraints,silent_customer

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
