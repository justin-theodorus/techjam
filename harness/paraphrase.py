"""Risk A gate: score the agent when the customer says the same thing differently.

The headline score is conditional on the customer speaking in the exact templates
`evaluator/local_evaluator.py` ships. `docs/competition_specification.md` says
paraphrasing "cannot decide correctness", which promises the *answer* will not move.
It does not promise the wording stays fixed, and the contract types `user_message`
as, in full, `{"type": "string"}`.

This gate holds every session fixed -- same targets, same scenario mix, same
disclosure order, same override turn -- and varies only the language the simulator
uses to say it. Any score change is therefore attributable to understanding and to
nothing else.

    python3 -m harness.paraphrase [--agent module:Class] [--styles clean,reworded]

The `clean` column re-emits the original strings through the same patched code path,
so it must reproduce `make eval` to the digit. If it does not, this harness is wrong.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from evaluator import local_evaluator
from evaluator.local_evaluator import evaluate, load_jsonl

from harness import record
from harness.run import catalog_index, load_agent_class, require_catalog


@dataclass(frozen=True)
class Style:
    """One way of saying the eight things the simulator can say.

    Every field is a tuple of format strings; one is chosen per session from a
    `Random` seeded on the sample id, so a column is reproducible regardless of
    iteration order. The payload placeholders carry the same information in every
    style, which is what makes the columns comparable.
    """

    name: str
    browsing: tuple[str, ...]
    buying: tuple[str, ...]
    override: tuple[str, ...]
    disclosure: tuple[str, ...]
    no_preference: tuple[str, ...]
    no_additional: tuple[str, ...]
    reject: tuple[str, ...]
    pivot: tuple[str, ...]
    joiner: tuple[str, ...] = ("; ",)
    head: tuple[str, ...] = ("",)
    tail: tuple[str, ...] = ("",)
    substitute: bool = False


# The shipped strings, verbatim. `initial_message` lines 156-163, `customer_reply`
# lines 169-185, `behavior_for` line 85.
CLEAN = Style(
    name="clean",
    browsing=("I'm looking for {category}, but I'm still exploring.",),
    buying=("I'm looking for {category}. A key requirement is: {constraint}.",),
    override=("I'm looking for {category}. {value}",),
    disclosure=("For that, what matters is: {payload}.",),
    no_preference=(
        "I don't have a preference for {attribute}; please use your judgment.",
    ),
    no_additional=("I don't have an additional preference for {attribute}.",),
    reject=(
        "Those options are not quite right yet. "
        "Ask me about one specific attribute.",
    ),
    pivot=("Actually, ignore my earlier preference. What I need is: {value}.",),
)

# Same information, different frame. No shipped literal survives.
REWORDED = Style(
    name="reworded",
    browsing=(
        "Show me some {category}, I'm just browsing for now.",
        "Hi, I want to see what {category} you have. Nothing specific yet.",
        "Can you help me find {category}? I haven't decided on details.",
        "I'm shopping for {category} and still figuring out what I want.",
    ),
    buying=(
        "I need {category}. It has to be {constraint}.",
        "Looking to buy {category}. The main thing is {constraint}.",
        "Can you find me {category}? Must be {constraint}.",
        "I want {category} and {constraint} is essential.",
    ),
    override=(
        "I want {category}. {value}",
        "Shopping for {category}. I like {value}.",
        "I need {category}. Ideally {value}.",
    ),
    disclosure=(
        "What I care about is {payload}.",
        "Mainly {payload}.",
        "It should be {payload}.",
        "Well, {payload} matters to me.",
    ),
    no_preference=(
        "No strong feelings on {attribute}, you decide.",
        "{attribute} doesn't matter to me, whatever you think is best.",
        "I'm not fussed about {attribute}.",
    ),
    no_additional=(
        "Nothing else on {attribute}.",
        "That's all I can think of for {attribute}.",
        "No more preferences there.",
    ),
    reject=(
        "Hmm, none of those look right. Ask me something specific.",
        "Not quite. What do you want to know?",
    ),
    pivot=(
        "Actually, scratch that. I really need {value}.",
        "Wait, forget what I said. It has to be {value}.",
        "Change of mind, I actually need {value}.",
    ),
    joiner=(" and ", ", "),
)

# The shipped frames with the separators moved. Isolates delimiter sensitivity
# from vocabulary sensitivity: the payload strings are untouched.
PUNCTUATION = Style(
    name="punctuation",
    browsing=("I'm looking for {category} but I'm still exploring",),
    buying=("I'm looking for {category} - a key requirement is {constraint}",),
    override=("I'm looking for {category}, {value}",),
    disclosure=("For that what matters is {payload}",),
    no_preference=(
        "I don't have a preference for {attribute}, please use your judgment",
    ),
    no_additional=("I don't have an additional preference for {attribute}",),
    reject=(
        "Those options are not quite right yet, "
        "ask me about one specific attribute",
    ),
    pivot=("Actually ignore my earlier preference, what I need is {value}",),
    joiner=(", ", " and "),
)

# The shipped frames, padded. Tests whether the parser anchors on the prefix.
FILLER = Style(
    name="filler",
    browsing=CLEAN.browsing,
    buying=CLEAN.buying,
    override=CLEAN.override,
    disclosure=CLEAN.disclosure,
    no_preference=CLEAN.no_preference,
    no_additional=CLEAN.no_additional,
    reject=CLEAN.reject,
    pivot=CLEAN.pivot,
    head=("Hi there. ", "Hmm, okay. ", "Thanks for asking. ", "Right, so. "),
    tail=(" Thanks!", " If that makes sense.", " Hope that helps.", ""),
)

# Vocabulary substitution inside the constraint strings themselves. Reproduces the
# corruption behind findings 3.17, which is the worst cell measured so far.
SYNONYM = Style(
    name="synonym",
    browsing=CLEAN.browsing,
    buying=CLEAN.buying,
    override=CLEAN.override,
    disclosure=CLEAN.disclosure,
    no_preference=CLEAN.no_preference,
    no_additional=CLEAN.no_additional,
    reject=CLEAN.reject,
    pivot=CLEAN.pivot,
    substitute=True,
)

STYLES = (CLEAN, REWORDED, PUNCTUATION, FILLER, SYNONYM)

# Applied to constraint text only, in the `synonym` column. Chosen so the meaning
# survives and the surface form does not, which is the shift a real paraphrase makes.
SUBSTITUTIONS = (
    ("cotton", "pure cotton fabric"),
    ("polyester", "poly blend"),
    ("nylon", "nylon weave"),
    ("leather", "genuine leather"),
    ("spandex", "stretch spandex"),
    ("machine wash", "washer safe"),
    ("hand wash", "wash by hand"),
    ("imported", "shipped from overseas"),
    ("lightweight", "light weight"),
    ("breathable", "airy"),
    ("comfortable", "comfy"),
    ("adjustable", "can be adjusted"),
    ("closure", "fastening"),
    ("sleeve", "arm"),
    ("pocket", "pouch"),
    ("elastic", "stretchy"),
    ("waistband", "waist band"),
    ("occasion", "event"),
    ("casual", "everyday"),
    ("durable", "long lasting"),
)


def substitute(value: str) -> str:
    """Returns `value` with known vocabulary swapped for an equivalent phrase."""
    lowered = value.lower()
    for source, target in SUBSTITUTIONS:
        if source in lowered:
            lowered = lowered.replace(source, target)
    return lowered


def session_rng(sample: dict) -> random.Random:
    """Returns a per-session generator, so a column does not depend on ordering."""
    return random.Random(f"paraphrase\0{sample.get('sample_id', '')}")


def dress(style: Style, rng: random.Random, template: str, **payload: str) -> str:
    """Formats one template and applies the style's padding."""
    body = template.format(**payload)
    return f"{rng.choice(style.head)}{body}{rng.choice(style.tail)}"


def reply_rng(sample: dict, disclosed: set[str], attribute: str | None) -> random.Random:
    """Returns a per-turn generator. `disclosed` grows monotonically, so it varies."""
    return random.Random(
        f"paraphrase\0{sample.get('sample_id', '')}\0{len(disclosed)}\0{attribute}"
    )


def patched(style: Style) -> dict:
    """Returns replacements for the four language producers in the evaluator.

    Each one reproduces the original's *logic* exactly -- which constraint is
    selected, what enters `disclosed`, how an off-enum attribute is coerced -- and
    changes only the surface string. The override turn is drawn by delegating to
    the original `behavior_for`, so `rng` is consumed in the same order and the
    pivot lands on the same turn it would have without this harness.
    """
    original_card = local_evaluator.intent_card
    original_behavior = local_evaluator.behavior_for

    def intent_card(product: dict, limit: int = 180) -> dict:
        card = original_card(product, limit)
        if not style.substitute:
            return card

        def swap(values: list) -> list:
            out = []
            for value in values:
                cleaned = local_evaluator._clean_constraint(substitute(str(value)), limit)
                out.append(cleaned or str(value))
            return out

        return {
            "target_category": card["target_category"],
            "hard_constraints": swap(card["hard_constraints"]),
            "soft_preferences": swap(card["soft_preferences"]),
        }

    def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
        behavior = original_behavior(scenario, card, rng)
        override = behavior.get("override")
        if override:
            override["message"] = dress(
                style, rng, rng.choice(style.pivot), value=override["new_value"]
            )
        return behavior

    def initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
        rng = session_rng(sample)
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            constraint = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(constraint)
            return dress(
                style, rng, rng.choice(style.buying),
                category=category, constraint=constraint,
            )
        if scenario == "intent_override":
            old_value = str(sample["behavior"]["override"]["old_value"])
            return dress(
                style, rng, rng.choice(style.override),
                category=category, value=old_value,
            )
        return dress(style, rng, rng.choice(style.browsing), category=category)

    def customer_reply(
        sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool
    ) -> tuple[str, bool]:
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        rng = reply_rng(sample, disclosed, attribute)
        if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
            text = dress(style, rng, rng.choice(style.no_preference), attribute=attribute)
            return text, True
        if not attribute:
            return dress(style, rng, rng.choice(style.reject)), boundary_used
        if attribute not in local_evaluator.ALLOWED_ATTRIBUTES:
            attribute = "other"
        constraints = [
            *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
            *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
        ]
        matches = [
            value for value in constraints
            if value not in disclosed
            and (attribute == "other"
                 or local_evaluator.classify_constraint(value) == attribute)
        ][:2]
        if not matches:
            text = dress(style, rng, rng.choice(style.no_additional), attribute=attribute)
            return text, boundary_used
        disclosed.update(matches)
        payload = rng.choice(style.joiner).join(matches)
        text = dress(style, rng, rng.choice(style.disclosure), payload=payload)
        return text, boundary_used

    return {
        "intent_card": intent_card,
        "behavior_for": behavior_for,
        "initial_message": initial_message,
        "customer_reply": customer_reply,
    }


def measure(agent, style: Style, public: list[dict], catalog_ids, categories, by_asin) -> dict:
    """Returns TechnicalScore and the health counters, with the customer in `style`.

    The health counters matter more than the score. A collapse can mean the agent
    crashed, or it can mean the agent politely served the same ten popular products
    to every session. Those are different bugs and only the counters tell them apart.
    """
    replacements = patched(style)
    originals = {name: getattr(local_evaluator, name) for name in replacements}
    for name, function in replacements.items():
        setattr(local_evaluator, name, function)
    recorder = record.RecordingAgent(agent)
    try:
        result = evaluate(recorder, public, catalog_ids, categories, by_asin)
    finally:
        for name, function in originals.items():
            setattr(local_evaluator, name, function)

    turns = [turn for session in recorder.sessions for turn in session["turns"]]
    first_slates = {
        tuple(session["turns"][0]["raw_recommendations"])
        for session in recorder.sessions if session["turns"]
    }
    return {
        "score": result["recommended_technical_score"],
        "exceptions": sum(1 for turn in turns if turn["error"]),
        "discarded": sum(1 for turn in turns if turn["discarded"]),
        "distinct_turn1_slates": len(first_slates),
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Paraphrase (Risk A) gate")
    parser.add_argument("--agent", default="submission.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--styles", default="", help="comma-separated subset of style names")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    require_catalog(catalog_path)

    wanted = {name.strip() for name in args.styles.split(",") if name.strip()}
    styles = [style for style in STYLES if not wanted or style.name in wanted]
    if not styles:
        raise SystemExit(f"no styles matched {args.styles!r}")

    public = load_jsonl(args.dataset)
    catalog_ids, categories, by_asin = catalog_index(catalog_path)

    started = perf_counter()
    agent = load_agent_class(args.agent)(str(catalog_path))
    build_s = round(perf_counter() - started, 2)

    rows = {
        style.name: measure(agent, style, public, catalog_ids, categories, by_asin)
        for style in styles
    }
    scores = {name: row["score"] for name, row in rows.items()}

    print(f"{args.agent}   catalog build {build_s}s, {len(public)} sessions per column\n")
    print(f"{'':>22}" + "".join(f"{name:>15}" for name in rows) + f"{'worst':>15}")
    print(
        f"{'TechnicalScore':>22}"
        + "".join(f"{value:>15.4f}" for value in scores.values())
        + f"{min(scores.values()):>15.4f}"
    )
    for label, key in (
        ("exceptions", "exceptions"),
        ("discarded", "discarded"),
        ("distinct turn-1 slates", "distinct_turn1_slates"),
    ):
        print(f"{label:>22}" + "".join(f"{row[key]:>15}" for row in rows.values()))

    if "clean" in scores:
        print("\n`clean` re-emits the shipped strings; it must equal `make eval` exactly")
    print(
        "a low score with zero exceptions and one distinct turn-1 slate means the\n"
        "agent understood nothing and fell back to global popularity for every session"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
