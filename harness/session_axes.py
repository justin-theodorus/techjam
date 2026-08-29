"""The five axes a synthetic session set can vary, one value at a time.

Every axis has a neutral value that reproduces what the public set already
does, so a recipe leaving all five alone manufactures rows the shipped
simulator handles exactly as it handles its own. Nothing here reads a score.

The transforms keep every message *frame* the evaluator ships and perturb only
the constraint payloads. `harness/analysis.py` recognises turns by those frames
and splits disclosures on `;`, so a reworded frame or a semicolon inside a
payload would silently stop the trace parsing.
"""

from __future__ import annotations

import collections
import random
import re
from dataclasses import dataclass

from evaluator import local_evaluator

from harness import counterfactual
from harness import paraphrase

CROWDED_BUCKET = 300
SPARSE_BUCKET = 20
THIN_FEATURES = 1
EARLY_PIVOT_TURN = 2
LATE_PIVOT_TURN = 8

# `intent_card`'s own default, repeated so a transform's output is clipped the
# same way the evaluator would have clipped it.
CONSTRAINT_LIMIT = 180

POOL_NAMES = ("any", "crowded", "sparse", "thin", "twin")
WEIGHT_NAMES = ("size-biased", "sqrt", "uniform")
TEXT_NAMES = ("verbatim", "synonym", "abbreviate", "negate", "comparative",
              "implicit", "typo")
PROFILE_NAMES = ("public", "wide", "adversarial", "empty")
DIALOGUE_NAMES = ("default", "front_loaded", "silent", "early_pivot",
                  "late_pivot", "unrelated_pivot")
SHOPPER_NAMES = ("distinct", "returning")

# How many sessions one returning shopper is given. Three, because two
# cannot distinguish "memory helped" from "the second visit was easier",
# and every visit past the first costs a third of the set's width.
VISITS_PER_SHOPPER = 3


@dataclass(frozen=True)
class CatalogFacts:
    """Bucket membership and intent-card uniqueness over the whole catalog."""

    bucket: dict[str, str]
    bucket_size: dict[str, int]
    twins: frozenset[str]
    bucket_names: tuple[str, ...]


def coarse(product: dict) -> str:
    """Returns the bucket `evaluate()` greets this product's session with."""
    return local_evaluator.coarse_category(
        [str(value) for value in product.get("categories") or []]
    )


def _card_key(product: dict) -> tuple:
    card = local_evaluator.intent_card(product)
    return (tuple(card["hard_constraints"]), tuple(card["soft_preferences"]))


def survey(products: list[dict]) -> CatalogFacts:
    """Returns the catalog facts the pool predicates and profiles need.

    Card uniqueness is computed inside each bucket rather than catalog-wide,
    because a twin only costs anything when it survives the category filter
    the agent applies first (`findings.md` 3.12).
    """
    bucket: dict[str, str] = {}
    members: dict[str, list[dict]] = collections.defaultdict(list)
    for product in products:
        name = coarse(product)
        bucket[str(product["parent_asin"])] = name
        members[name].append(product)

    twins: set[str] = set()
    for group in members.values():
        keys = [_card_key(item) for item in group]
        counts = collections.Counter(keys)
        for item, key in zip(group, keys):
            if counts[key] > 1:
                twins.add(str(item["parent_asin"]))

    return CatalogFacts(
        bucket=bucket,
        bucket_size={name: len(group) for name, group in members.items()},
        twins=frozenset(twins),
        bucket_names=tuple(sorted(members)),
    )


def _bucket_size(product: dict, facts: CatalogFacts) -> int:
    return facts.bucket_size[facts.bucket[str(product["parent_asin"])]]


def _is_crowded(product: dict, facts: CatalogFacts) -> bool:
    return _bucket_size(product, facts) >= CROWDED_BUCKET


def _is_sparse(product: dict, facts: CatalogFacts) -> bool:
    return _bucket_size(product, facts) <= SPARSE_BUCKET


def _is_thin(product: dict, facts: CatalogFacts) -> bool:
    """A short `features` list and no price starve the intent card together."""
    return (len(product.get("features") or []) <= THIN_FEATURES
            and product.get("price") in (None, ""))


def _is_twin(product: dict, facts: CatalogFacts) -> bool:
    return str(product["parent_asin"]) in facts.twins


_POOL_PREDICATES = {
    "crowded": _is_crowded,
    "sparse": _is_sparse,
    "thin": _is_thin,
    "twin": _is_twin,
}


def pool(name: str, products: list[dict], facts: CatalogFacts) -> list[dict]:
    """Returns the products a recipe with this `pool` may draw targets from.

    Predicates stack with `+`, so "crowded+thin" is the intersection. That is
    how a compound recipe reaches a difficulty no single lever produces.
    """
    parts = name.split("+")
    unknown = [part for part in parts if part not in POOL_NAMES]
    if unknown:
        raise ValueError(
            f"unknown pool {'+'.join(unknown)!r}, expected some of {POOL_NAMES}"
        )
    keeps = [_POOL_PREDICATES[part] for part in parts if part != "any"]
    if not keeps:
        return list(products)
    return [item for item in products
            if all(keep(item, facts) for keep in keeps)]


def _settle(value: str, original: str) -> str:
    """Cleans a transformed constraint the way the evaluator would have.

    A transform that changed nothing hands back the original untouched. That is
    what makes a neutral recipe's authored card byte-identical to the derived
    one, and the derived one carries strings this would otherwise re-clean.
    """
    if value == original:
        return original
    cleaned = local_evaluator._clean_constraint(value, CONSTRAINT_LIMIT)
    return cleaned or original


FUNCTION_WORDS = frozenset({
    "a", "an", "and", "the", "of", "for", "with", "in", "on", "to", "is",
    "are", "made", "from", "by", "or", "that", "this", "it", "its",
})

# One tuple per `classify_constraint` arm. A card holds at most four
# constraints, so four phrases per arm is what it takes for two constraints of
# the same type never to want the same string; `constraints()` still resolves a
# collision if one happens. None of them names anything in the catalog, which
# is the point.
COMPARATIVES = {
    "budget": ("cheaper than the last one", "a bit under my usual spend",
               "better value than the one I bought", "not as pricey as that"),
    "size": ("a size up from usual", "roomier than the one I own",
             "narrower than my last pair", "longer in the body than that"),
    "material": ("softer than what I have", "less scratchy than my last one",
                 "heavier than the one I own", "thinner than that felt"),
    "color": ("a shade darker than that", "less bright than the last",
              "warmer toned than mine", "closer to what I wore before"),
    "style": ("a bit dressier than that", "plainer than the one I own",
              "more fitted than my usual", "less fussy than the last"),
    "use_case": ("sturdier than what I own", "better for daily wear than mine",
                 "more suited to the cold than that",
                 "easier to travel with than my last"),
    "feature": ("better made than the last one", "nicer than the one I have",
                "less fiddly than my current one",
                "more practical than that was"),
}

# How far to walk a transform's own option list looking for a phrase that has
# not already been used in this card.
COLLISION_TRIES = 8


def _verbatim(value: str, position: int, product: dict,
              rng: random.Random) -> str:
    return value


def _synonym(value: str, position: int, product: dict,
             rng: random.Random) -> str:
    return paraphrase.substitute(value)


def _abbreviate(value: str, position: int, product: dict,
                rng: random.Random) -> str:
    tokens = [token for token in re.findall(r"[\w$%.-]+", value)
              if token.lower() not in FUNCTION_WORDS]
    return " ".join(tokens[:3])


def _negate(value: str, position: int, product: dict,
            rng: random.Random) -> str:
    """States a material the target genuinely does not have.

    Drawing the word from the target's own record is what keeps the sentence
    true. An authored negative naming something the product *is* would be
    fiction, and would make the set untestable rather than hard.
    """
    corpus = local_evaluator.searchable_text(product).lower()
    absent = [word for word in local_evaluator.MATERIALS if word not in corpus]
    if not absent:
        return f"not {value}"
    return f"not {absent[position % len(absent)]}"


def _comparative(value: str, position: int, product: dict,
                 rng: random.Random) -> str:
    options = COMPARATIVES[local_evaluator.classify_constraint(value)]
    return options[position % len(options)]


# Wide enough to reach 47.3% of the catalog against a 52.2% ceiling set by
# `description` population, narrow enough that the result still reads as one
# thing a customer would say.
PHRASE_WORDS = (3, 20)


def _description_phrases(product: dict) -> list[str]:
    text = " ".join(str(item) for item in product.get("description") or [])
    parts = re.split(r"[.!?;]", text)
    shortest, longest = PHRASE_WORDS
    return [part for part in (raw.strip() for raw in parts)
            if shortest <= len(part.split()) <= longest]


def _implicit(value: str, position: int, product: dict,
              rng: random.Random) -> str:
    """Names something true of the product that its indexed text never says.

    `description` is the one populated field the agent deliberately does not
    index (`findings.md` 3.13), so a phrase drawn from it is genuinely the
    product's own and genuinely invisible to retrieval. Only 52.2% of the
    catalog carries one; the rest fall back to the constraint unchanged.
    """
    phrases = _description_phrases(product)
    if not phrases:
        return value
    return phrases[position % len(phrases)]


def _typo(value: str, position: int, product: dict,
          rng: random.Random) -> str:
    words = []
    for word in value.split(" "):
        if len(word) <= 5:
            words.append(word)
            continue
        at = rng.randrange(1, len(word) - 1)
        words.append(word[:at] + word[at + 1] + word[at] + word[at + 2:])
    return " ".join(words)


_TEXT_TRANSFORMS = {
    "verbatim": _verbatim,
    "synonym": _synonym,
    "abbreviate": _abbreviate,
    "negate": _negate,
    "comparative": _comparative,
    "implicit": _implicit,
    "typo": _typo,
}


def constraints(name: str, values: list[str], product: dict,
                rng: random.Random) -> list[str]:
    """Returns `values` reworded by the named transform, count and order kept.

    **A transform must be a bijection on the card's distinct values.** It has
    to fail in both directions and each one silently rewrites how much the
    session knows, which is the one thing this axis must not touch:

    - Splitting a repeat. `intent_card()` lets `soft_preferences` fall back to
      `cleaned[:1]`, so a card repeats a constraint on 1.5% of the catalog and
      6.9% of the thin pool. `customer_reply()` dedupes those against
      `disclosed`, so a repeat is worth one disclosure, not two. Rewriting each
      position independently would hand the session a constraint it never had.
      Each distinct value is therefore transformed once and reused, which is
      also what a customer does: saying the same thing twice, they say it the
      same way twice.
    - Merging two facts. A transform drawing from a fixed phrase list can hand
      two genuinely different constraints the same string, and then the second
      is already in `disclosed` and can never be surfaced by a probe. Measured
      before this guard: `comparative` merged on 56.5% of catalog cards,
      `implicit` on 25.3%, `synonym` on 2.0%. Each distinct value therefore
      walks its own option list until it lands on something unused, and keeps
      its original wording if nothing is free.

    `position` is the index among the distinct values, so the caller must pass
    hard and soft constraints in one call. Two calls would restart it and could
    hand two different constraints the same replacement.
    """
    if name not in TEXT_NAMES:
        raise ValueError(f"unknown text {name!r}, expected one of {TEXT_NAMES}")
    transform = _TEXT_TRANSFORMS[name]
    reworded: dict[str, str] = {}
    taken: set[str] = set()
    for value in values:
        text = str(value)
        if text in reworded:
            continue
        position = len(reworded)
        chosen = text
        for offset in range(COLLISION_TRIES):
            candidate = _settle(
                transform(text, position + offset, product, rng), text)
            if candidate not in taken:
                chosen = candidate
                break
        reworded[text] = chosen
        taken.add(chosen)
    return [reworded[str(value)] for value in values]


WIDE_TAGS = (
    "fit", "comfort", "material", "style", "durability", "performance",
    "warmth", "weather", "price", "packability", "breathability", "colourfast",
    "stretch", "coverage", "sustainability", "ease of care", "brand",
    "return policy", "sizing consistency", "occasion", "layering", "grip",
    "weight", "adjustability",
)
FREQUENCIES = (
    "first purchase", "1-2 prior purchases", "3-4 prior purchases",
    "5-9 prior purchases", "10+ prior purchases",
)
RATING_STYLES = (
    "usually positive", "critical", "mixed", "terse", "effusive",
)
TAGS_PER_PROFILE = 3


def _summarise(tags: list[str], style: str) -> str:
    return (f"Prior purchases emphasize {', '.join(tags)}; "
            f"ratings are {style}.")


def _wide(rng: random.Random, product: dict) -> dict:
    tags = rng.sample(WIDE_TAGS, TAGS_PER_PROFILE)
    style = rng.choice(RATING_STYLES)
    return {
        "purchase_frequency": rng.choice(FREQUENCIES),
        "average_prior_rating": round(rng.uniform(1.0, 5.0), 1),
        "rating_style": style,
        "preference_tags": tags,
        "summary": _summarise(tags, style),
    }


def _adversarial(rng: random.Random, product: dict) -> dict:
    """Tags whose words the target does not carry, so the profile misleads.

    Authored, and provisional: it assumes a tag word absent from the target's
    text is one the target would not satisfy. That is a proxy for contradiction
    rather than contradiction itself.
    """
    corpus = local_evaluator.searchable_text(product).lower()
    away = [tag for tag in WIDE_TAGS if tag not in corpus]
    tags = rng.sample(away or list(WIDE_TAGS), TAGS_PER_PROFILE)
    style = rng.choice(RATING_STYLES)
    return {
        "purchase_frequency": rng.choice(FREQUENCIES),
        "average_prior_rating": round(rng.uniform(1.0, 5.0), 1),
        "rating_style": style,
        "preference_tags": tags,
        "summary": _summarise(tags, style),
    }


def _empty(rng: random.Random, product: dict) -> dict:
    return {
        "purchase_frequency": "",
        "average_prior_rating": None,
        "rating_style": "",
        "preference_tags": [],
        "summary": "",
    }


def profile(name: str, rng: random.Random, public_profiles: list[dict],
            product: dict) -> dict:
    """Returns one `user_profile`, always with exactly the contract's keys."""
    if name not in PROFILE_NAMES:
        raise ValueError(
            f"unknown profiles {name!r}, expected one of {PROFILE_NAMES}"
        )
    if name == "public":
        return dict(rng.choice(public_profiles))
    if name == "wide":
        return _wide(rng, product)
    if name == "adversarial":
        return _adversarial(rng, product)
    return _empty(rng, product)


def shape_card(name: str, card: dict, rng: random.Random) -> dict:
    """Returns `card` with its disclosure *shape* changed, not its wording."""
    if name not in DIALOGUE_NAMES:
        raise ValueError(
            f"unknown dialogue {name!r}, expected one of {DIALOGUE_NAMES}"
        )
    if name == "front_loaded":
        values = list(dict.fromkeys(
            [*card["hard_constraints"], *card["soft_preferences"]]
        ))
        if not values:
            return card
        return {
            "target_category": card["target_category"],
            "hard_constraints": [_settle(", ".join(values), values[0])],
            "soft_preferences": [],
        }
    if name == "silent":
        return {
            "target_category": card["target_category"],
            "hard_constraints": [],
            "soft_preferences": [],
        }
    return card


def _other_bucket(product: dict, facts: CatalogFacts,
                  rng: random.Random) -> str:
    own = facts.bucket[str(product["parent_asin"])]
    for _ in range(4):
        name = rng.choice(facts.bucket_names)
        if name != own:
            return name.lower()
    return "something else entirely"


def shape_behavior(name: str, behavior: dict, product: dict,
                   facts: CatalogFacts, rng: random.Random) -> dict:
    """Returns `behavior` with the pivot moved or repointed.

    A no-op outside intent_override, because `evaluate():234` starts every
    other scenario with `override_applied` already true and never reads the
    dict. The pivot frame is reproduced byte for byte so `analysis.py` keeps
    recognising the turn.
    """
    override = behavior.get("override")
    if not override or name not in ("early_pivot", "late_pivot",
                                    "unrelated_pivot"):
        return behavior
    turn = override["turn"]
    new_value = override["new_value"]
    if name == "early_pivot":
        turn = EARLY_PIVOT_TURN
    elif name == "late_pivot":
        turn = LATE_PIVOT_TURN
    else:
        new_value = _other_bucket(product, facts, rng)
    return {
        **behavior,
        "override": {
            "turn": turn,
            "old_value": override["old_value"],
            "new_value": new_value,
            "message": ("Actually, ignore my earlier preference. "
                        f"What I need is: {new_value}."),
        },
    }


SCENARIO_MIX = counterfactual.SCENARIO_MIX


@dataclass(frozen=True)
class Shoppers:
    """Who each row belongs to, and the target that visit lands on."""

    targets: tuple[dict, ...]
    ids: tuple[str | None, ...]
    visits: tuple[int, ...]


def _members(candidates: list[dict],
             facts: CatalogFacts) -> dict[str, list[dict]]:
    """Groups the drawable products by the bucket a session opens with."""
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for product in candidates:
        grouped[facts.bucket[str(product["parent_asin"])]].append(product)
    return grouped


def _visits(anchor: dict, members: list[dict], scheme: str, bucket: str,
            rng: random.Random) -> list[dict]:
    """Returns one shopper's visits, the anchor first.

    Later visits are drawn from the anchor's own bucket under the set's own
    weighting scheme. Holding that rule fixed is what makes the visit blocks
    comparable at all: drawn any other way, a later block confounds "memory
    helped" with "this one was easier", which is a difference the control can
    only subtract after the fact rather than prevent.
    """
    chosen = [anchor]
    taken = {str(anchor["parent_asin"])}
    available = [
        product for product in members
        if str(product["parent_asin"]) not in taken
    ]
    weights = counterfactual.weight_schemes(available)[scheme]
    while len(chosen) < VISITS_PER_SHOPPER and available:
        total = sum(weights)
        if total <= 0.0:
            break
        cut = rng.random() * total
        running = 0.0
        for position, weight in enumerate(weights):
            running += weight
            if running >= cut:
                break
        chosen.append(available.pop(position))
        weights.pop(position)
    # An intersected pool such as `crowded+thin` can leave a bucket with one
    # or two members. Refuse the set rather than pad from another bucket: a
    # shopper whose visits span two departments has nothing true to remember,
    # which is the one property this instrument exists to have. Repeating a
    # target would be worse still, since it hands a later visit a session the
    # shopper has already converted.
    if len(chosen) < VISITS_PER_SHOPPER:
        raise ValueError(
            f"bucket {bucket!r} holds {len(members)} drawable products, "
            f"too few for {VISITS_PER_SHOPPER} distinct visits"
        )
    return chosen



def shoppers(name: str, targets: list[dict], facts: CatalogFacts,
             candidates: list[dict], scheme: str,
             rng: random.Random) -> Shoppers:
    """Assigns an identity to each row, and the visits that identity implies.

    The sixth axis, and the only one that relates rows to each other: an
    identity is by definition not a property of one row, so this reads the
    whole drawn target list where every other axis reads one product.

    Rows are laid out visit-major, so every first visit is scored before any
    second visit. That is not cosmetic: `evaluate()` walks the rows in order,
    so a memory has to be written by an earlier row to be read by a later one.

    Each shopper's first visit is the target the set already drew for that
    position, so the visit-1 block is distributed exactly as any other set's
    rows are; only the later visits are drawn here.

    Args:
        name: One of `SHOPPER_NAMES`.
        targets: The products `_draw` selected, in row order.
        facts: Bucket membership, for keeping a shopper inside one bucket.
        candidates: The pool those targets were drawn from.
        scheme: The set's own weighting scheme, reused for the later visits.
        rng: This axis's own generator, never the set's profile generator.

    Raises:
        ValueError: If `name` is not a known identity axis value.
    """
    if name not in SHOPPER_NAMES:
        raise ValueError(f"unknown shoppers {name!r}")
    total = len(targets)
    if name == "distinct":
        return Shoppers(tuple(targets), (None,) * total, (0,) * total)

    if total < VISITS_PER_SHOPPER:
        raise ValueError(
            f"{total} rows cannot hold a shopper of "
            f"{VISITS_PER_SHOPPER} visits"
        )
    count = total // VISITS_PER_SHOPPER
    members = _members(candidates, facts)
    groups = []
    for shopper in range(count):
        bucket = facts.bucket[str(targets[shopper]["parent_asin"])]
        groups.append(
            _visits(targets[shopper], members[bucket], scheme, bucket, rng)
        )

    placed = list(targets)
    ids: list[str | None] = [None] * total
    visits = [0] * total
    for position in range(count * VISITS_PER_SHOPPER):
        shopper, visit = position % count, position // count
        placed[position] = groups[shopper][visit]
        ids[position] = f"shopper_{shopper:04d}"
        visits[position] = visit + 1
    return Shoppers(tuple(placed), tuple(ids), tuple(visits))


@dataclass(frozen=True)
class Recipe:
    """One reproducible session set: a point in the five-axis space.

    Every default reproduces what the public set already does, so a recipe that
    names only itself manufactures sessions the shipped simulator cannot tell
    from its own.
    """

    name: str
    seed: int
    count: int = 200
    mix: tuple[tuple[str, int], ...] = SCENARIO_MIX
    pool: str = "any"
    weights: str = "size-biased"
    text: str = "verbatim"
    profiles: str = "public"
    dialogue: str = "default"
    shoppers: str = "distinct"

    @property
    def is_authored(self) -> bool:
        """Whether rows must carry their own card rather than deriving one."""
        return self.text != "verbatim" or self.dialogue != "default"
