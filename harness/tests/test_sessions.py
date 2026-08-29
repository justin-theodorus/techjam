from __future__ import annotations

import json
import os
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

from evaluator import local_evaluator

from harness import session_axes
from harness import session_sets
from harness import sessions
from harness.tests import fixtures

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "docs" / "agent_api_contract.json"
REAL_CATALOG = REPO_ROOT / "data" / "catalog.jsonl"

CROWDED_COUNT = session_axes.CROWDED_BUCKET + 20
SPARSE_COUNT = session_axes.SPARSE_BUCKET - 8
MIDDLE_COUNT = 60

PUBLIC_PROFILES = [
    {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 5.0,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort"],
        "summary": "Prior purchases emphasize fit, comfort.",
    },
    {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 1.0,
        "rating_style": "critical",
        "preference_tags": ["style"],
        "summary": "Prior purchases emphasize style; ratings are critical.",
    },
]

JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "array": list,
    "object": dict,
    "null": type(None),
}


def _product(index: int, bucket: str, thin: bool = False,
             twin: bool = False) -> dict:
    """One row, shaped so the pool predicates can actually separate it."""
    thin = thin and not twin  # a two-bullet twin is not a thin card
    if twin:
        features = ["cotton blend upper", "Imported"]
        description = ["A plain thing that many others also are."]
    elif thin:
        features = []
        description = []
    else:
        features = [
            f"polyester shell number {index}",
            f"Machine wash cold on cycle {index % 7}",
            "Adjustable strap for a secure comfortable fit",
        ]
        description = [
            f"Built for daily wear and packs down small. Ships in box {index}."
        ]
    return {
        "parent_asin": f"B{index:07d}",
        "title": f"Example product {index}",
        "features": features,
        "description": description,
        "price": None if thin else 19.99,
        "categories": ["Clothing, Shoes & Jewelry", "Women", bucket],
        "details": {} if thin else {"Department": "Womens",
                                    "Color": "blue" if index % 2 else "black"},
        "average_rating": 4.1,
        "rating_number": 1 if thin else (index * 13) % 9000 + 1,
        "store": f"Store {index % 5}",
    }


def catalog_rows() -> list[dict]:
    """A catalog small enough to be fast and varied enough to exercise pools."""
    rows, index = [], 0
    for _ in range(CROWDED_COUNT):
        rows.append(_product(index, "Shirts", thin=index % 5 == 0,
                             twin=index % 11 == 0))
        index += 1
    for _ in range(SPARSE_COUNT):
        rows.append(_product(index, "Leg Warmers"))
        index += 1
    for _ in range(MIDDLE_COUNT):
        rows.append(_product(index, "Belts", twin=index % 4 == 0))
        index += 1
    return rows


def write_catalog(root: pathlib.Path) -> pathlib.Path:
    path = root / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in catalog_rows()),
        encoding="utf-8",
    )
    return path


def build(recipe: session_axes.Recipe) -> list[dict]:
    """Generates one set against the fixture catalog."""
    products = catalog_rows()
    facts = session_axes.survey(products)
    return sessions.generate(recipe, products, facts, PUBLIC_PROFILES)


def serialized(name: str = "probe") -> str:
    """Stable JSON for the determinism check, callable from a subprocess.

    `pool="twin"` and `dialogue="unrelated_pivot"` are deliberate: they are the
    only two paths that read `CatalogFacts.twins` and `bucket_names`, which are
    the only structures a `set` could leak iteration order through.
    """
    recipe = session_axes.Recipe(name, 4242, count=40, pool="twin", text="typo",
                                profiles="wide", dialogue="unrelated_pivot",
                                mix=(("intent_override", 1),))
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in build(recipe)
    )


def _matches(value: object, spec: dict) -> bool:
    kinds = spec["type"]
    if isinstance(kinds, str):
        kinds = [kinds]
    for kind in kinds:
        if kind == "number" and isinstance(value, bool):
            continue
        if isinstance(value, JSON_TYPES[kind]):
            return True
    return False


class ContractTest(unittest.TestCase):
    """The contract fixes `user_profile` with `additionalProperties: false`.

    A sixth key is contract-invalid even though the local evaluator tolerates
    it, so a synthetic profile that drifts would pass here and fail officially.
    """

    def setUp(self) -> None:
        self.schema = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.profile_schema = (
            self.schema["reset_request"]["properties"]["user_profile"]
        )

    def test_module_constant_matches_the_contract_file(self) -> None:
        self.assertEqual(
            sessions.PROFILE_KEYS, frozenset(self.profile_schema["required"])
        )

    def test_every_profile_flavour_validates_against_the_contract(self) -> None:
        self.assertFalse(self.profile_schema["additionalProperties"])
        for flavour in session_axes.PROFILE_NAMES:
            recipe = session_axes.Recipe("p", 1, count=12, profiles=flavour)
            for row in build(recipe):
                with self.subTest(profiles=flavour):
                    profile = row["user_profile"]
                    self.assertEqual(
                        set(profile), set(self.profile_schema["required"])
                    )
                    for key, spec in self.profile_schema["properties"].items():
                        self.assertTrue(
                            _matches(profile[key], spec),
                            f"{flavour}.{key} is {profile[key]!r}",
                        )


class FormatTest(unittest.TestCase):
    """A synthetic row must be indistinguishable from a public one in shape."""

    def test_neutral_rows_carry_exactly_the_six_public_keys(self) -> None:
        for row in build(session_axes.Recipe("f", 1, count=20)):
            self.assertEqual(set(row), {
                "sample_id", "scenario_type", "category_bucket",
                "difficulty_bucket", "ground_truth", "user_profile",
            })

    def test_authored_rows_carry_both_hidden_fields_or_neither(self) -> None:
        recipe = session_axes.Recipe("f", 1, count=20, text="negate")
        for row in build(recipe):
            self.assertIn("intent_card", row)
            self.assertIn("behavior", row)

    def test_sample_ids_are_unique_and_never_look_public(self) -> None:
        rows = build(session_axes.Recipe("f", 1, count=40))
        identifiers = [row["sample_id"] for row in rows]
        self.assertEqual(len(set(identifiers)), len(identifiers))
        for identifier in identifiers:
            self.assertFalse(identifier.startswith("public_"))

    def test_every_target_is_in_the_catalog(self) -> None:
        catalog = {row["parent_asin"] for row in catalog_rows()}
        for row in build(session_axes.Recipe("f", 1, count=40)):
            self.assertIn(row["ground_truth"]["parent_asin"], catalog)

    def test_the_scenario_mix_is_kept_when_the_count_changes(self) -> None:
        for count in (40, 100, 200, 37):
            with self.subTest(count=count):
                rows = build(session_axes.Recipe("f", 1, count=count))
                self.assertEqual(len(rows), count)


class DeterminismTest(unittest.TestCase):
    """A frozen seed is worthless if the bytes move between processes.

    String hashing is salted per process, so a generator that iterated a set
    anywhere would drift silently and every number taken against a set would
    stop being reproducible.
    """

    def _run(self, hash_seed: str) -> str:
        script = (
            "import sys;"
            "from harness.tests import test_sessions as suite;"
            "sys.stdout.write(suite.serialized())"
        )
        environment = {
            **os.environ,
            "PYTHONHASHSEED": hash_seed,
            "PYTHONPATH": str(REPO_ROOT),
        }
        finished = subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True,
            text=True, cwd=REPO_ROOT, env=environment,
        )
        return finished.stdout

    def test_the_same_seed_repeats_inside_one_process(self) -> None:
        self.assertEqual(serialized(), serialized())

    def test_the_same_seed_is_byte_identical_across_hash_seeds(self) -> None:
        self.assertEqual(self._run("0"), self._run("12345"))

    def test_a_different_seed_produces_a_different_set(self) -> None:
        first = build(session_axes.Recipe("d", 1, count=40))
        second = build(session_axes.Recipe("d", 2, count=40))
        self.assertNotEqual(
            [row["ground_truth"] for row in first],
            [row["ground_truth"] for row in second],
        )


class AuthoredParityTest(unittest.TestCase):
    """The authoring path must reproduce the derivation before it perturbs it.

    Every difficulty claim rests on this: if a neutral authored card already
    differs from the derived one, no later column can be attributed to the axis
    that was supposed to have moved.
    """

    def test_a_neutral_recipe_authors_what_evaluate_derives(self) -> None:
        recipe = session_axes.Recipe("a", 1)
        products = catalog_rows()
        facts = session_axes.survey(products)
        by_asin = {row["parent_asin"]: row for row in products}
        scenarios = ("buying", "browsing", "intent_override", "boundary")
        for product in products[:60]:
            for scenario in scenarios:
                with self.subTest(asin=product["parent_asin"],
                                  scenario=scenario):
                    sample_id = "syn_a_0000"
                    got = sessions._author(
                        recipe, scenario, sample_id, product, facts)
                    row = {
                        "sample_id": sample_id,
                        "scenario_type": scenario,
                        "ground_truth": {
                            "parent_asin": product["parent_asin"]},
                    }
                    self.assertEqual(
                        got,
                        local_evaluator.materialize_hidden_fields(row, by_asin),
                    )


class AxisIsolationTest(unittest.TestCase):
    """An effect is attributable only if one axis moved and no other did."""

    def _targets(self, **overrides) -> list[str]:
        recipe = session_axes.Recipe("i", 9, count=60, **overrides)
        return [row["ground_truth"]["parent_asin"] for row in build(recipe)]

    def _profiles(self, **overrides) -> list[dict]:
        recipe = session_axes.Recipe("i", 9, count=60, **overrides)
        return [row["user_profile"] for row in build(recipe)]

    def test_wording_and_shape_never_move_the_targets(self) -> None:
        baseline = self._targets()
        for axis, value in (("text", "typo"), ("text", "negate"),
                            ("profiles", "wide"), ("profiles", "empty"),
                            ("dialogue", "silent"),
                            ("dialogue", "late_pivot")):
            with self.subTest(axis=axis, value=value):
                self.assertEqual(self._targets(**{axis: value}), baseline)

    def test_wording_and_shape_never_move_the_profiles(self) -> None:
        baseline = self._profiles()
        for axis, value in (("text", "typo"), ("dialogue", "silent"),
                            ("pool", "crowded"), ("weights", "uniform")):
            with self.subTest(axis=axis, value=value):
                self.assertEqual(self._profiles(**{axis: value}), baseline)

    def test_the_target_axes_do_move_the_targets(self) -> None:
        baseline = self._targets()
        for axis, value in (("pool", "sparse"), ("weights", "uniform")):
            with self.subTest(axis=axis, value=value):
                self.assertNotEqual(self._targets(**{axis: value}), baseline)

    def test_a_transform_keeps_a_repeated_constraint_repeated(self) -> None:
        """`intent_card()` lets soft_preferences fall back to hard's first.

        `customer_reply()` dedupes those against `disclosed`, so a transform
        that split them would hand the session a constraint it never had and
        make a wording column read as an information column.
        """
        product = {
            "parent_asin": "B0000001", "title": "Plain thing",
            "features": ["cotton"], "description": [], "price": None,
            "categories": ["Clothing, Shoes & Jewelry", "Shirts"],
            "details": {}, "average_rating": 4.0, "rating_number": 5,
            "store": "Store 0",
        }
        card = local_evaluator.intent_card(product)
        values = [*card["hard_constraints"], *card["soft_preferences"]]
        self.assertGreater(len(values), len(set(values)))
        for transform in session_axes.TEXT_NAMES:
            with self.subTest(text=transform):
                out = session_axes.constraints(
                    transform, values, product, random.Random("seed"))
                for first in range(len(values)):
                    for second in range(first + 1, len(values)):
                        if values[first] == values[second]:
                            self.assertEqual(out[first], out[second])

    def test_a_transform_never_merges_two_distinct_constraints(self) -> None:
        """The other half of the bijection, and the more damaging half.

        A transform drawing from a fixed phrase list can hand two different
        facts the same string. `customer_reply()` then finds the second already
        in `disclosed` and no probe can ever surface it, so the session holds
        fewer facts than its card claims and a wording column reads as an
        information column. Measured before the guard: `comparative` merged on
        56.5% of catalog cards and `implicit` on 25.3%.
        """
        products = catalog_rows()
        for transform in session_axes.TEXT_NAMES:
            merged = 0
            for product in products[:120]:
                card = local_evaluator.intent_card(product)
                values = [*card["hard_constraints"],
                          *card["soft_preferences"]]
                out = session_axes.constraints(
                    transform, values, product, random.Random("seed"))
                if len(set(out)) < len(set(values)):
                    merged += 1
            with self.subTest(text=transform):
                self.assertEqual(merged, 0)

    def test_a_text_transform_moves_the_card_not_the_pivot(self) -> None:
        plain = build(session_axes.Recipe(
            "i", 9, count=60, mix=(("intent_override", 1),),
            dialogue="early_pivot"))
        moved = build(session_axes.Recipe(
            "i", 9, count=60, mix=(("intent_override", 1),),
            dialogue="early_pivot", text="typo"))
        self.assertEqual(
            [row["behavior"]["override"]["turn"] for row in plain],
            [row["behavior"]["override"]["turn"] for row in moved],
        )
        self.assertNotEqual(
            [row["intent_card"] for row in plain],
            [row["intent_card"] for row in moved],
        )


class ValidationTest(unittest.TestCase):
    """`evaluate()` crashes outside its try block on several of these.

    `initial_message():161` indexes `behavior.override.old_value` directly and
    `int(override["turn"])` is uncaught, so a malformed row aborts a whole run
    rather than costing one session.
    """

    def setUp(self) -> None:
        self.catalog_ids = {row["parent_asin"] for row in catalog_rows()}
        self.good = build(
            session_axes.Recipe("v", 1, count=40, text="typo"))[0]

    def test_every_generated_row_validates(self) -> None:
        for recipe in session_sets.MANIFEST:
            with self.subTest(set=recipe.name):
                trimmed = session_axes.Recipe(
                    recipe.name, recipe.seed, count=40, mix=recipe.mix,
                    pool=recipe.pool, weights=recipe.weights, text=recipe.text,
                    profiles=recipe.profiles, dialogue=recipe.dialogue,
                    shoppers=recipe.shoppers,
                )
                for row in build(trimmed):
                    sessions.validate_row(row, self.catalog_ids)

    def test_malformed_rows_are_refused(self) -> None:
        override = {"turn": 3, "old_value": "a", "new_value": "b",
                    "message": "m"}
        card = {"target_category": "t", "hard_constraints": ["cotton"],
                "soft_preferences": ["airy"]}
        cases = (
            ("no sample_id", {**self.good, "sample_id": ""}),
            ("no ground_truth",
             {k: v for k, v in self.good.items() if k != "ground_truth"}),
            ("no user_profile",
             {k: v for k, v in self.good.items() if k != "user_profile"}),
            ("public id", {**self.good, "sample_id": "public_0001"}),
            ("unknown target",
             {**self.good, "ground_truth": {"parent_asin": "NOPE"}}),
            ("short profile",
             {**self.good, "user_profile": {"summary": "x"}}),
            ("extra profile key",
             {**self.good,
              "user_profile": {**self.good["user_profile"], "extra": 1}}),
            ("card without behavior",
             {k: v for k, v in self.good.items() if k != "behavior"}),
            ("behavior without card",
             {k: v for k, v in self.good.items() if k != "intent_card"}),
            ("override missing", {
                **self.good, "scenario_type": "intent_override",
                "intent_card": card,
                "behavior": {"scenario_type": "intent_override"}}),
            ("pivot too early", {
                **self.good, "scenario_type": "intent_override",
                "intent_card": card,
                "behavior": {"scenario_type": "intent_override",
                             "override": {**override, "turn": 1}}}),
            ("pivot too late", {
                **self.good, "scenario_type": "intent_override",
                "intent_card": card,
                "behavior": {"scenario_type": "intent_override",
                             "override": {**override, "turn": 11}}}),
            ("pivot not an int", {
                **self.good, "scenario_type": "intent_override",
                "intent_card": card,
                "behavior": {"scenario_type": "intent_override",
                             "override": {**override, "turn": "3"}}}),
            ("empty constraint", {
                **self.good,
                "intent_card": {**card, "hard_constraints": ["cotton", "  "]}}),
        )
        for label, row in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    sessions.validate_row(row, self.catalog_ids)


class DialogueTest(unittest.TestCase):
    """Every shape must survive the real dialogue loop, not just the format."""

    def _drive(self, recipe: session_axes.Recipe) -> dict:
        rows = build(recipe)
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(pathlib.Path(directory))
            catalog_ids, categories, by_asin = local_evaluator.catalog_index(
                path)
            for row in rows:
                sessions.validate_row(row, catalog_ids)
            agent = fixtures.ConstantAgent(
                [row["parent_asin"] for row in catalog_rows()[:10]])
            return sessions.measure(
                agent, rows, catalog_ids, categories, by_asin)

    def test_every_dialogue_shape_completes_without_an_exception(self) -> None:
        for shape in session_axes.DIALOGUE_NAMES:
            with self.subTest(dialogue=shape):
                result = self._drive(session_axes.Recipe(
                    "d", 3, count=24, dialogue=shape))
                self.assertEqual(result["health"]["agent_exceptions"], 0)
                self.assertEqual(result["health"]["discarded_responses"], 0)

    def test_every_text_transform_completes_without_an_exception(self) -> None:
        for transform in session_axes.TEXT_NAMES:
            with self.subTest(text=transform):
                result = self._drive(session_axes.Recipe(
                    "t", 3, count=24, text=transform))
                self.assertEqual(result["health"]["agent_exceptions"], 0)

    def test_a_silent_buying_row_gets_the_browsing_greeting(self) -> None:
        recipe = session_axes.Recipe("s", 3, count=20, mix=(("buying", 1),),
                                     dialogue="silent")
        rows = build(recipe)
        products = {row["parent_asin"]: row for row in catalog_rows()}
        profile = sessions.difficulty(
            rows, products, session_axes.survey(catalog_rows()))
        self.assertEqual(profile["silent_buying"], len(rows))


class ManifestTest(unittest.TestCase):
    """The manifest is frozen, so a drifting seed invalidates past readings."""

    def test_names_and_seeds_are_unique(self) -> None:
        names = [recipe.name for recipe in session_sets.MANIFEST]
        seeds = [recipe.seed for recipe in session_sets.MANIFEST]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_every_axis_value_is_exercised_by_some_set(self) -> None:
        axes = {
            "pool": (session_axes.POOL_NAMES,
                     {part for recipe in session_sets.MANIFEST
                      for part in recipe.pool.split("+")}),
            "weights": (session_axes.WEIGHT_NAMES,
                        {r.weights for r in session_sets.MANIFEST}),
            "text": (session_axes.TEXT_NAMES,
                     {r.text for r in session_sets.MANIFEST}),
            "profiles": (session_axes.PROFILE_NAMES,
                         {r.profiles for r in session_sets.MANIFEST}),
            "dialogue": (session_axes.DIALOGUE_NAMES,
                         {r.dialogue for r in session_sets.MANIFEST}),
            "shoppers": (session_axes.SHOPPER_NAMES,
                         {r.shoppers for r in session_sets.MANIFEST}),
        }
        for axis, (declared, used) in axes.items():
            with self.subTest(axis=axis):
                self.assertEqual(set(declared) - used, set())

    def test_mirror_is_neutral_on_every_axis(self) -> None:
        mirror = session_sets.MANIFEST[0]
        self.assertEqual(mirror.name, "mirror")
        self.assertEqual(
            (mirror.pool, mirror.weights, mirror.text, mirror.profiles,
             mirror.dialogue, mirror.shoppers, mirror.mix),
            ("any", "size-biased", "verbatim", "public", "default",
             "distinct", session_axes.SCENARIO_MIX),
        )
        self.assertFalse(mirror.is_authored)


class DifficultyTest(unittest.TestCase):
    """A set is only useful as an instrument if its levers actually bite."""

    def test_unpopular_targets_are_measurably_less_reviewed(self) -> None:
        products = {row["parent_asin"]: row for row in catalog_rows()}
        facts = session_axes.survey(catalog_rows())

        def reviews(**overrides) -> float:
            recipe = session_axes.Recipe("h", 5, count=80, **overrides)
            return sessions.difficulty(build(recipe), products,
                                       facts)["reviews"]

        self.assertLess(reviews(weights="uniform"), reviews())

    def test_thin_targets_carry_less_constraint_text(self) -> None:
        products = {row["parent_asin"]: row for row in catalog_rows()}
        facts = session_axes.survey(catalog_rows())

        def features(**overrides) -> float:
            recipe = session_axes.Recipe("h", 5, count=60, **overrides)
            return sessions.difficulty(build(recipe), products,
                                       facts)["features"]

        self.assertLess(features(pool="thin"), features())

    @unittest.skipUnless(REAL_CATALOG.exists(), "needs data/catalog.jsonl")
    def test_compound_hard_is_harder_than_the_public_set(self) -> None:
        products = sessions.load_products(REAL_CATALOG)
        facts = session_axes.survey(products)
        by_asin = {str(row["parent_asin"]): row for row in products}
        public = local_evaluator.load_jsonl(REPO_ROOT / "data"
                                            / "public_set.jsonl")
        public_reviews = statistics.median(
            float(by_asin[row["ground_truth"]["parent_asin"]]
                  .get("rating_number") or 0)
            for row in public
        )
        recipe = next(item for item in session_sets.MANIFEST
                      if item.name == "compound_hard")
        rows = sessions.generate(
            recipe, products, facts,
            [row["user_profile"] for row in public])
        profile = sessions.difficulty(rows, by_asin, facts)

        self.assertLess(profile["reviews"], public_reviews / 100)
        self.assertGreater(profile["bucket"], session_axes.CROWDED_BUCKET)
        self.assertLessEqual(profile["features"], session_axes.THIN_FEATURES)


if __name__ == "__main__":
    unittest.main()


class IdentityAxisTest(unittest.TestCase):
    """The sixth axis: who each row belongs to.

    It is the only axis that relates rows to each other, and the only
    instrument on which per-person memory is readable at all, because the
    organizer's harness never sends the same shopper twice (findings 3.33).
    """

    def _returning(self, **overrides) -> list[dict]:
        recipe = session_axes.Recipe(
            "r", 23, count=36, mix=(("boundary", 1),), shoppers="returning",
            **overrides,
        )
        return build(recipe)

    def test_the_neutral_axis_names_nobody(self) -> None:
        """Which is what leaves the twenty-two frozen sets untouched."""
        for row in build(session_axes.Recipe("n", 1, count=12)):
            self.assertNotIn("shopper_id", row)
            self.assertNotIn("visit", row)

    def test_the_neutral_axis_places_the_drawn_targets_unchanged(self) -> None:
        products = catalog_rows()
        facts = session_axes.survey(products)
        targets = products[:9]

        layout = session_axes.shoppers(
            "distinct", targets, facts, products, "size-biased",
            random.Random(1))

        self.assertEqual(list(layout.targets), targets)
        self.assertEqual(set(layout.ids), {None})

    def test_a_returning_row_carries_both_identity_keys(self) -> None:
        for row in self._returning():
            self.assertIsInstance(row["shopper_id"], str)
            self.assertGreaterEqual(row["visit"], 1)

    def test_visit_order_follows_row_order(self) -> None:
        """A memory must be written by an earlier row to be read by a later."""
        visits = [row["visit"] for row in self._returning()]

        self.assertEqual(visits, sorted(visits))

    def test_every_shopper_visits_once_per_block(self) -> None:
        seen: dict[str, list[int]] = {}
        for row in self._returning():
            seen.setdefault(row["shopper_id"], []).append(row["visit"])

        expected = list(range(1, session_axes.VISITS_PER_SHOPPER + 1))
        for shopper, visits in seen.items():
            with self.subTest(shopper=shopper):
                self.assertEqual(visits, expected)

    def test_a_shoppers_visits_stay_inside_one_bucket(self) -> None:
        """Memory is only worth carrying if the visits relate."""
        products = {row["parent_asin"]: row for row in catalog_rows()}
        facts = session_axes.survey(catalog_rows())
        seen: dict[str, set[str]] = {}
        for row in self._returning():
            target = products[row["ground_truth"]["parent_asin"]]
            seen.setdefault(row["shopper_id"], set()).add(
                facts.bucket[str(target["parent_asin"])])

        for shopper, buckets in seen.items():
            with self.subTest(shopper=shopper):
                self.assertEqual(len(buckets), 1)

    def test_a_shopper_never_revisits_its_own_target(self) -> None:
        """A repeat target hands the later visit a session already converted."""
        seen: dict[str, list[str]] = {}
        for row in self._returning():
            seen.setdefault(row["shopper_id"], []).append(
                row["ground_truth"]["parent_asin"])

        for shopper, targets in seen.items():
            with self.subTest(shopper=shopper):
                self.assertEqual(len(set(targets)), len(targets))

    def test_the_axis_never_moves_the_profiles(self) -> None:
        """The profile generator is set-level; an extra draw would shift it."""
        plain = session_axes.Recipe("r", 23, count=36, mix=(("boundary", 1),))
        named = session_axes.Recipe(
            "r", 23, count=36, mix=(("boundary", 1),), shoppers="returning")

        self.assertEqual(
            [row["user_profile"] for row in build(plain)],
            [row["user_profile"] for row in build(named)],
        )

    def test_a_shoppers_visits_share_index_parity(self) -> None:
        """So `run.split_samples` cannot deal one visit into each half."""
        rows = self._returning()
        positions: dict[str, set[int]] = {}
        for index, row in enumerate(rows):
            positions.setdefault(row["shopper_id"], set()).add(index % 2)

        for shopper, parities in positions.items():
            with self.subTest(shopper=shopper):
                self.assertEqual(len(parities), 1)

    def test_an_unknown_axis_value_is_refused(self) -> None:
        products = catalog_rows()
        facts = session_axes.survey(products)

        with self.assertRaises(ValueError):
            session_axes.shoppers(
                "nobody", products[:6], facts, products, "size-biased",
                random.Random(1))

    def test_a_half_named_row_is_refused(self) -> None:
        """A row naming a shopper but no visit reads as a first visit always."""
        catalog_ids = {row["parent_asin"] for row in catalog_rows()}
        row = self._returning()[0]

        for dropped in ("shopper_id", "visit"):
            with self.subTest(dropped=dropped):
                broken = {key: value for key, value in row.items()
                          if key != dropped}
                with self.assertRaises(ValueError):
                    sessions.validate_row(broken, catalog_ids)

    def test_a_visit_number_must_be_a_real_int(self) -> None:
        catalog_ids = {row["parent_asin"] for row in catalog_rows()}
        row = self._returning()[0]

        for visit in (True, 0, "1", 1.0):
            with self.subTest(visit=visit):
                with self.assertRaises(ValueError):
                    sessions.validate_row(
                        {**row, "visit": visit}, catalog_ids)


class ShopperRefusalTest(unittest.TestCase):
    """The axis refuses a set it cannot build correctly.

    Bucket locality is the one property the returning-shopper set exists to
    have: a shopper whose visits span two departments has nothing true to
    remember, so the visit blocks stop being comparable. Padding from another
    bucket would keep generation working and quietly destroy that, which is
    worse than a loud failure at generation time.
    """

    def _product(self, asin: str, bucket: str) -> dict:
        return {
            "parent_asin": asin,
            "categories": ["Clothing, Shoes & Jewelry", "Women", bucket],
            "rating_number": 10,
            "features": [],
            "title": "a thing",
        }

    def test_a_bucket_too_small_for_three_visits_is_refused(self) -> None:
        alone = [self._product("A", "Tiny")]
        rest = [self._product(f"B{number}", "Other") for number in range(9)]
        candidates = alone + rest
        facts = session_axes.survey(candidates)

        with self.assertRaises(ValueError):
            session_axes.shoppers(
                "returning", alone + rest[:8], facts, candidates,
                "size-biased", random.Random(1))

    def test_a_set_shorter_than_one_shopper_is_refused(self) -> None:
        """Previously an IndexError from an under-sized layout."""
        candidates = [
            self._product(f"B{number}", "Other") for number in range(9)
        ]
        facts = session_axes.survey(candidates)

        with self.assertRaises(ValueError):
            session_axes.shoppers(
                "returning", candidates[:2], facts, candidates,
                "size-biased", random.Random(1))

    def test_the_neutral_axis_accepts_a_set_of_any_size(self) -> None:
        """`distinct` names nobody, so no visit count has to divide it."""
        candidates = [
            self._product(f"B{number}", "Other") for number in range(9)
        ]
        facts = session_axes.survey(candidates)

        layout = session_axes.shoppers(
            "distinct", candidates[:2], facts, candidates, "size-biased",
            random.Random(1))

        self.assertEqual(len(layout.targets), 2)
