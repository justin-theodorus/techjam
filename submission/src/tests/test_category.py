from __future__ import annotations

import unittest

from submission.src import category

KEYS = (
    "Shirts T-Shirts",
    "Shirts Polos",
    "Tops Blouses",
    "Socks & Hosiery Leg Warmers",
    "Novelty Clothing",
)
RESOLVER = category.build(KEYS)


class ContainsTest(unittest.TestCase):
    def test_a_real_bucket_key_is_recognized(self) -> None:
        self.assertTrue(RESOLVER.contains("Shirts T-Shirts"))

    def test_an_unknown_key_is_not(self) -> None:
        self.assertFalse(RESOLVER.contains("Shirts T-Shirts but exploring"))

    def test_none_is_not_a_key(self) -> None:
        self.assertFalse(RESOLVER.contains(None))


class NamedCategoryTest(unittest.TestCase):
    """A stated category name resolves to itself, whatever frames it."""

    def test_the_shipped_template_resolves(self) -> None:
        self.assertEqual(
            RESOLVER.buckets(
                "I'm looking for Tops Blouses, but I'm still exploring."
            ),
            ("Tops Blouses",),
        )

    def test_a_reworded_frame_resolves_identically(self) -> None:
        self.assertEqual(
            RESOLVER.buckets(
                "Show me some Tops Blouses, just browsing for now."
            ),
            ("Tops Blouses",),
        )

    def test_padding_around_the_frame_does_not_matter(self) -> None:
        self.assertEqual(
            RESOLVER.buckets("Hi there. I want Tops Blouses please. Thanks!"),
            ("Tops Blouses",),
        )

    def test_missing_punctuation_does_not_matter(self) -> None:
        self.assertEqual(
            RESOLVER.buckets(
                "I'm looking for Tops Blouses but I'm still exploring"
            ),
            ("Tops Blouses",),
        )

    def test_the_longest_stated_name_wins(self) -> None:
        """`Shirts` alone must not beat the full name the customer said."""
        self.assertEqual(
            RESOLVER.buckets("I need Shirts T-Shirts today"),
            ("Shirts T-Shirts",),
        )


class SoftMatchTest(unittest.TestCase):
    def test_a_partial_name_unions_near_ties_instead_of_guessing(self) -> None:
        resolver = category.build(
            ("Running Shoes", "Walking Shoes", "Dress Shoes")
        )
        buckets = resolver.buckets("do you have any shoes")

        self.assertGreater(len(buckets), 1)
        self.assertIn("Running Shoes", buckets)

    def test_a_name_whose_words_all_repeat_is_covered_by_one_of_them(
        self,
    ) -> None:
        """`Shirts T-Shirts` tokenizes to a single distinct word.

        The tokenizer drops single characters, so `T-Shirts` contributes only
        `shirts`. Saying `shirts` therefore covers that bucket's whole name.
        Recorded because it is surprising, and because it makes the outright
        naming rule, not coverage, the thing that separates the shirt buckets.
        """
        self.assertEqual(RESOLVER.buckets("any shirts?"), ("Shirts T-Shirts",))
        self.assertEqual(
            RESOLVER.buckets("any Shirts Polos?"), ("Shirts Polos",)
        )

    def test_a_soft_match_is_capped(self) -> None:
        self.assertLessEqual(
            len(RESOLVER.buckets("shirts tops socks clothing")),
            category.MAX_BUCKETS,
        )

    def test_nothing_recognizable_resolves_to_nothing(self) -> None:
        self.assertEqual(RESOLVER.buckets("hello, how are you today"), ())

    def test_an_empty_message_resolves_to_nothing(self) -> None:
        self.assertEqual(RESOLVER.buckets(""), ())


class ScoringTest(unittest.TestCase):
    def test_full_coverage_outranks_partial(self) -> None:
        matches = RESOLVER.resolve(["shirts", "polos"])

        self.assertEqual(matches[0].key, "Shirts Polos")
        self.assertEqual(matches[0].coverage, 1.0)

    def test_a_rare_word_carries_more_weight_than_a_common_one(self) -> None:
        """`shirts` spans two buckets, so it covers less of either name."""
        common = RESOLVER.resolve(["shirts"])
        rare = RESOLVER.resolve(["polos"])
        by_key = {match.key: match.coverage for match in common}

        self.assertLess(
            by_key.get("Shirts Polos", 0.0),
            next(m.coverage for m in rare if m.key == "Shirts Polos"),
        )

    def test_an_empty_key_set_builds_without_raising(self) -> None:
        self.assertEqual(category.build(()).buckets("anything"), ())


if __name__ == "__main__":
    unittest.main()
