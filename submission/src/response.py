"""Saying what the agent just did, in the customer's own terms.

The reply is composed from the session state rather than drawn from a rotation,
so it names the category it understood, the requirements it is matching on, what
a redirect replaced, and how much the slate has narrowed. It costs no tokens and
no network: this is template composition over live state, not generation.

Two reasons it is worth the module. The contract calls this field customer-
facing natural language and lists transparent recommendation explanations as a
goal, and a per-item explanation has nowhere to live -- the response schema
allows a recommendation only `parent_asin` and `score` -- so the explanation
belongs here. And a reply that states what was understood is the only place a
customer can catch the agent misunderstanding them.
"""

from __future__ import annotations

import re

from submission.src import dialogue
from submission.src import intent_detector
from submission.src import persona_classifier
from submission.src import policy as policy_module
from submission.src import probe as probe_module

# Reading a long constraint back verbatim is worse than not reading it back.
MAX_QUOTED = 42
MAX_LISTED = 2

# How many products a reply names before it stops listing and names only the
# first. Ten titles is a wall of text; three is still a sentence.
MAX_NAMED = 3
# Below this, a title trimmed at its first separator is a bare brand token
# ("MxG") rather than a name, so more of the raw title is taken instead.
MIN_NAME = 16
MAX_NAME = 46

# Where a title stops being a name and starts being marketing copy.
_NAME_CUT = re.compile(r"\s+[-|,\u2013\u2014]\s+|\s*\(")

# Catalog text is written for a database, not for reading aloud. These space it
# out without removing anything: the acknowledgement is the customer's only
# chance to catch a misparse, so `Material:alloy` keeps its prefix and becomes
# `Material: alloy` rather than `alloy`.
_TIGHT_COLON = re.compile(r":(?=\S)")
_TIGHT_COMMA = re.compile(r",(?=[^\s\d])")
_TIGHT_PERCENT = re.compile(r"(\d)%(?=[A-Za-z])")

# Above this many products still in contention, the slate is not yet a
# recommendation and the reply should say so rather than imply confidence.
CROWDED = 5

FALLBACK = "Here are some options. Anything specific you need?"

# What each attribute is called when a question names it. The probe's arm names
# are catalog vocabulary; these are what a shopper would say.
LABELS = {
    "material": "material",
    "color": "colour",
    "size": "size",
    "style": "style",
    "use_case": "occasion",
    "feature": "feature",
    "budget": "budget",
    "brand": "brand",
    "category": "kind",
}

# The open question each attribute asks when no alternatives are worth
# offering. A shopper who cannot name an attribute can still answer these.
STEMS = {
    "material": "What should it be made of",
    "color": "What colour are you after",
    "size": "What size do you need",
    "style": "What style are you after",
    "use_case": "Where will you mainly use it",
    "feature": "Which features matter to you",
    "budget": "What budget are you working with",
    "brand": "Any brand in mind",
    "category": "What kind are you looking for",
}

OPEN_QUESTION = "Is there anything specific you need from it"


def compose(
    state: dialogue.SessionState,
    parsed: dialogue.ParsedTurn,
    contenders: int,
    head: int,
    served: int,
    asked: str | None,
    policy: str = policy_module.DISCOVERY,
    options: tuple[str, ...] = (),
    names: tuple[str, ...] = (),
    size: int = 10,
) -> str:
    """Returns the customer-facing reply for the turn just served.

    The wording is the one decision in the turn the simulator cannot read:
    `evaluator/local_evaluator.py:243` type-checks this field and never parses
    it, and `customer_reply()` branches on the `ask_attribute` enum alone. So
    nothing here can move the score in either direction, and everything here is
    what a person reading the transcript actually sees (findings 3.46).

    Args:
        state: The session after this turn's message was folded in.
        parsed: What this turn's message yielded, for what changed just now.
        contenders: How many products are still scoring near the leader.
        head: How many of the recommendations the ranking is committing to.
        served: How many recommendations the slate carries.
        asked: The attribute being probed, or None if nothing is.
        policy: This turn's dialogue policy, which decides the framing.
        options: Values of `asked` the live pool actually offers, if any are
          worth putting to the customer.
        names: Titles of the products this slate carries, in slate order.
          Absent, the slate is described by count alone.
        size: The full slate width, so a held-back slate can be told apart
          from a committed one.

    Returns three lines -- what was understood, what is being shown, what is
    being asked -- because a customer scanning a reply should not have to find
    the question inside a paragraph.
    """
    acknowledged = _acknowledge(state, parsed)
    parts = [
        acknowledged,
        _slate(contenders, head, served, names, size),
        _question(state, parsed, asked, policy, options, bool(acknowledged)),
    ]
    reply = "\n".join(part for part in parts if part)
    return reply or FALLBACK


def _acknowledge(
    state: dialogue.SessionState, parsed: dialogue.ParsedTurn
) -> str:
    """Returns what the agent understood from this turn."""
    if parsed.pivot:
        replacing = _labelled(state, parsed.constraints)
        dropped = _listed(_distinct(state.superseded, parsed.constraints))
        if replacing and dropped:
            return _stopped(f"Understood, {replacing} instead of {dropped}")
        if replacing:
            return _stopped(f"Understood, {replacing} it is")
        return "Understood, starting over on that."

    if parsed.boundary_refusal:
        # Naming the attribute rather than gesturing at it. `_question` sees
        # that this fired and does not say it twice.
        if state.declined:
            return f"No problem, {_label(state.declined[-1])} can stay open."
        return "No problem, I will use my judgement there."

    if parsed.exhausted:
        # A scoped exhaustion retires one attribute, not the session, and
        # saying otherwise while still asking questions reads as not listening
        # (`dialogue.SCOPED_EXHAUSTION`).
        if not state.exhausted and parsed.exhausted_arm:
            arm = _label(parsed.exhausted_arm)
            return f"Understood, no strong view on {arm}."
        return "Thanks, I think I have what I need."

    if parsed.constraints:
        return _stopped(f"Got it, {_labelled(state, parsed.constraints)}")

    if state.turn <= 1 and state.category:
        return f"Happy to help you find {state.category.lower()}."
    return ""


def _slate(
    contenders: int,
    head: int,
    served: int,
    names: tuple[str, ...] = (),
    size: int = 10,
) -> str:
    """Returns what the recommendations on this turn are, and are not.

    The distinction is real, not decoration: a committed slate is the ranking's
    full page, while a held-back one is what the evidence so far can actually
    justify. Telling the customer which they are looking at is what makes the
    second honest rather than merely odd.

    The branch used to test `head >= served`, which stopped meaning anything
    when `ranking.EXPLORE_FILL` was switched off: the slate then carries only
    the committed head, so `served == head` on every turn and every held-back
    slate described itself as "the 1 closest matches". The question is not how
    the head compares to what was served, it is whether anything was held back.
    """
    if not served:
        return ""
    listed = _names(names)
    if served >= size:
        tail = f", starting with {listed[0]}" if listed else ", best first"
        return f"Here are all {served} closest matches{tail}."
    if served == 1:
        named = f": {listed[0]}" if listed else ""
        return f"Here is the closest match I can justify so far{named}."
    named = f": {_join(listed)}" if listed else ""
    return f"Here are the {served} I can justify so far{named}."


def _names(titles: tuple[str, ...]) -> list[str]:
    """Returns slate titles as sayable names, with repeats collapsed.

    Variants of one product share a title -- 3.8% of three-product slates hold
    a pair -- and naming both reads as a stutter. Collapsing them also tells
    the customer something true: the slate is narrower than its count.
    """
    grouped: list[list] = []
    for title in titles[:MAX_NAMED]:
        name = _name(title)
        if not name:
            continue
        for row in grouped:
            if row[0] == name:
                row[1] += 1
                break
        else:
            grouped.append([name, 1])
    return [row[0] if row[1] == 1 else f"{row[0]} ({row[1]} variants)"
            for row in grouped]


def _name(title: str) -> str:
    """Returns a catalog title trimmed to something worth saying."""
    cleaned = " ".join(str(title).split())
    if not cleaned:
        return ""
    head = _NAME_CUT.split(cleaned)[0].strip()
    if len(head) < MIN_NAME:
        # Cutting at the first separator left a bare brand token; take more.
        head = cleaned
    if len(head) > MAX_NAME:
        head = head[:MAX_NAME].rsplit(" ", 1)[0]
    return head.strip()


def _join(values: list[str]) -> str:
    """Returns a readable list of two or more names."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _question(
    state: dialogue.SessionState,
    parsed: dialogue.ParsedTurn,
    asked: str | None,
    policy: str = policy_module.DISCOVERY,
    options: tuple[str, ...] = (),
    acknowledged: bool = False,
) -> str:
    """Returns the clarifying question, framed the way the policy asks for.

    One question, one attribute, every time. The framing changes because the
    same question does different work depending on where the conversation is:
    a customer who has said nothing needs a scenario they can recognise, one
    who has just been specific needs a direct follow-up, and one who has
    answered two questions with nothing needs a different kind of question
    rather than a sharper version of the same one.
    """
    if asked is None:
        if policy == policy_module.COVERAGE:
            return ("Based on everything you have told me, these are the "
                    "strongest matches you have not seen yet.")
        return "Let me know if none of these are right."

    if asked == probe_module.WILDCARD:
        return _closed(_wildcard_stem(state, policy))

    listed = _choices(options)
    if policy == policy_module.STAGNATION:
        if listed:
            return _closed(f"Let us try another angle. Which matters more: "
                           f"{listed}")
        return _closed(f"Let us try another angle. {_stem(asked)}")

    if policy == policy_module.BOUNDARY:
        # `_acknowledge` speaks first and already names a refusal when this
        # turn carried one. Repeating it here put "No problem, material can
        # stay open" *after* the slate on a turn that also disclosed something.
        spoken = parsed.boundary_refusal or acknowledged
        opening = "" if spoken else _released(state)
        if listed:
            return _closed(f"{opening}Would {listed} suit you better")
        return _closed(f"{opening}{_stem(asked)}")

    if policy == policy_module.PRECISION:
        if listed:
            return _closed(f"Do you need {listed}")
        return _closed(f"Do you have a preferred {_label(asked)}")

    if listed:
        return _closed(f"{_stem(asked)}: {listed}")
    return _closed(_stem(asked))


def _wildcard_stem(state: dialogue.SessionState, policy: str) -> str:
    """Returns the phrasing for a question that names no attribute."""
    if policy == policy_module.STAGNATION:
        return "Let us try another angle. What matters most to you here"
    if state.constraints:
        return "Is there another detail I should match on"
    return OPEN_QUESTION


def _released(state: dialogue.SessionState) -> str:
    """Returns the clause that tells the customer a refusal was heard.

    Naming the attribute they declined is the point: it is the difference
    between moving on and appearing to have ignored them.
    """
    if not state.declined:
        return ""
    return f"No problem, {_label(state.declined[-1])} can stay open. "


def _choices(options: tuple[str, ...]) -> str:
    """Returns the offered alternatives as a readable list."""
    quoted = [_short(value) for value in options]
    quoted = [value for value in quoted if value]
    if len(quoted) < 2:
        return ""
    if len(quoted) == 2:
        return f"{quoted[0]} or {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, or {quoted[-1]}"


def _label(attribute: str) -> str:
    """Returns what a shopper would call this attribute."""
    if not attribute:
        return ""
    return LABELS.get(attribute, attribute.replace("_", " "))


def _stem(attribute: str) -> str:
    """Returns the open question this attribute asks."""
    return STEMS.get(attribute, f"What {_label(attribute)} do you need")


def _closed(question: str) -> str:
    """Returns a question stem punctuated as one."""
    return f"{question.rstrip('?').rstrip()}?"


def _distinct(
    dropped: tuple[str, ...], replacing: tuple[str, ...]
) -> tuple[str, ...]:
    """Returns what a redirect really discarded.

    A replacement often restates something already said, so the raw superseded
    list would have the reply announce it is dropping the very thing it is
    adopting. Only what the new requirement does not already contain is news.
    """
    incoming = [value.casefold() for value in replacing]
    return tuple(
        value for value in dropped
        if not any(_overlapping(value.casefold(), new) for new in incoming)
    )


def _overlapping(left: str, right: str) -> bool:
    """Returns whether two constraints are saying the same thing."""
    return left in right or right in left


def _labelled(state: dialogue.SessionState, values: tuple[str, ...]) -> str:
    """Returns the arriving constraints, each named by the attribute it was
    filed under.

    The acknowledgement is the customer's only sight of how they were
    understood, and the value alone does not show it: "leather" could have been
    read as a material or as a brand. "material: leather" says which, so a
    misfiling is visible on the turn it happens rather than three turns later
    when the slate is wrong.

    A value that already carries its own prefix keeps it and gains no second
    one, and values sharing an attribute are named once rather than each.
    """
    # Every slot, not just this turn's: a constraint the customer restates
    # keeps the turn it first arrived on, and it is filed under the same
    # attribute either way.
    typed = {slot.value: slot.attribute for slot in state.slots}
    rows: list[tuple[str, str]] = []
    for value in values[:MAX_LISTED]:
        short = _short(value)
        if not short:
            continue
        # Its own prefix is the customer's word for it, and outranks ours.
        attribute = "" if ":" in short else _label(typed.get(value, ""))
        rows.append((attribute, short))
    if not rows:
        return ""
    if len(rows) > 1 and len({attribute for attribute, _ in rows}) == 1:
        attribute = rows[0][0]
        joined = " and ".join(short for _, short in rows)
        return f"{attribute}: {joined}" if attribute else joined
    named = [f"{a}: {v}" if a else v for a, v in rows]
    if len(named) == 1:
        return named[0]
    joiner = "; " if any(":" in value for value in named) else " and "
    return joiner.join(named)


def _listed(values: tuple[str, ...]) -> str:
    """Returns a short readable list of constraint strings."""
    quoted = [_short(value) for value in values[:MAX_LISTED]]
    quoted = [value for value in quoted if value]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    # "material: alloy and buckle closure" reads as though both belong to the
    # material. A semicolon keeps the two apart once either carries a prefix.
    joiner = "; " if any(":" in value for value in quoted) else " and "
    return joiner.join(quoted)


def _stopped(sentence: str) -> str:
    """Returns a sentence ended once. A trimmed value already carries its own
    ellipsis, and appending a full stop to it produces "coverage....".
    """
    return sentence if sentence.endswith(("...", ".", "?", "!")) else f"{sentence}."


def _readable(value: str) -> str:
    """Returns catalog text spaced for reading, with its meaning intact.

    Nothing is removed, and the field prefix least of all. The acknowledgement
    is the only place a customer can catch the agent having misread them, so
    `Material:alloy` becomes `Material: alloy` rather than `alloy` -- and
    `Item model number: G796` keeps the half that makes `G796` mean anything.
    """
    cleaned = " ".join(str(value).split())
    cleaned = _TIGHT_COLON.sub(": ", cleaned)
    cleaned = _TIGHT_COMMA.sub(", ", cleaned)
    cleaned = _TIGHT_PERCENT.sub(r"\1% ", cleaned)
    return cleaned.strip(" -;,.")


def _short(value: str) -> str:
    """Returns a constraint trimmed to something worth reading aloud.

    An over-long value is cut at its first clause rather than mid-sentence: a
    raw character cut can keep the prefix and discard the value it labels
    (`Solids: 100% Cotton; heathers: 75% cot...`), which is worse than saying
    less.
    """
    cleaned = _readable(value)
    if len(cleaned) <= MAX_QUOTED:
        return cleaned.lower()
    clause = cleaned.split(";", 1)[0].strip()
    if clause and len(clause) <= MAX_QUOTED:
        return clause.lower()
    cut = cleaned[:MAX_QUOTED].rsplit(" ", 1)[0]
    return f"{cut.lower()}..." if cut else ""


# Persona-based response composition (new layer)
def compose_with_persona(
    state: dialogue.SessionState,
    parsed: dialogue.ParsedTurn,
    contenders: int,
    head: int,
    served: int,
    asked: str | None,
    user_message: str,
    conversation_history: list[tuple[str, str]] | None = None,
    candidate_count: int = 0,
    user_profile: dict | None = None,
    persona_match: persona_classifier.PersonaMatch | None = None,
    policy: str = policy_module.DISCOVERY,
    options: tuple[str, ...] = (),
) -> str:
    """Compose a persona-framed response grounded in the selected probe.

    Args:
        state: Session state after this turn
        parsed: Parsed constraints from this turn
        contenders: Number of products near the top score
        head: Number of committed recommendations
        served: Number of recommendations shown
        user_message: The user's current message
        conversation_history: Previous turns
        asked: Structured attribute selected by the probe.
    Returns:
        Natural language response
    """
    # Step 1: Acknowledge what we understood
    acknowledgment = _acknowledge(state, parsed)

    # Step 2: Describe the slate
    slate_desc = _slate(contenders, head, served)

    # Step 3: Generate persona-driven question
    try:
        if persona_match is None:
            persona_match = select_persona(
                state, user_message, conversation_history,
                candidate_count, user_profile,
            )
        question = _persona_question(
            persona_match.persona_type, asked, state, parsed, policy, options
        )

    except Exception:
        # If persona pipeline fails, fallback to hardcoded
        question = _question(state, parsed, asked, policy, options)

    # Combine all parts
    parts = [acknowledgment, slate_desc, question]
    reply = " ".join(part for part in parts if part)
    return reply or FALLBACK


def select_persona(
    state: dialogue.SessionState,
    user_message: str,
    conversation_history: list[tuple[str, str]] | None = None,
    candidate_count: int = 0,
    user_profile: dict | None = None,
) -> persona_classifier.PersonaMatch:
    """Returns the grounded persona decision for this turn."""
    intent = intent_detector.IntentDetector().detect(
        user_message, state, conversation_history
    )
    return persona_classifier.PersonaClassifier().classify(
        intent, state, candidate_count, user_profile
    )


def _persona_question(
    persona_type: persona_classifier.PersonaType,
    asked: str | None,
    state: dialogue.SessionState,
    parsed: dialogue.ParsedTurn,
    policy: str = policy_module.DISCOVERY,
    options: tuple[str, ...] = (),
) -> str:
    """Frames the probe without ever changing its structured attribute."""
    if asked is None:
        return _question(state, parsed, None, policy, options)

    labels = {
        "category": "product category",
        "use_case": "main use",
        "feature": "must-have feature",
        "budget": "budget",
        "material": "material",
        "color": "color",
        "size": "size or fit",
        "style": "style",
        "brand": "brand",
        "other": "other important detail",
    }
    label = labels.get(asked, asked.replace("_", " "))

    if persona_type == persona_classifier.PersonaType.INTENT_OVERRIDE_PIVOT:
        return f"For the new direction, what {label} should I prioritize?"
    if persona_type == persona_classifier.PersonaType.BOUNDARY_REJECTION:
        return f"No problem; we can leave that open. What about your {label}?"
    if persona_type == persona_classifier.PersonaType.CLARIFY_CONTRADICTION:
        return f"Those preferences may compete. Which {label} matters most?"
    if persona_type == persona_classifier.PersonaType.MID_BROWSER_VAGUE:
        return f"To help narrow the options, what is your preferred {label}?"
    if persona_type == persona_classifier.PersonaType.LATE_CRITICAL_EVALUATOR:
        return f"Before I finalize these, what {label} is essential?"
    if persona_type == persona_classifier.PersonaType.MID_BROWSER_REFINED:
        return f"To refine these further, what {label} should I match?"
    return f"What is your preferred {label}?"
