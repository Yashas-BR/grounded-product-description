"""APG renderer: the section 9 property, article fixing and fallbacks."""

from __future__ import annotations

import random
import re

import pytest

from groundedgen import apg, claims
from groundedgen import taxonomy as tx

U = tx.UNKNOWN
KURTA_RED_SILK = {"category": "kurta", "color": "red", "material": "silk"}
SHIRT_BLUE_UNK = {"category": "shirt", "color": "blue", "material": U}
UNK_RED_UNK = {"category": U, "color": "red", "material": U}
TOP_ORANGE_KNIT = {"category": "top", "color": "orange", "material": "knit"}
NOTHING = {"category": U, "color": U, "material": U}
ONLY_MATERIAL = {"category": U, "color": U, "material": "denim"}
TSHIRT_GREY_SYN = {"category": "t-shirt", "color": "grey", "material": "synthetic"}


def assert_property(text: str, evidence) -> None:
    """Section 9 guarantee, checked with every counted mention."""
    assert not re.search(r"[{}]", text), f"raw placeholder or brace in {text!r}"
    for m in claims.find_mentions(text):
        if m.in_styling:
            continue
        assert evidence[m.attribute] != U, f"states unknown {m.attribute} in {text!r}"
        assert m.label == evidence[m.attribute], f"{m.attribute}={m.label} vs {evidence} in {text!r}"
        assert not m.hedged, f"hedged {m.attribute} in {text!r}"
    found = claims.extract_claims(text)
    for attribute in tx.ATTRIBUTES:
        for value in found[attribute]["values"]:
            assert value == evidence[attribute]


# 25 hand-written adversarial outputs: (raw LLM output, evidence)
ADVERSARIAL = [
    ("This {category} is made of pure cotton.", SHIRT_BLUE_UNK),             # unslotted material
    ("A {category} in {material}. Comfortable all day.", SHIRT_BLUE_UNK),    # slot for unknown
    ("A {colour} {category} for summer.", KURTA_RED_SILK),                   # misspelt slot
    ("A {category} in red.", SHIRT_BLUE_UNK),                                # wrong color word
    ("This {category} might be {color}.", SHIRT_BLUE_UNK),                   # hedge
    ("This {Category} is lovely. A {color} piece.", KURTA_RED_SILK),         # wrong-case slot
    ("A {category in {color}.", KURTA_RED_SILK),                             # unclosed brace
    ("A {color} kurta for festive days.", SHIRT_BLUE_UNK),                   # wrong garment word
    ("Crafted in silk, this {category} drapes well.", SHIRT_BLUE_UNK),       # guessed fabric
    ("Available in navy. A {category} for work.", KURTA_RED_SILK),           # navy -> blue != red
    ("This {category} is perfect. Pair it with blue jeans.", KURTA_RED_SILK),  # styling clause
    ("", KURTA_RED_SILK),                                                    # empty output
    ("{category} {category} {category}.", TOP_ORANGE_KNIT),                  # repeated slot
    ("{category}{color}{material}", TOP_ORANGE_KNIT),                        # glued slots
    ("A teal {category} with charm.", KURTA_RED_SILK),                       # off-taxonomy color
    ("It looks like {material}. A {category}.", KURTA_RED_SILK),             # hedge cue
    ("A {color} and white {category}.", KURTA_RED_SILK),                     # extra color
    ('Product attributes: {"category": "kurta"}. A {category}.', KURTA_RED_SILK),  # JSON leak
    ("This t-shirt is great.", {"category": "top", "color": U, "material": U}),     # wrong garment
    ("A {color} {category} in {material}.", NOTHING),                        # all unknown
    ("A stylish piece for any day.", KURTA_RED_SILK),                        # category never stated
    ("**{category}** in {color}!", KURTA_RED_SILK),                          # markdown around slot
    ("A {color} {category}. Possibly cotton.", TSHIRT_GREY_SYN),             # hedged guess
    ("A {category} in polyester and spandex.", TSHIRT_GREY_SYN),             # unslotted but consistent
    ("This {category} in {color} {material} feels luxurious. The silk sheen of a sari.", KURTA_RED_SILK),
    ("A {color} {category}. Made of denim. Great for the weekend.", UNK_RED_UNK),
    ("A {material} piece with {color} accents.", ONLY_MATERIAL),
    ("Comes in a soft {color}} shade. A {category}.", KURTA_RED_SILK),       # stray brace
]


@pytest.mark.parametrize("raw, evidence", ADVERSARIAL)
def test_property_on_adversarial_outputs(raw, evidence):
    result = apg.render(raw, evidence)
    assert_property(result.text, evidence)
    if evidence["category"] != U:
        assert claims.extract_claims(result.text)["category"]["values"] == [evidence["category"]]


def test_at_least_twenty_adversarial_cases():
    assert len(ADVERSARIAL) >= 20


def test_property_on_random_outputs():
    """Seeded fuzz: random mixes of slots, lexicon words, hedges and braces."""
    rng = random.Random(42)
    words = (["{category}", "{color}", "{material}", "{colour}", "{", "}", "{x}",
              "this", "piece", "is", "a", "an", "in", "with", "and", "soft", "lovely",
              "pair it with", "might", "looks like", "top quality", "dress up"]
             + list(tx.lexicon("category")) + list(tx.lexicon("color"))
             + list(tx.lexicon("material")))
    evidences = [KURTA_RED_SILK, SHIRT_BLUE_UNK, UNK_RED_UNK, TOP_ORANGE_KNIT, NOTHING,
                 ONLY_MATERIAL, TSHIRT_GREY_SYN]
    for _ in range(3000):
        n_sent = rng.randint(0, 4)
        raw = " ".join(
            " ".join(rng.choice(words) for _ in range(rng.randint(1, 9))) + rng.choice(".!?")
            for _ in range(n_sent)
        )
        evidence = rng.choice(evidences)
        result = apg.render(raw, evidence)
        assert_property(result.text, evidence)


# ---- filling ------------------------------------------------------------------------

def test_fill_surface_forms_articles_and_capitals():
    out = apg.render("{category} for every day. this is a {color} {category} in {material}.",
                     TOP_ORANGE_KNIT)
    assert out.text == "Top for every day. This is an orange top in knit fabric."
    assert not out.log["fallback_used"]


def test_article_fixed_from_an_to_a():
    out = apg.render("An {category} in {color}.", SHIRT_BLUE_UNK)
    assert out.text == "A shirt in blue."


def test_plural_garments_take_a_pair_of():
    out = apg.render("Meet a {category} in {color}.", {"category": "trousers", "color": "grey", "material": U})
    assert out.text == "Meet a pair of trousers in grey."


def test_material_and_category_surface_forms():
    out = apg.render("A {color} {category} made of {material}.", TSHIRT_GREY_SYN)
    assert out.text == "A grey t-shirt made of synthetic fabric."


def test_styling_mentions_survive_but_are_not_claims():
    out = apg.render("A {color} {category}. Pair it with blue jeans.", KURTA_RED_SILK)
    assert out.text == "A red kurta. Pair it with blue jeans."


# ---- pruning and verification are logged -------------------------------------------------

def test_log_counts():
    raw = "A {color} {category}. Made in {material}. Pure cotton comfort. {oops} here."
    out = apg.render(raw, SHIRT_BLUE_UNK)
    assert out.text == "A blue shirt."
    assert out.log == {
        "raw_with_slots": raw,
        "placeholders_offered": ["category", "color"],
        "placeholders_used": ["category", "color"],
        "sentences_total": 4,
        "sentences_dropped_prune": 2,   # {material} unknown, {oops}
        "sentences_dropped_verify": 1,  # "Pure cotton"
        "fallback_used": False,
    }


# ---- fallbacks -------------------------------------------------------------------------

@pytest.mark.parametrize("evidence, expected", [
    (KURTA_RED_SILK, "A red kurta in silk."),
    (SHIRT_BLUE_UNK, "A blue shirt."),
    ({"category": "jeans", "color": U, "material": "denim"}, "Jeans in denim."),
    ({"category": "shorts", "color": "black", "material": U}, "Black shorts."),
    ({"category": "dress", "color": U, "material": U}, "A dress."),
    (TOP_ORANGE_KNIT, "An orange top in knit fabric."),
    (UNK_RED_UNK, "This piece comes in red."),
    (ONLY_MATERIAL, "This piece is made of denim."),
    (NOTHING, "Product details are available on request."),
])
def test_fallback_templates(evidence, expected):
    assert apg.fallback_text(evidence) == expected
    out = apg.render("", evidence)
    assert out.text == expected and out.log["fallback_used"]
    assert_property(out.text, evidence)


def test_fallback_when_category_known_but_unstated():
    out = apg.render("A lovely {color} piece for summer.", KURTA_RED_SILK)
    assert out.log["fallback_used"] and out.text == "A red kurta in silk."


def test_no_fallback_when_category_unknown_and_text_survives():
    out = apg.render("This piece comes in a lovely {color}.", UNK_RED_UNK)
    assert out.text == "This piece comes in a lovely red." and not out.log["fallback_used"]
