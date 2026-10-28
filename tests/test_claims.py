from __future__ import annotations

import pytest

from groundedgen import claims, utils


def values(text, attribute):
    return claims.extract_claims(text)[attribute]["values"]


# ---- longest match and word boundaries -----------------------------------------

@pytest.mark.parametrize("text, category", [
    ("A classic t-shirt for everyday wear.", "t-shirt"),
    ("A classic T-Shirt.", "t-shirt"),
    ("A relaxed t shirt.", "t-shirt"),
    ("A soft tshirt.", "t-shirt"),
    ("A cosy sweatshirt with a hood.", "sweatshirt"),
    ("A crisp shirt with a collar.", "shirt"),
    ("A zip-up hoodie.", "sweatshirt"),
    ("Slim chinos for work.", "trousers"),
    ("A woollen pullover.", "sweater"),
    ("A tailored blazer.", "jacket"),
    ("An elegant sari.", "saree"),
])
def test_longest_match_precedence(text, category):
    assert values(text, "category") == [category]


def test_t_shirt_never_also_counts_as_shirt():
    mentions = claims.find_mentions("This t-shirt and that sweatshirt.")
    assert [m.label for m in mentions if m.attribute == "category"] == ["t-shirt", "sweatshirt"]


def test_no_match_inside_other_words():
    text = "A tank top with a tangerine stripe, shirtless model, redesigned topology."
    found = claims.extract_claims(text)
    assert found["color"]["values"] == []          # "tan", "red" inside words
    assert found["category"]["values"] == ["top"]  # "tank top" counts as top


# ---- plurals and hyphenated forms --------------------------------------------------

@pytest.mark.parametrize("text, attribute, expected", [
    ("Two kurtas in a set.", "category", ["kurta"]),
    ("A pack of tees.", "category", ["t-shirt"]),
    ("Summer dresses for every day.", "category", ["dress"]),
    ("Comfortable kurtis.", "category", ["kurta"]),
    ("A navy-blue shirt.", "color", ["blue"]),
    ("An off-white top.", "color", ["white"]),
    ("A cotton-blend shirt.", "material", ["cotton"]),
    ("A multi-colored print.", "color", ["other"]),
])
def test_plurals_and_hyphens(text, attribute, expected):
    assert values(text, attribute) == expected


# ---- styling clauses ------------------------------------------------------------------

def test_styling_clause_mentions_are_ignored():
    text = "A red kurta in silk. Pair it with blue jeans and a cotton dupatta."
    found = claims.extract_claims(text)
    assert found["category"]["values"] == ["kurta"]
    assert found["color"]["values"] == ["red"]
    assert found["material"]["values"] == ["silk"]
    styled = [m for m in claims.find_mentions(text) if m.in_styling]
    assert {m.label for m in styled} == {"blue", "jeans", "cotton"}


def test_styling_clause_runs_to_end_of_sentence_only():
    text = "Wear it with white sneakers. The black jacket keeps you warm."
    found = claims.extract_claims(text)
    assert found["color"]["values"] == ["black"]
    assert found["category"]["values"] == ["jacket"]


@pytest.mark.parametrize("trigger", ["pair with", "team it with", "team with", "style it with",
                                     "wear with", "goes well with", "match it with",
                                     "complete the look"])
def test_every_styling_trigger(trigger):
    found = claims.extract_claims(f"A grey sweater. It {trigger} black trousers.")
    assert found["color"]["values"] == ["grey"]
    assert found["category"]["values"] == ["sweater"]


# ---- "top" and "dress" exclusions -------------------------------------------------------

@pytest.mark.parametrize("phrase", ["top quality", "top-quality", "top choice", "on top",
                                    "top-notch", "tops off"])
def test_top_exclusions(phrase):
    assert values(f"This is {phrase} fabric.", "category") == []


@pytest.mark.parametrize("phrase", ["dress up", "dress down", "dress code", "dressy"])
def test_dress_exclusions(phrase):
    assert values(f"Great to {phrase} for any day.", "category") == []


def test_exclusion_does_not_hide_a_real_mention():
    assert values("A top-quality top for summer.", "category") == ["top"]
    assert values("Dress up in this dress.", "category") == ["dress"]


# ---- category first mention, multi-value sets ---------------------------------------------

def test_category_is_first_mention():
    assert values("This kurta works like a dress.", "category") == ["kurta"]


def test_multi_value_sets():
    found = claims.extract_claims("A red and navy dress in cotton and polyester. Bright red trim.")
    assert found["color"]["values"] == ["red", "blue"]
    assert found["material"]["values"] == ["cotton", "synthetic"]
    assert len(found["color"]["spans"]) == 3  # red, navy, red


def test_off_taxonomy_color_is_other():
    assert values("A teal shirt with gold buttons.", "color") == ["other"]


def test_material_synonyms():
    assert values("Soft georgette and faux leather.", "material") == ["chiffon", "leather"]
    assert values("Knitted from wool.", "material") == ["knit"]


# ---- hedges --------------------------------------------------------------------------------

@pytest.mark.parametrize("cue", ["may vary", "might", "may be", "appears to", "looks like",
                                 "seems", "possibly", "likely", "or similar"])
def test_hedges(cue):
    found = claims.extract_claims(f"The color is blue but {cue} different. A cotton shirt.")
    assert found["color"]["hedged"]
    assert not found["material"]["hedged"]  # different sentence


def test_unhedged_statement():
    found = claims.extract_claims("A blue shirt.")
    assert found["color"] == {"stated": True, "values": ["blue"], "hedged": False, "spans": [[2, 6]]}
    assert found["material"] == {"stated": False, "values": [], "hedged": False, "spans": []}


def test_spans_index_original_text():
    text = "An Off-White KURTA."
    for attr, info in claims.extract_claims(text).items():
        for s, e in info["spans"]:
            assert text[s:e].lower().replace("-", " ") in ("off white", "kurta")


def test_sentence_split():
    assert claims.split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert claims.split_sentences("Price 2.5 lakhs. Next") == ["Price 2.5 lakhs.", "Next"]


def test_run_claims_writes_cached_file(tmp_path):
    gen = tmp_path / "outputs" / "generation" / "B" / "llm__x__ext.jsonl"
    utils.write_jsonl(gen, [
        {"item_id": "fpi_1", "condition": "B", "generator": "llm/x", "extractor": "ext",
         "mask_attr": None, "text": "A red kurta."},
    ])
    out = claims.run_claims(tmp_path, gen)
    assert out.name == "B__llm__x__ext.jsonl"
    [rec] = utils.read_jsonl(out)
    assert rec["claims"]["color"]["values"] == ["red"] and rec["item_id"] == "fpi_1"
    claims.run_claims(tmp_path, gen)
    assert len(utils.read_jsonl(out)) == 1  # cached
