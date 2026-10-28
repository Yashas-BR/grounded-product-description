from __future__ import annotations

import re

import pytest

from groundedgen import taxonomy as tx


def test_label_set_sizes():
    assert len(tx.CATEGORIES) == 13
    assert len(tx.COLORS) == 12
    assert len(tx.MATERIALS) == 8
    for labels in tx.LABELS.values():
        assert len(set(labels)) == len(labels)
        assert not {tx.UNKNOWN, tx.EXCLUDE, tx.OTHER} & set(labels)


def test_every_category_mapping_lands_in_label_set():
    assert set(tx.CATEGORY_RAW_TO_LABEL.values()) == set(tx.CATEGORIES)
    assert tx.map_category("Kurtis") == "kurta"
    assert tx.map_category(" Tshirts ") == "t-shirt"
    assert tx.map_category("Handbags") is None
    assert tx.map_category(None) is None


def test_every_color_mapping_lands_in_label_set():
    assert set(tx.COLOR_RAW_TO_LABEL.values()) == set(tx.COLORS)
    assert tx.map_color("Navy Blue") == "blue"
    assert tx.map_color("navy  BLUE") == "blue"  # case- and space-insensitive
    assert tx.map_color("Off White") == "white"
    assert tx.map_color("Fluorescent Green") == "green"
    assert tx.map_color("Coffee Brown") == "brown"


@pytest.mark.parametrize("raw", [
    "Multi", "Teal", "Gold", "Silver", "Copper", "Bronze", "Metallic",
    "Nude", "Skin", "Taupe", "Mauve", "Rust", "Peach", "Khaki", "khaki",
])
def test_excluded_colors_map_to_nothing(raw):
    assert tx.map_color(raw) is None
    assert tx.is_excluded_color(raw)


def test_unlisted_color_is_unmapped_but_not_excluded():
    assert tx.map_color("Fuchsia Sparkle") is None
    assert not tx.is_excluded_color("Fuchsia Sparkle")


def test_excluded_and_mapped_colors_are_disjoint():
    assert not tx.EXCLUDED_RAW_COLORS & set(tx.COLOR_RAW_TO_LABEL)


def _color_and_material_words() -> set[str]:
    words = set()
    for attr in ("color", "material"):
        words |= set(tx.lexicon(attr))
    return words


def _mentions(text: str, words: set[str]) -> list[str]:
    text = text.lower()
    return [w for w in words if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text)]


def test_category_prompts_contain_no_color_or_material_word():
    words = _color_and_material_words()
    for label, prompts in tx.clip_prompts("category").items():
        for prompt in prompts:
            assert not _mentions(prompt, words), (label, prompt)


def test_clip_prompts_have_four_templates_per_class():
    for attr in tx.ATTRIBUTES:
        prompts = tx.clip_prompts(attr)
        assert list(prompts) == list(tx.LABELS[attr])
        for filled in prompts.values():
            assert len(filled) == 4
            assert all("{" not in p and "}" not in p for p in filled)
    assert tx.clip_prompts("material")["silk"][1] == "clothing made from shiny silk or satin."
    assert tx.clip_prompts("color")["red"][0] == "a photo of a red garment."


def test_surface_forms():
    assert tx.surface_form("material", "knit") == "knit fabric"
    assert tx.surface_form("material", "synthetic") == "synthetic fabric"
    assert tx.surface_form("category", "t-shirt") == "t-shirt"
    assert tx.surface_form("color", "beige") == "beige"
    with pytest.raises(ValueError):
        tx.surface_form("color", "teal")


def test_lexicon_covers_labels_and_is_unambiguous():
    for attr in tx.ATTRIBUTES:
        synonyms = tx.SYNONYMS[attr]
        assert set(tx.LABELS[attr]) <= set(synonyms)
        forms = [f for fs in synonyms.values() for f in fs]
        assert len(forms) == len(set(forms)), f"{attr} surface form maps to two labels"
        assert all(f == f.lower() for f in forms)
    # Every surface form of a label is also its own label's synonym.
    for label in tx.CATEGORIES:
        assert label in tx.CATEGORY_SYNONYMS[label]


def test_color_lexicon_contents():
    lex = tx.lexicon("color")
    for raw, label in tx.COLOR_RAW_TO_LABEL.items():
        assert lex[raw] == label
    assert lex["ivory"] == "white"
    assert lex["navy"] == "blue" and lex["indigo"] == "blue"
    assert lex["wine"] == "red" and lex["crimson"] == "red" and lex["scarlet"] == "red"
    for word in ("teal", "khaki", "mint", "coral", "peach", "rust", "mauve", "gold",
                 "silver", "multicolor"):
        assert lex[word] == tx.OTHER


def test_material_lexicon_contents():
    lex = tx.lexicon("material")
    assert lex["georgette"] == "chiffon"
    assert lex["cashmere"] == "knit"
    assert lex["faux leather"] == "leather" and lex["suede"] == "leather"
    assert lex["satin"] == "silk"
    assert lex["viscose"] == "synthetic"


def test_gold_values_and_validation():
    assert tx.gold_values("category")[-1] == tx.EXCLUDE
    assert tx.gold_values("color")[-1] == tx.EXCLUDE
    assert tx.gold_values("material")[-1] == tx.UNKNOWN
    assert tx.validate_gold("material", "unknown") == "unknown"
    with pytest.raises(ValueError):
        tx.validate_gold("material", "exclude")
    with pytest.raises(ValueError):
        tx.validate_gold("color", "teal")
    assert tx.is_label("color", "red")
    assert not tx.is_label("material", tx.UNKNOWN)


def test_ethnic_group():
    assert tx.is_ethnic("kurta") and tx.is_ethnic("saree")
    assert not tx.is_ethnic("dress")
