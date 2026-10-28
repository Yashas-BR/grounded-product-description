from __future__ import annotations

import json
import re

import pytest

from groundedgen import prompts
from groundedgen import taxonomy as tx

FULL = {"category": "kurta", "color": "red", "material": "silk"}
NO_CAT = {"category": "unknown", "color": "red", "material": "unknown"}
NONE = {"category": "unknown", "color": "unknown", "material": "unknown"}
SLOT_RE = re.compile(r"\{(category|color|material)\}")


def _leftover_braces(text: str) -> list[str]:
    """Brace tokens other than APG slots and the evidence JSON object."""
    stripped = SLOT_RE.sub("", text)
    stripped = re.sub(r"\{\"category\": \"[^\"]*\", \"color\": \"[^\"]*\", \"material\": \"[^\"]*\"\}",
                      "", stripped)
    stripped = stripped.replace('{"claims": [{"text": "...", "type": "..."}]}', "")
    return re.findall(r"[{}]", stripped)


@pytest.mark.parametrize("condition", ["A", "A-F", "B", "B-Abs", "C", "C-Mask", "APG"])
@pytest.mark.parametrize("evidence", [FULL, NO_CAT, NONE])
def test_templates_format_without_leftover_braces(condition, evidence):
    p = prompts.build_prompt(condition, evidence, image_path="img.jpg")
    for text in (p.system or "", p.user):
        assert not _leftover_braces(text), text
    if condition == "APG":
        assert set(SLOT_RE.findall(p.user)) == set(prompts.apg_placeholders(evidence)) | (
            {"category"} if evidence["category"] != "unknown" else set())
    else:
        assert not SLOT_RE.search(p.user)


def test_condition_prompts():
    a = prompts.build_prompt("A", image_path="x.jpg")
    assert a.system is None and a.user == prompts.IMAGE_USER and a.image_path == "x.jpg"
    af = prompts.build_prompt("A-F", image_path="x.jpg")
    assert af.system == prompts.FAITHFUL_SYSTEM_IMAGE
    b = prompts.build_prompt("B", FULL)
    assert b.system == prompts.FAITHFUL_SYSTEM_TEXT
    assert b.user.startswith('Product attributes:\n{"category": "kurta", "color": "red", "material": "silk"}\n\n')
    assert b.image_path is None
    with pytest.raises(ValueError):
        prompts.build_prompt("B")


def test_evidence_json_key_order_and_no_confidences():
    shuffled = {"material": "silk", "color": "red", "category": "kurta", "confidence": 0.9}
    assert prompts.evidence_json(shuffled) == '{"category": "kurta", "color": "red", "material": "silk"}'
    assert list(json.loads(prompts.evidence_json(shuffled))) == list(tx.ATTRIBUTES)


def test_apg_prompt_variants():
    full = prompts.apg_user(FULL)
    assert "placeholders exactly where the values belong: {category}, {color}, {material}." in full
    assert "use {category} in the first sentence" in full
    no_cat = prompts.apg_user(NO_CAT)
    assert "{category}" not in no_cat and "{color}" in no_cat
    assert '"this piece"' in no_cat
    none = prompts.apg_user(NONE)
    assert "placeholders" not in none and '"this piece"' in none
    for text in (full, no_cat, none):
        assert text.endswith("Output only the description.")


def test_judge_prompt_and_retry():
    p = prompts.judge_prompt(FULL, "A red silk kurta.")
    assert "Description:\nA red silk kurta." in p.user
    assert p.user.endswith('{"claims": [{"text": "...", "type": "..."}]}')
    retry = prompts.judge_prompt(FULL, "x", retry=True)
    assert retry.user.endswith("Answer with valid JSON only.")


def test_hashes_are_stable_and_distinct():
    first = prompts.template_hashes()
    assert first == prompts.template_hashes()
    assert len(set(first.values())) == len(first)
    assert all(len(h) == 64 for h in first.values())
    assert prompts.condition_sha256("B") == prompts.condition_sha256("C")  # same templates
    assert prompts.condition_sha256("A") != prompts.condition_sha256("A-F")
    record = prompts.freeze_record()
    assert record["prompt_version"] == prompts.PROMPT_VERSION
    assert set(record["conditions"]) == {"A", "A-F", "B", "B-Abs", "APG", "C", "C-Mask"}


def test_prompt_texts_match_spec():
    assert prompts.FAITHFUL_SYSTEM_TEXT.endswith(
        'If an attribute is "unknown", do not mention it and do not guess it.')
    assert "(40 to 70 words). Output only the description." in prompts.GROUNDED_USER
    assert prompts.IMAGE_USER.startswith("Write a description of this product for an online fashion store")
