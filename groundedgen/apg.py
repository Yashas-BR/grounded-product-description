"""Abstention-Preserving Generation: deterministic slot renderer (spec §9).

The LLM writes placeholders (``{category}``, ``{color}``, ``{material}``);
this module fills, prunes and verifies them with no model calls, so attribute
values reach the text only by copying from the evidence.

Guarantee (property-tested): for any evidence and any LLM output, the
rendered text never states a category, color or material that differs from
the evidence, never states an attribute whose evidence is ``unknown``, and
never contains a raw placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from groundedgen import claims
from groundedgen import taxonomy as tx

SLOT_RE = re.compile(r"\{(category|color|material)\}")
ANY_BRACE_TOKEN_RE = re.compile(r"\{[^{}]*\}")
STRAY_BRACE_RE = re.compile(r"[{}]")
ARTICLE_SLOT_RE = re.compile(r"\b(a|an)(\s+)\{(category|color|material)\}", re.IGNORECASE)

NOTHING_KNOWN_TEXT = "Product details are available on request."


@dataclass
class APGResult:
    """Rendered text plus the per-record log stored in the ``apg`` field."""

    text: str
    log: dict[str, Any] = field(default_factory=dict)


def _surface(attribute: str, evidence: Mapping[str, str]) -> str:
    return tx.surface_form(attribute, evidence[attribute])


def _article(word: str, original: str) -> str:
    art = "an" if word[:1].lower() in "aeiou" else "a"
    return art.capitalize() if original[:1].isupper() else art


def _capitalise(sentence: str) -> str:
    for i, ch in enumerate(sentence):
        if ch.isalpha():
            return sentence[:i] + ch.upper() + sentence[i + 1:]
    return sentence


def fill(sentence: str, evidence: Mapping[str, str]) -> str:
    """Replace slots with surface forms, fixing "a"/"an" before each filled word."""
    def article(m: re.Match[str]) -> str:
        word = _surface(m.group(3), evidence)
        if m.group(3) == "category" and evidence["category"] in tx.PLURAL_CATEGORIES:
            return f"{_article('pair', m.group(1))} pair of {word}"
        return f"{_article(word, m.group(1))}{m.group(2)}{word}"

    sentence = ARTICLE_SLOT_RE.sub(article, sentence)
    sentence = SLOT_RE.sub(lambda m: _surface(m.group(1), evidence), sentence)
    return _capitalise(sentence)


def _invalid(sentence: str, evidence: Mapping[str, str]) -> bool:
    """Prune rule: an unknown token, a stray brace, or a slot for an unknown attribute."""
    tokens = ANY_BRACE_TOKEN_RE.findall(sentence)
    if any(not SLOT_RE.fullmatch(t) for t in tokens):
        return True
    if STRAY_BRACE_RE.search(ANY_BRACE_TOKEN_RE.sub("", sentence)):
        return True
    return any(evidence[a] == tx.UNKNOWN for a in SLOT_RE.findall(sentence))


def violates(sentence: str, evidence: Mapping[str, str]) -> bool:
    """Verify rule: the sentence states a value other than the evidence, states
    an attribute whose evidence is unknown, or hedges an attribute.

    Every counted mention is checked (not only the first category mention), so
    a kept sentence is consistent with the evidence however it is read.
    """
    for m in claims.find_mentions(sentence):
        if m.in_styling:
            continue
        if m.hedged or evidence[m.attribute] == tx.UNKNOWN or m.label != evidence[m.attribute]:
            return True
    return False


def fallback_text(evidence: Mapping[str, str]) -> str:
    """Template used when no sentence survives or the category went unstated.

    "A {color} {category} in {material}." with unknown parts left out; without
    a category, "This piece comes in {color}." (or "This piece is made of
    {material}." if only material is known); with nothing known, a neutral
    sentence.
    """
    known = {a: evidence[a] != tx.UNKNOWN for a in tx.ATTRIBUTES}
    if known["category"]:
        parts = (["{color}"] if known["color"] else []) + ["{category}"]
        # Plural-only garments take no article: "Blue jeans in denim."
        lead = [] if evidence["category"] in tx.PLURAL_CATEGORIES else ["A"]
        template = " ".join(lead + parts) + (" in {material}." if known["material"] else ".")
    elif known["color"]:
        template = "This piece comes in {color}."
    elif known["material"]:
        template = "This piece is made of {material}."
    else:
        return NOTHING_KNOWN_TEXT
    return fill(template, evidence)


def render(text: str, evidence: Mapping[str, str]) -> APGResult:
    """Render one APG output (spec §9 steps 1–6) and log what happened."""
    offered = [a for a in tx.ATTRIBUTES if evidence[a] != tx.UNKNOWN]
    sentences = claims.split_sentences(text)
    kept_after_prune = [s for s in sentences if not _invalid(s, evidence)]
    filled = [(s, fill(s, evidence)) for s in kept_after_prune]
    verified = [(raw, out) for raw, out in filled if not violates(out, evidence)]

    used = sorted({a for raw, _ in verified for a in SLOT_RE.findall(raw)},
                  key=tx.ATTRIBUTES.index)
    output = " ".join(out.strip() for _, out in verified)
    category_missing = (
        evidence["category"] != tx.UNKNOWN
        and not claims.extract_claims(output)["category"]["stated"]
    )
    fallback = not verified or category_missing
    if fallback:
        output = fallback_text(evidence)
    return APGResult(output, {
        "raw_with_slots": text,
        "placeholders_offered": offered,
        "placeholders_used": used,
        "sentences_total": len(sentences),
        "sentences_dropped_prune": len(sentences) - len(kept_after_prune),
        "sentences_dropped_verify": len(filled) - len(verified),
        "fallback_used": fallback,
    })
