"""Every prompt template, versioned and hashed (spec §8, §9, §10).

Templates are frozen on dev; their SHA-256 hashes are logged with every
generation record and written to ``outputs/frozen.json`` before any test run.
Never edit a template without bumping ``PROMPT_VERSION``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from groundedgen import taxonomy as tx
from groundedgen import utils

PROMPT_VERSION = "v1"

# Literal braces in templates are doubled; ``{evidence}`` and ``{text}`` are
# the only format fields.

FAITHFUL_SYSTEM_TEXT = (
    "You write product descriptions for an online fashion store. Use only the "
    "product attributes you are given. Do not state any color, material, fabric, "
    "pattern, fit, size, brand, care instruction, origin or occasion that is not "
    "listed. If an attribute is \"unknown\", do not mention it and do not guess it."
)

GROUNDED_USER = (
    "Product attributes:\n"
    "{evidence}\n"
    "\n"
    "Write a description of this product for the store, in 2 to 3 sentences (40 to "
    "70 words). Output only the description."
)

FAITHFUL_SYSTEM_IMAGE = (
    "You write product descriptions for an online fashion store. Describe only "
    "what is clearly visible in the image. Do not state any color, material, "
    "fabric, pattern, fit, size, brand, care instruction, origin or occasion that "
    "you cannot see. If you cannot tell an attribute from the image, do not "
    "mention it and do not guess it."
)

IMAGE_USER = (
    "Write a description of this product for an online fashion store, in 2 to 3 "
    "sentences (40 to 70 words). Output only the description."
)

# APG user prompt (§9). ``{placeholders}`` and the category clause are filled
# by :func:`apg_user`; slot names stay literal in the text sent to the LLM.
APG_USER_HEAD = (
    "Product attributes:\n"
    "{evidence}\n"
    "\n"
    "Write a description of this product for the store, in 2 to 3 sentences (40 to "
    "70 words)."
)
APG_SLOT_INSTRUCTION = (
    " Do not write the attribute values yourself. Instead, write these "
    "placeholders exactly where the values belong: {placeholders}."
)
APG_CATEGORY_CLAUSE = (
    " Use each placeholder at most once, and use {{category}} in the first "
    "sentence if it is listed."
)
APG_NO_CATEGORY_CLAUSE = (
    " Use each placeholder at most once, and refer to the product as \"this piece\"."
)
APG_NO_SLOTS_CLAUSE = " Refer to the product as \"this piece\"."
APG_USER_TAIL = (
    " Do not write any other color, fabric or garment-type words. Output only the "
    "description."
)

JUDGE_USER = (
    "You check product descriptions for unsupported claims. Evidence (the only "
    "facts known about the product):\n"
    "{evidence}\n"
    "\n"
    "Description:\n"
    "{text}\n"
    "\n"
    "List every claim in the description that is not supported by the evidence, "
    "except claims about category, color or material. For each, give the exact "
    "words and one type: fit, size, function, care, composition, brand, origin, "
    "occasion, pattern, or subjective. Use \"subjective\" for opinions such as "
    "stylish or elegant. Answer with JSON only: "
    "{{\"claims\": [{{\"text\": \"...\", \"type\": \"...\"}}]}}"
)
JUDGE_RETRY_SUFFIX = "\n\nAnswer with valid JSON only."

#: Every template, for hashing and freezing.
TEMPLATES: Mapping[str, str] = {
    "FAITHFUL_SYSTEM_TEXT": FAITHFUL_SYSTEM_TEXT,
    "GROUNDED_USER": GROUNDED_USER,
    "FAITHFUL_SYSTEM_IMAGE": FAITHFUL_SYSTEM_IMAGE,
    "IMAGE_USER": IMAGE_USER,
    "APG_USER_HEAD": APG_USER_HEAD,
    "APG_SLOT_INSTRUCTION": APG_SLOT_INSTRUCTION,
    "APG_CATEGORY_CLAUSE": APG_CATEGORY_CLAUSE,
    "APG_NO_CATEGORY_CLAUSE": APG_NO_CATEGORY_CLAUSE,
    "APG_NO_SLOTS_CLAUSE": APG_NO_SLOTS_CLAUSE,
    "APG_USER_TAIL": APG_USER_TAIL,
    "JUDGE_USER": JUDGE_USER,
    "JUDGE_RETRY_SUFFIX": JUDGE_RETRY_SUFFIX,
}

#: Templates each generation condition uses (system first).
CONDITION_TEMPLATES: Mapping[str, tuple[str, ...]] = {
    "A": ("IMAGE_USER",),
    "A-F": ("FAITHFUL_SYSTEM_IMAGE", "IMAGE_USER"),
    "B": ("FAITHFUL_SYSTEM_TEXT", "GROUNDED_USER"),
    "B-Abs": ("FAITHFUL_SYSTEM_TEXT", "GROUNDED_USER"),
    "C": ("FAITHFUL_SYSTEM_TEXT", "GROUNDED_USER"),
    "C-Mask": ("FAITHFUL_SYSTEM_TEXT", "GROUNDED_USER"),
    "APG": ("FAITHFUL_SYSTEM_TEXT", "APG_USER_HEAD", "APG_SLOT_INSTRUCTION",
            "APG_CATEGORY_CLAUSE", "APG_NO_CATEGORY_CLAUSE", "APG_NO_SLOTS_CLAUSE",
            "APG_USER_TAIL"),
}


@dataclass(frozen=True)
class ChatPrompt:
    """One model input: optional system text, user text, optional image path."""

    system: str | None
    user: str
    image_path: str | None = None


def evidence_json(evidence: Mapping[str, str]) -> str:
    """``json.dumps`` of the evidence with keys in the order category, color, material.

    Confidences are never shown to the LLM, so B, B-Abs and C differ only in
    the values.
    """
    return json.dumps({a: evidence[a] for a in tx.ATTRIBUTES})


def apg_placeholders(evidence: Mapping[str, str]) -> list[str]:
    """Slots offered to the LLM: attributes whose evidence is not ``unknown``."""
    return [a for a in tx.ATTRIBUTES if evidence[a] != tx.UNKNOWN]


def apg_user(evidence: Mapping[str, str]) -> str:
    """The APG user prompt for this evidence (spec §9).

    With category known the text must use ``{category}`` in the first
    sentence; without it the text refers to "this piece". With no known
    attribute at all, no slots are offered.
    """
    slots = apg_placeholders(evidence)
    text = APG_USER_HEAD.format(evidence=evidence_json(evidence))
    if slots:
        text += APG_SLOT_INSTRUCTION.format(
            placeholders=", ".join("{" + s + "}" for s in slots)
        )
        text += (APG_CATEGORY_CLAUSE if "category" in slots else APG_NO_CATEGORY_CLAUSE).format()
    else:
        text += APG_NO_SLOTS_CLAUSE
    return text + APG_USER_TAIL


def build_prompt(
    condition: str, evidence: Mapping[str, str] | None = None, image_path: str | None = None
) -> ChatPrompt:
    """The model input for one generation of ``condition``."""
    if condition == "A":
        return ChatPrompt(None, IMAGE_USER, image_path)
    if condition == "A-F":
        return ChatPrompt(FAITHFUL_SYSTEM_IMAGE, IMAGE_USER, image_path)
    if evidence is None:
        raise ValueError(f"Condition {condition} needs evidence")
    if condition == "APG":
        return ChatPrompt(FAITHFUL_SYSTEM_TEXT, apg_user(evidence))
    if condition in ("B", "B-Abs", "C", "C-Mask"):
        return ChatPrompt(FAITHFUL_SYSTEM_TEXT, GROUNDED_USER.format(evidence=evidence_json(evidence)))
    raise ValueError(f"Unknown condition: {condition!r}")


def judge_prompt(evidence: Mapping[str, str], text: str, retry: bool = False) -> ChatPrompt:
    """Judge input (spec §10); ``retry`` appends the valid-JSON reminder."""
    user = JUDGE_USER.format(evidence=evidence_json(evidence), text=text)
    return ChatPrompt(None, user + (JUDGE_RETRY_SUFFIX if retry else ""))


def template_hashes() -> dict[str, str]:
    """SHA-256 of every template."""
    return {name: utils.sha256_text(t) for name, t in TEMPLATES.items()}


def condition_sha256(condition: str) -> str:
    """One hash over the templates a condition uses, logged with each record."""
    names: Sequence[str] = CONDITION_TEMPLATES[condition]
    return utils.sha256_text("\n\x1e\n".join(f"{n}:{TEMPLATES[n]}" for n in names))


def freeze_record() -> dict[str, Any]:
    """What ``outputs/frozen.json`` stores about prompts (M8)."""
    return {
        "prompt_version": PROMPT_VERSION,
        "templates": template_hashes(),
        "conditions": {c: condition_sha256(c) for c in CONDITION_TEMPLATES},
    }
