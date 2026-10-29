"""Label sets, metadata mappings, CLIP prompts and the claim lexicon (spec §4).

This module is the only place where labels, synonyms and lexical rules are
defined. Every other module imports them from here.

Special values:
    ``unknown``: not visually determinable, or the extractor abstained.
    ``exclude``: the annotator marked the item unusable for that attribute.
    ``other``:   an off-taxonomy color word in a description; it counts as a
                 stated color that never matches gold.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

ATTRIBUTES: tuple[str, ...] = ("category", "color", "material")
UNKNOWN = "unknown"
EXCLUDE = "exclude"
OTHER = "other"

# --------------------------------------------------------------------------
# Category: raw articleType -> label -> CLIP description
# --------------------------------------------------------------------------

# Descriptions must not mention any color or material word (tested).
_CATEGORY_TABLE: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("Tshirts",), "t-shirt", "t-shirt"),
    (("Shirts",), "shirt", "button-down shirt"),
    (("Tops",), "top", "women's top"),
    (("Kurtas", "Kurtis"), "kurta", "kurta, a long Indian tunic"),
    (("Sarees",), "saree", "saree, a draped Indian garment"),
    (("Jeans",), "jeans", "pair of jeans"),
    (("Trousers",), "trousers", "pair of trousers"),
    (("Shorts",), "shorts", "pair of shorts"),
    (("Dresses",), "dress", "dress"),
    (("Skirts",), "skirt", "skirt"),
    (("Sweaters",), "sweater", "sweater"),
    (("Sweatshirts",), "sweatshirt", "sweatshirt"),
    (("Jackets",), "jacket", "jacket"),
)

CATEGORIES: tuple[str, ...] = tuple(label for _, label, _ in _CATEGORY_TABLE)
CATEGORY_RAW_TO_LABEL: Mapping[str, str] = MappingProxyType(
    {raw: label for raws, label, _ in _CATEGORY_TABLE for raw in raws}
)
CATEGORY_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {label: desc for _, label, desc in _CATEGORY_TABLE}
)
#: Ethnic-wear group used for stratified analysis (T7).
ETHNIC_CATEGORIES: frozenset[str] = frozenset({"kurta", "saree"})
#: Plural-only garment nouns ("a pair of jeans", never "a jeans"); used by APG.
PLURAL_CATEGORIES: frozenset[str] = frozenset({"jeans", "trousers", "shorts"})

# --------------------------------------------------------------------------
# Color: raw baseColour (case-insensitive) -> 12 basic terms
# --------------------------------------------------------------------------

_COLOR_TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("black", ("Black",)),
    ("white", ("White", "Off White")),
    ("grey", ("Grey", "Grey Melange", "Charcoal", "Steel")),
    ("blue", ("Blue", "Navy Blue", "Turquoise Blue")),
    ("red", ("Red", "Maroon", "Burgundy")),
    ("green", ("Green", "Olive", "Sea Green", "Lime Green", "Fluorescent Green")),
    ("yellow", ("Yellow", "Mustard")),
    ("orange", ("Orange",)),
    ("pink", ("Pink", "Magenta", "Rose")),
    ("purple", ("Purple", "Lavender")),
    ("brown", ("Brown", "Coffee Brown", "Mushroom Brown", "Tan")),
    ("beige", ("Beige", "Cream")),
)


def _norm(text: str) -> str:
    """Lower-case and collapse internal whitespace."""
    return " ".join(str(text).split()).lower()


COLORS: tuple[str, ...] = tuple(label for label, _ in _COLOR_TABLE)
COLOR_RAW_TO_LABEL: Mapping[str, str] = MappingProxyType(
    {_norm(raw): label for label, raws in _COLOR_TABLE for raw in raws}
)
#: Raw colors whose items are dropped from the dataset.
EXCLUDED_RAW_COLORS: frozenset[str] = frozenset(
    _norm(c)
    for c in (
        "Multi", "Teal", "Gold", "Silver", "Copper", "Bronze", "Metallic",
        "Nude", "Skin", "Taupe", "Mauve", "Rust", "Peach", "Khaki",
    )
)

# --------------------------------------------------------------------------
# Material: label -> (CLIP description, surface form used in APG text)
# --------------------------------------------------------------------------

_MATERIAL_TABLE: tuple[tuple[str, str, str], ...] = (
    ("chiffon", "sheer chiffon or georgette", "chiffon"),
    ("cotton", "cotton", "cotton"),
    ("denim", "denim", "denim"),
    ("knit", "knitted wool", "knit fabric"),
    ("leather", "leather", "leather"),
    ("linen", "linen", "linen"),
    ("silk", "shiny silk or satin", "silk"),
    ("synthetic", "polyester or synthetic fabric", "synthetic fabric"),
)

MATERIALS: tuple[str, ...] = tuple(label for label, _, _ in _MATERIAL_TABLE)
MATERIAL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {label: desc for label, desc, _ in _MATERIAL_TABLE}
)
MATERIAL_SURFACE_FORMS: Mapping[str, str] = MappingProxyType(
    {label: surface for label, _, surface in _MATERIAL_TABLE}
)

LABELS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"category": CATEGORIES, "color": COLORS, "material": MATERIALS}
)

# --------------------------------------------------------------------------
# CLIP prompt templates (4 per attribute; {} = class description)
# --------------------------------------------------------------------------

CLIP_TEMPLATES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "category": (
        "a product photo of a {}.",
        "an e-commerce catalog photo of a {}.",
        "a photo of a person wearing a {}.",
        "a {} on a plain background.",
    ),
    "color": (
        "a photo of a {} garment.",
        "a product photo of {} clothing.",
        "a {} colored piece of clothing.",
        "an item of clothing that is {} in color.",
    ),
    "material": (
        "a close-up photo of a garment made of {}.",
        "clothing made from {}.",
        "a product photo of a garment with the texture of {}.",
        "a garment made of {}.",
    ),
})

# --------------------------------------------------------------------------
# Synonym lexicon for claim extraction (used only by claims.py, spec §10)
# --------------------------------------------------------------------------

CATEGORY_SYNONYMS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "t-shirt": ("t-shirt", "t shirt", "tshirt", "tee"),
    "shirt": ("shirt",),
    "top": ("top",),
    "kurta": ("kurta", "kurti"),
    "saree": ("saree", "sari"),
    "jeans": ("jeans",),
    "trousers": ("trousers", "pants", "chinos"),
    "shorts": ("shorts",),
    "dress": ("dress",),
    "skirt": ("skirt",),
    "sweater": ("sweater", "pullover", "jumper"),
    "sweatshirt": ("sweatshirt", "hoodie"),
    "jacket": ("jacket", "blazer"),
})

#: Phrases in which a category word is not a garment mention.
CATEGORY_EXCLUSION_PHRASES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "top": ("top quality", "top-quality", "top choice", "on top", "top-notch", "tops off"),
    "dress": ("dress up", "dress down", "dress code", "dressy"),
})

# Every raw color value counts as stating its label, plus these extras.
# "gray" and the multicolour spellings are additions (DECISIONS.md, M1 #2).
_COLOR_EXTRA_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("white", ("ivory",)),
    ("grey", ("gray",)),
    ("blue", ("navy", "indigo")),
    ("red", ("crimson", "scarlet", "wine")),
)
_OFF_TAXONOMY_COLOR_WORDS: tuple[str, ...] = (
    "teal", "khaki", "mint", "coral", "peach", "rust", "mauve", "gold", "silver",
    "multicolor", "multicolour", "multicolored", "multicoloured",
    "multi-colored", "multi-coloured",
)


def _build_color_synonyms() -> dict[str, tuple[str, ...]]:
    forms: dict[str, list[str]] = {label: [] for label in COLORS}
    for label, raws in _COLOR_TABLE:
        forms[label].extend(_norm(r) for r in raws)
    for label, extras in _COLOR_EXTRA_SYNONYMS:
        forms[label].extend(extras)
    forms[OTHER] = list(_OFF_TAXONOMY_COLOR_WORDS)
    return {label: tuple(dict.fromkeys(values)) for label, values in forms.items()}


COLOR_SYNONYMS: Mapping[str, tuple[str, ...]] = MappingProxyType(_build_color_synonyms())

MATERIAL_SYNONYMS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "chiffon": ("chiffon", "georgette"),
    "cotton": ("cotton",),
    "denim": ("denim",),
    "knit": ("knit", "knitted", "wool", "woollen", "woolen", "cashmere"),
    "leather": ("leather", "faux leather", "suede"),
    "linen": ("linen",),
    "silk": ("silk", "satin"),
    "synthetic": (
        "polyester", "nylon", "spandex", "lycra", "elastane", "rayon", "viscose",
        "synthetic",
    ),
})

SYNONYMS: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "category": CATEGORY_SYNONYMS,
    "color": COLOR_SYNONYMS,
    "material": MATERIAL_SYNONYMS,
})

#: A styling clause starts at one of these and runs to the end of the sentence.
STYLING_TRIGGERS: tuple[str, ...] = (
    "pair it with", "pair with", "team it with", "team with", "style it with",
    "wear it with", "wear with", "goes well with", "match it with",
    "complete the look",
)

#: A color word directly describing one of these (optionally one word in
#: between: "white floral print") is the color of a detail, not of the garment.
#: Added after claims validation on dev (DECISIONS.md, M7 validation).
#: Stripes, accents and patterns are deliberately absent: both validation
#: labellers counted their colors as garment colors.
DETAIL_NOUNS: tuple[str, ...] = (
    "logo", "logos", "design", "designs", "print", "prints", "graphic", "graphics",
    "text", "lettering", "letters", "trim", "trims", "embroidery", "detail", "details",
    "detailing", "button", "buttons", "motif", "motifs", "piping",
)

#: A mention is hedged if its sentence contains any of these.
HEDGE_CUES: tuple[str, ...] = (
    "may vary", "might", "may be", "appears to", "looks like", "seems",
    "possibly", "likely", "or similar",
)

# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------


def map_category(raw: str | None) -> str | None:
    """Map a raw articleType to a category label, or ``None`` if unmapped."""
    if raw is None:
        return None
    return CATEGORY_RAW_TO_LABEL.get(str(raw).strip())


def map_color(raw: str | None) -> str | None:
    """Map a raw baseColour (case-insensitive) to a color label.

    Returns ``None`` for excluded colors and for any other unmapped value;
    use :func:`is_excluded_color` to tell the two apart in reports.
    """
    if raw is None:
        return None
    return COLOR_RAW_TO_LABEL.get(_norm(raw))


def is_excluded_color(raw: str | None) -> bool:
    """True if ``raw`` is on the explicit excluded-color list."""
    return raw is not None and _norm(raw) in EXCLUDED_RAW_COLORS


def class_descriptions(attribute: str) -> dict[str, str]:
    """Label → text inserted into the CLIP templates for ``attribute``."""
    if attribute == "category":
        return dict(CATEGORY_DESCRIPTIONS)
    if attribute == "color":
        return {label: label for label in COLORS}
    if attribute == "material":
        return dict(MATERIAL_DESCRIPTIONS)
    raise ValueError(f"Unknown attribute: {attribute!r}")


def clip_prompts(attribute: str) -> dict[str, list[str]]:
    """Label → the 4 filled CLIP prompts for ``attribute`` (for ensembling)."""
    templates = CLIP_TEMPLATES[attribute]
    return {
        label: [template.format(desc) for template in templates]
        for label, desc in class_descriptions(attribute).items()
    }


def surface_form(attribute: str, label: str) -> str:
    """Word(s) APG writes into the text for ``label`` (spec §9, renderer step 4)."""
    if attribute == "material":
        return MATERIAL_SURFACE_FORMS[label]
    if label not in LABELS[attribute]:
        raise ValueError(f"{label!r} is not a {attribute} label")
    return label


def lexicon(attribute: str) -> dict[str, str]:
    """Flattened surface form → label map for ``attribute`` (claims.py)."""
    return {form: label for label, forms in SYNONYMS[attribute].items() for form in forms}


def is_label(attribute: str, value: str | None) -> bool:
    """True if ``value`` is a real label (not unknown, exclude or missing)."""
    return value in LABELS[attribute]


def gold_values(attribute: str) -> tuple[str, ...]:
    """Allowed gold / annotation values, in dropdown order (spec §6).

    Category and color: labels + ``exclude``. Material: labels + ``unknown``.
    """
    special = UNKNOWN if attribute == "material" else EXCLUDE
    return (*LABELS[attribute], special)


def validate_gold(attribute: str, value: str) -> str:
    """Return ``value`` if it is an allowed gold value, else raise ``ValueError``."""
    if value not in gold_values(attribute):
        raise ValueError(f"Invalid {attribute} gold value: {value!r}")
    return value


def is_ethnic(category: str | None) -> bool:
    """True for the ethnic-wear group (kurta, saree)."""
    return category in ETHNIC_CATEGORIES
