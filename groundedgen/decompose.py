"""Per-pair outcomes, hallucination origins, leakage and inflation (spec §11).

The unit of analysis is one (description, attribute) pair. Pairs whose gold
is ``exclude`` or missing are not scored. Notation: ``g`` gold (label or
``unknown``), ``e`` evidence given to the generator (absent for A / A-F),
``D`` the set of values the description states, ``h`` hedged.

Outcomes:
    correct               D = {g} and g is a label
    hallucination         D non-empty and D ≠ {g} (any statement when g = unknown)
    omission              D empty and e is a label (A/A-F: g is a label)
    preserved_abstention  D empty and e = unknown
    silent                D empty, no evidence and g = unknown (A/A-F only)

Origins (hallucinations with evidence only):
    EP  e label, D = {e}, e ≠ g   vision stage
    GD  e label, D ≠ {e}           language stage
    GF  e = unknown, D non-empty   language stage

"Lucky" statements (e label, D ≠ {e}, D = {g}) count as correct and are
reported separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from groundedgen import claims, metrics, utils
from groundedgen import taxonomy as tx
from groundedgen.generate import CACHE_KEY

OUTCOMES: tuple[str, ...] = (
    "correct", "hallucination", "omission", "preserved_abstention", "silent",
)
ORIGINS: tuple[str, ...] = ("EP", "GD", "GF")
GROUP_KEYS: tuple[str, ...] = ("condition", "generator", "extractor", "attribute")


def classify(g: str, e: str | None, stated: Sequence[str], hedged: bool = False) -> dict[str, Any]:
    """Outcome, origin and flags of one pair (``e`` is ``None`` for A / A-F)."""
    D = set(stated)
    g_label = g != tx.UNKNOWN
    e_label = e is not None and e != tx.UNKNOWN
    if D:
        outcome = "correct" if (g_label and D == {g}) else "hallucination"
    elif e is None:
        outcome = "omission" if g_label else "silent"
    else:
        outcome = "omission" if e_label else "preserved_abstention"

    origin = None
    if outcome == "hallucination" and e is not None:
        if e_label:
            origin = "EP" if D == {e} else "GD"  # D = {e} here implies e ≠ g
        else:
            origin = "GF"
    return {
        "outcome": outcome,
        "origin": origin,
        "lucky": bool(e_label and D != {e} and g_label and D == {g}),
        "stated": bool(D),
        "leak": bool(e is not None and not e_label and D),
        "inflated": bool(e_label and (not D or hedged)),
    }


def decompose_records(
    generations: Iterable[Mapping[str, Any]],
    claims_by_key: Mapping[tuple, Mapping[str, Any]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Scored pairs for every generation record whose item is in ``manifest``.

    ``claims_by_key`` maps a record's cache key to its claims; records without
    stored claims are extracted on the fly.
    """
    gold = manifest.set_index("item_id")
    rows = []
    for r in generations:
        if r["item_id"] not in gold.index:
            continue
        item = gold.loc[r["item_id"]]
        key = utils.record_key(r, CACHE_KEY)
        found = claims_by_key.get(key) or claims.extract_claims(r["text"])
        evidence = r.get("evidence")
        for attribute in tx.ATTRIBUTES:
            g = item[f"gt_{attribute}"]
            if g in ("", tx.EXCLUDE) or g is None or (isinstance(g, float) and np.isnan(g)):
                continue
            e = evidence[attribute] if evidence else None
            info = found[attribute]
            rows.append({
                **{k: r.get(k) for k in CACHE_KEY},
                "attribute": attribute,
                "gold": g,
                "evidence": e,
                "stated_values": list(info["values"]),
                "hedged": bool(info["hedged"]),
                "masked": r.get("mask_attr") == attribute,
                "is_ethnic": bool(item.get("is_ethnic", False)),
                **classify(g, e, info["values"], info["hedged"]),
            })
    columns = [*CACHE_KEY, "attribute", "gold", "evidence", "stated_values", "hedged", "masked",
               "is_ethnic", "outcome", "origin", "lucky", "stated", "leak", "inflated"]
    return pd.DataFrame(rows, columns=columns)


def load_pairs(cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    """Decompose every generation file of ``split`` (claims from ``outputs/claims``)."""
    from groundedgen import data

    manifest = data.load_manifest(cfg, splits=[split])
    frames = []
    for gen_file in sorted(utils.project_path(cfg, "outputs", "generation").rglob("*.jsonl")):
        stored = {utils.record_key(r, CACHE_KEY): r["claims"]
                  for r in utils.read_jsonl(claims.claims_path(cfg["project_dir"], gen_file))}
        frames.append(decompose_records(utils.read_jsonl(gen_file), stored, manifest))
    return pd.concat(frames, ignore_index=True) if frames else decompose_records([], {}, manifest)


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------

def _rate(num: pd.Series, den: pd.Series, clusters: pd.Series, boot: Mapping[str, Any]) -> dict:
    return metrics.bootstrap_ratio(num.astype(float), den.astype(float), clusters, **boot).as_dict()


def group_rates(pairs: pd.DataFrame, boot: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Rates per condition × generator × extractor × attribute, with CIs.

    HR = hallucinations / N; coverage = stated / N; precision = correct /
    statements; EP, GD, GF = counts / N; extraction share = EP /
    hallucinations; leakage = (e unknown and stated) / (e unknown), on the
    masked attribute only for C-Mask; inflation = (e label and (not stated or
    hedged)) / (e label); lucky = lucky / N.
    """
    boot = dict(boot or {})
    rows = []
    frame = pairs.assign(extractor=pairs["extractor"].fillna(""))
    for keys, g in frame.groupby(list(GROUP_KEYS), sort=True):
        hall = g["outcome"] == "hallucination"
        correct = g["outcome"] == "correct"
        ones = pd.Series(1.0, index=g.index)
        c = g["item_id"]
        e_unknown = g["evidence"] == tx.UNKNOWN
        e_label = g["evidence"].notna() & ~e_unknown
        leak_scope = e_unknown & (g["masked"] if keys[0] == "C-Mask" else True)
        row = dict(zip(GROUP_KEYS, keys), N=int(len(g)))
        row["HR"] = _rate(hall, ones, c, boot)
        row["coverage"] = _rate(g["stated"], ones, c, boot)
        row["precision"] = _rate(correct, g["stated"], c, boot)
        row["lucky"] = _rate(g["lucky"], ones, c, boot)
        if g["evidence"].notna().any():
            for origin in ORIGINS:
                row[origin] = _rate(g["origin"] == origin, ones, c, boot)
            row["extraction_share"] = _rate(g["origin"] == "EP", hall, c, boot)
            row["leakage"] = _rate(g["stated"] & leak_scope, leak_scope, c, boot)
            row["inflation"] = _rate(g["inflated"], e_label, c, boot)
        rows.append(row)
    columns = [*GROUP_KEYS, "N", "HR", "coverage", "precision", "lucky", *ORIGINS,
               "extraction_share", "leakage", "inflation"]
    return pd.DataFrame(rows, columns=columns)


def description_rates(pairs: pd.DataFrame, boot: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Share of descriptions with at least one attribute hallucination."""
    boot = dict(boot or {})
    per_desc = (pairs.assign(extractor=pairs["extractor"].fillna(""),
                             mask_attr=pairs["mask_attr"].fillna(""),
                             hall=pairs["outcome"] == "hallucination")
                .groupby([*CACHE_KEY])["hall"].any().reset_index())
    rows = []
    for keys, g in per_desc.groupby(["condition", "generator", "extractor"], sort=True):
        rows.append({**dict(zip(("condition", "generator", "extractor"), keys)),
                     "n_descriptions": int(len(g)),
                     "any_hallucination": _rate(g["hall"], pd.Series(1.0, index=g.index),
                                                g["item_id"], boot)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------

#: Planned family per attribute (primary extractor, each LLM), spec §11.
PLANNED_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A", "A-F"), ("A-F", "B-Abs"), ("A-F", "APG"), ("B", "B-Abs"), ("B-Abs", "APG"),
)


def _indicator(pairs: pd.DataFrame, condition: str, generator: str | None,
               extractor: str | None, attribute: str) -> pd.Series:
    sel = (pairs["condition"] == condition) & (pairs["attribute"] == attribute)
    if generator is not None:
        sel &= pairs["generator"] == generator
    if extractor is not None and condition in ("B", "B-Abs", "APG"):
        sel &= pairs["extractor"] == extractor
    sub = pairs[sel]
    return (sub["outcome"] == "hallucination").groupby(sub["item_id"]).first()


def planned_tests(pairs: pd.DataFrame, primary_extractor: str, llms: Sequence[str],
                  vlm: str, alpha: float = 0.05) -> pd.DataFrame:
    """Exact McNemar for each planned comparison, Holm-corrected within each
    (attribute, LLM) family. A and A-F use ``vlm``; text conditions use the
    family's LLM and the primary extractor."""
    rows = []
    for llm in llms:
        for attribute in tx.ATTRIBUTES:
            family = []
            for left, right in PLANNED_COMPARISONS:
                gl = vlm if left in ("A", "A-F") else llm
                gr = vlm if right in ("A", "A-F") else llm
                x = _indicator(pairs, left, gl, primary_extractor, attribute)
                y = _indicator(pairs, right, gr, primary_extractor, attribute)
                common = x.index.intersection(y.index)
                res = metrics.mcnemar_exact(x.loc[common].to_numpy(), y.loc[common].to_numpy())
                family.append({"llm": llm, "attribute": attribute, "left": left, "right": right,
                               "n_items": int(len(common)), **res})
            adjusted, reject = metrics.holm([f["p_value"] for f in family], alpha)
            for f, p_adj, rej in zip(family, adjusted, reject):
                rows.append({**f, "p_holm": float(p_adj), "reject": bool(rej)})
    return pd.DataFrame(rows)


def origin_test(pairs: pd.DataFrame, primary_extractor: str, llms: Sequence[str],
                boot: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """RQ1: bootstrap CI of GD rate(B) − GD rate(C) per LLM and attribute."""
    boot = dict(boot or {})
    rows = []
    for llm in llms:
        for attribute in tx.ATTRIBUTES:
            def triple(condition: str):
                sel = ((pairs["condition"] == condition) & (pairs["generator"] == llm)
                       & (pairs["attribute"] == attribute))
                if condition == "B":
                    sel &= pairs["extractor"] == primary_extractor
                sub = pairs[sel]
                return ((sub["origin"] == "GD").astype(float), pd.Series(1.0, index=sub.index),
                        sub["item_id"])
            res = metrics.bootstrap_ratio_difference(triple("B"), triple("C"), **boot)
            rows.append({"llm": llm, "attribute": attribute, "gd_diff_B_minus_C": res.as_dict()})
    return pd.DataFrame(rows)


def write_pairs(cfg: Mapping[str, Any], pairs: pd.DataFrame, split: str) -> Path:
    """Save the scored pairs as CSV for inspection."""
    path = utils.project_path(cfg, "outputs", "analysis", f"pairs_{split}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(path, index=False)
    return path
