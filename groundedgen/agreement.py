"""Inter-annotator agreement, disagreement sheets and gold assembly (spec §6).

Flow per split:

1. ``agreement_report`` — Cohen's kappa and raw agreement per attribute.
2. ``write_disagreements`` — ``annotations/disagreements_{split}.csv`` for a
   third team member to fill ``final_value`` (existing entries are kept).
3. ``build_ground_truth`` — gold = agreed value, else adjudicated value;
   writes ``data/manifest_final.csv`` and ``data/ground_truth_report.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from groundedgen import metrics, utils
from groundedgen import taxonomy as tx
from groundedgen.annotate import annotation_path, read_annotations

DISAGREEMENT_COLUMNS: tuple[str, ...] = (
    "item_id", "attribute", "annotator_a", "annotator_b", "final_value",
)


def disagreement_path(project_dir: str | Path, split: str) -> Path:
    return Path(project_dir) / "annotations" / f"disagreements_{split}.csv"


def validate_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """Raise ``ValueError`` if any annotation is outside the allowed set."""
    for attribute in tx.ATTRIBUTES:
        bad = annotations.loc[~annotations[attribute].isin(tx.gold_values(attribute))]
        if len(bad):
            raise ValueError(
                f"Invalid {attribute} labels for {bad['item_id'].tolist()[:5]}: "
                f"{sorted(set(bad[attribute]))}"
            )
    return annotations


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Items labelled by both annotators, with ``{attr}_a`` / ``{attr}_b`` columns."""
    cols = ["item_id", *tx.ATTRIBUTES]
    return a[cols].merge(b[cols], on="item_id", suffixes=("_a", "_b"))


def agreement_report(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Kappa, raw agreement and item count per attribute on doubly-labelled items."""
    both = paired(validate_labels(a), validate_labels(b))
    return {
        attribute: {
            "n_items": int(len(both)),
            "kappa": metrics.cohen_kappa(both[f"{attribute}_a"], both[f"{attribute}_b"]),
            "raw_agreement": metrics.raw_agreement(both[f"{attribute}_a"], both[f"{attribute}_b"]),
        }
        for attribute in tx.ATTRIBUTES
    }


def find_disagreements(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, attribute) where the two annotators differ."""
    both = paired(a, b)
    rows = [
        {
            "item_id": r["item_id"],
            "attribute": attribute,
            "annotator_a": r[f"{attribute}_a"],
            "annotator_b": r[f"{attribute}_b"],
            "final_value": "",
        }
        for _, r in both.iterrows()
        for attribute in tx.ATTRIBUTES
        if r[f"{attribute}_a"] != r[f"{attribute}_b"]
    ]
    return pd.DataFrame(rows, columns=list(DISAGREEMENT_COLUMNS))


def read_disagreements(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(DISAGREEMENT_COLUMNS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_disagreements(path: str | Path, current: pd.DataFrame) -> pd.DataFrame:
    """Write the disagreement sheet, keeping ``final_value`` already filled in.

    A stored value is kept only while both annotators' labels are unchanged,
    so a re-labelled item is adjudicated again.
    """
    previous = read_disagreements(path)
    key = ["item_id", "attribute", "annotator_a", "annotator_b"]
    merged = current.drop(columns="final_value").merge(
        previous[[*key, "final_value"]], on=key, how="left"
    )
    merged["final_value"] = merged["final_value"].fillna("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged[list(DISAGREEMENT_COLUMNS)].to_csv(path, index=False)
    return merged[list(DISAGREEMENT_COLUMNS)]


def assemble_gold(
    manifest: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Overwrite gold columns of ``manifest`` from the annotations.

    Per item and attribute: the agreed value; else the adjudicated
    ``final_value``; else unresolved (gold left empty, i.e. missing, and
    reported). An item labelled by one annotator only takes that label and
    ``gt_source = annotated_single``. Items nobody labelled keep their
    metadata gold.

    Returns the updated manifest and the list of unresolved pairs.
    """
    a = validate_labels(a).set_index("item_id")
    b = validate_labels(b).set_index("item_id")
    adjudicated = {
        (r["item_id"], r["attribute"]): r["final_value"]
        for _, r in disagreements.iterrows()
        if r["final_value"]
    }
    for (_, attribute), value in adjudicated.items():
        tx.validate_gold(attribute, value)

    out = manifest.copy()
    unresolved: list[dict[str, str]] = []
    for idx, row in out.iterrows():
        item_id = row["item_id"]
        in_a, in_b = item_id in a.index, item_id in b.index
        if not (in_a or in_b):
            continue
        if in_a and in_b:
            out.at[idx, "gt_source"] = "annotated"
            for attribute in tx.ATTRIBUTES:
                va, vb = a.at[item_id, attribute], b.at[item_id, attribute]
                if va == vb:
                    value = va
                elif (item_id, attribute) in adjudicated:
                    value = adjudicated[(item_id, attribute)]
                else:
                    value = ""
                    unresolved.append({"item_id": item_id, "attribute": attribute,
                                       "annotator_a": va, "annotator_b": vb})
                out.at[idx, f"gt_{attribute}"] = value
        else:
            single = a if in_a else b
            out.at[idx, "gt_source"] = "annotated_single"
            for attribute in tx.ATTRIBUTES:
                out.at[idx, f"gt_{attribute}"] = single.at[item_id, attribute]
    return out, unresolved


def gold_counts(manifest: pd.DataFrame) -> dict[str, Any]:
    """Per-split value counts of each gold attribute and the material unknown share."""
    out: dict[str, Any] = {}
    for split, frame in manifest.groupby("split", sort=False):
        material = frame["gt_material"]
        labelled = material[material != ""]
        out[split] = {
            "gt_source": frame["gt_source"].value_counts().to_dict(),
            **{
                f"gt_{a}": {k or "<missing>": int(v) for k, v in frame[f"gt_{a}"].value_counts().items()}
                for a in tx.ATTRIBUTES
            },
            "material_unknown_share": (
                float((labelled == tx.UNKNOWN).mean()) if len(labelled) else float("nan")
            ),
        }
    return out


def build_ground_truth(
    cfg: Mapping[str, Any],
    splits: Sequence[str] = ("dev", "test"),
    annotators: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Agreement, disagreement sheets and gold for every split; write outputs.

    Splits with no annotations yet are reported as such and keep metadata gold.
    """
    from groundedgen import data

    project_dir = Path(cfg["project_dir"])
    ann_a, ann_b = annotators or cfg["annotation"]["annotators"]
    manifest = data.read_manifest(project_dir / "data" / "manifest.csv")
    report: dict[str, Any] = {
        "annotators": [ann_a, ann_b],
        "guidelines_version": cfg["annotation"]["guidelines_version"],
        "agreement": {},
        "unresolved": {},
    }
    for split in splits:
        a = read_annotations(annotation_path(project_dir, split, ann_a))
        b = read_annotations(annotation_path(project_dir, split, ann_b))
        if a.empty and b.empty:
            report["agreement"][split] = None
            continue
        sheet = write_disagreements(disagreement_path(project_dir, split), find_disagreements(a, b))
        if not a.empty and not b.empty:
            agreement = agreement_report(a, b)
            agreement["material_kappa_below_threshold"] = bool(
                agreement["material"]["kappa"] < cfg["annotation"]["material_kappa_min"]
            )
            report["agreement"][split] = agreement
        part = manifest["split"] == split
        updated, unresolved = assemble_gold(manifest[part], a, b, sheet)
        manifest.loc[part] = updated
        report["unresolved"][split] = unresolved

    report["counts"] = gold_counts(manifest)
    report["created"] = utils.utc_now()
    manifest.to_csv(project_dir / "data" / "manifest_final.csv", index=False)
    utils.write_json(project_dir / "data" / "ground_truth_report.json", report)
    return manifest, report
