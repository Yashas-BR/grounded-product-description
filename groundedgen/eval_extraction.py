"""Extraction evaluation: dev thresholds, test metrics, T2 and F2 (spec §7).

Correctness per (item, attribute):

* gold a real label → determinable; the prediction is correct if it equals gold;
* gold ``unknown`` (material only) → non-determinable; any assertion is wrong,
  only abstaining is correct;
* gold ``exclude`` or missing → skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from groundedgen import extract, metrics, utils
from groundedgen import taxonomy as tx


def scored_pairs(records: Sequence[Mapping[str, Any]], manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per scored (item, attribute) with gold, prediction and correctness."""
    gold = manifest.set_index("item_id")
    rows = []
    for rec in records:
        if rec["item_id"] not in gold.index:
            continue
        item = gold.loc[rec["item_id"]]
        for attribute in tx.ATTRIBUTES:
            g = item[f"gt_{attribute}"]
            if g in ("", tx.EXCLUDE) or (isinstance(g, float) and math.isnan(g)):
                continue
            attr = rec["attributes"][attribute]
            determinable = tx.is_label(attribute, g)
            rows.append({
                "item_id": rec["item_id"],
                "attribute": attribute,
                "gold": g,
                "pred": attr["pred"],
                "confidence": attr["confidence"],
                "determinable": determinable,
                "correct": determinable and attr["pred"] == g,
                "is_ethnic": bool(item.get("is_ethnic", False)),
            })
    return pd.DataFrame(rows, columns=["item_id", "attribute", "gold", "pred", "confidence",
                                       "determinable", "correct", "is_ethnic"])


def select_threshold(confidence: Sequence[float], correct: Sequence[bool],
                     target_accuracy: float = 0.90) -> float:
    """Lowest t such that accuracy on {confidence ≥ t} is at least the target.

    This is maximum coverage under the accuracy constraint. Returns infinity
    (always abstain) if no threshold reaches the target.
    """
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    best = math.inf
    for t in np.unique(conf):  # ascending
        kept = conf >= t
        if ok[kept].mean() >= target_accuracy:
            best = float(t)
            break
    return best


def tune_thresholds(pairs: pd.DataFrame, target_accuracy: float = 0.90) -> dict[str, float]:
    """Per-attribute thresholds on dev pairs."""
    out = {}
    for attribute in tx.ATTRIBUTES:
        sub = pairs[pairs["attribute"] == attribute]
        out[attribute] = select_threshold(sub["confidence"], sub["correct"], target_accuracy)
    return out


def thresholds_path(project_dir: str | Path, model_name: str) -> Path:
    return Path(project_dir) / "outputs" / "extraction" / extract.model_slug(model_name) / "thresholds.json"


def save_thresholds(path: str | Path, thresholds: Mapping[str, float]) -> None:
    """Write thresholds; infinity is stored as ``null`` (always abstain)."""
    utils.write_json(path, {a: (None if math.isinf(t) else t) for a, t in thresholds.items()})


def load_thresholds(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {a: (math.inf if t is None else float(t)) for a, t in raw.items()}


def attribute_metrics(
    pairs: pd.DataFrame,
    threshold: float,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Test metrics for one model and attribute (T2, F2, confusion, ethnic split)."""
    det = pairs[pairs["determinable"]]
    non = pairs[~pairs["determinable"]]
    boot = dict(n_resamples=n_resamples, ci=ci, seed=seed)

    acc = metrics.bootstrap_ratio(det["correct"], np.ones(len(det)), det["item_id"], **boot)
    f1 = metrics.cluster_bootstrap(det, lambda d: metrics.macro_f1(d["gold"], d["pred"]), **boot)
    rc = metrics.risk_coverage(pairs["confidence"], pairs["correct"])
    kept = pairs["confidence"] >= threshold
    by_ethnic = {
        ("ethnic" if flag else "western"): metrics.accuracy(g["gold"], g["pred"])
        for flag, g in det.groupby("is_ethnic")
    }
    labels = list(tx.LABELS[pairs["attribute"].iloc[0]]) if len(pairs) else []
    confusion = pd.crosstab(det["gold"], det["pred"]).reindex(
        index=labels, columns=labels, fill_value=0
    ) if len(det) else pd.DataFrame()
    return {
        "n_pairs": int(len(pairs)),
        "n_determinable": int(len(det)),
        "forced_accuracy": acc.as_dict(),
        "macro_f1": f1.as_dict(),
        "aurc": rc.aurc,
        "risk_coverage": {"coverage": rc.coverage.tolist(), "risk": rc.risk.tolist()},
        "threshold": None if math.isinf(threshold) else threshold,
        "coverage": float(kept.mean()) if len(pairs) else float("nan"),
        "selective_accuracy": float(pairs.loc[kept, "correct"].mean()) if kept.any() else float("nan"),
        "abstention_non_determinable": (
            float((non["confidence"] < threshold).mean()) if len(non) else float("nan")
        ),
        "n_non_determinable": int(len(non)),
        "accuracy_by_group": by_ethnic,
        "confusion": confusion.to_dict(orient="index"),
    }


def evaluate_model(
    pairs: pd.DataFrame, thresholds: Mapping[str, float], **boot: Any
) -> dict[str, dict[str, Any]]:
    """:func:`attribute_metrics` for every attribute of one model."""
    return {
        a: attribute_metrics(pairs[pairs["attribute"] == a], thresholds[a], **boot)
        for a in tx.ATTRIBUTES
    }


def primary_extractor(dev_material_aurc: Mapping[str, float]) -> str:
    """The model with the lower material AURC on dev (ties: first given)."""
    return min(dev_material_aurc, key=lambda m: dev_material_aurc[m])


# --------------------------------------------------------------------------
# Paper assets: T2 and F2
# --------------------------------------------------------------------------

def _fmt_ci(d: Mapping[str, float]) -> str:
    return f"{d['estimate']:.3f} [{d['low']:.3f}, {d['high']:.3f}]"


def table_t2(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> pd.DataFrame:
    """T2: per model and attribute, forced accuracy, macro-F1, AURC, coverage
    and selective accuracy at the dev-tuned threshold."""
    rows = []
    for model, by_attr in results.items():
        for attribute, m in by_attr.items():
            rows.append({
                "model": extract.model_slug(model).split("__")[-1],
                "attribute": attribute,
                "forced_accuracy": _fmt_ci(m["forced_accuracy"]),
                "macro_f1": _fmt_ci(m["macro_f1"]),
                "aurc": round(m["aurc"], 3),
                "threshold": "∞" if m["threshold"] is None else round(m["threshold"], 3),
                "coverage": round(m["coverage"], 3),
                "selective_accuracy": round(m["selective_accuracy"], 3),
                "abstain_non_det": round(m["abstention_non_determinable"], 3),
            })
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, out_dir: str | Path, name: str) -> None:
    """Save a paper table as CSV and LaTeX (booktabs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{name}.csv", index=False)
    latex = frame.to_latex(index=False, escape=True).replace("∞", r"$\infty$")
    (out_dir / f"{name}.tex").write_text(latex, encoding="utf-8")


def plot_f2(results: Mapping[str, Mapping[str, Mapping[str, Any]]], out_dir: str | Path) -> None:
    """F2: risk–coverage curves per attribute, both extractors (PDF and PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tx.ATTRIBUTES), figsize=(10, 3.2), sharey=True)
    for ax, attribute in zip(axes, tx.ATTRIBUTES):
        for model, by_attr in results.items():
            m = by_attr[attribute]
            label = f"{extract.model_slug(model).split('__')[-1]} (AURC {m['aurc']:.3f})"
            ax.step(m["risk_coverage"]["coverage"], m["risk_coverage"]["risk"],
                    where="post", label=label)
        ax.set_title(attribute)
        ax.set_xlabel("coverage")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("risk (error rate)")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"F2_risk_coverage.{ext}", dpi=200)
    plt.close(fig)


def tune_on_dev(cfg: Mapping[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    """Tune and save thresholds on dev, pick the primary extractor, and report
    dev metrics. Touches no test data, so it can run before test annotation.

    Writes ``thresholds.json`` per model and ``outputs/extraction/dev_tuning.json``.
    """
    project_dir = Path(cfg["project_dir"])
    target = cfg["extraction"]["threshold_rule"]["target_accuracy"]
    boot = dict(n_resamples=cfg["metrics"]["bootstrap_resamples"],
                ci=cfg["metrics"]["ci_level"], seed=cfg["metrics"]["seed"])
    thresholds: dict[str, dict[str, float]] = {}
    dev_material_aurc: dict[str, float] = {}
    dev_results: dict[str, Any] = {}
    for model in cfg["extraction"]["models"]:
        dev = scored_pairs(utils.read_jsonl(extract.extraction_path(project_dir, model, "dev")),
                           manifest[manifest["split"] == "dev"])
        thresholds[model] = tune_thresholds(dev, target)
        save_thresholds(thresholds_path(project_dir, model), thresholds[model])
        mat = dev[dev["attribute"] == "material"]
        dev_material_aurc[model] = metrics.risk_coverage(mat["confidence"], mat["correct"]).aurc
        dev_results[model] = evaluate_model(dev, thresholds[model], **boot)
    summary = {
        "primary_extractor": primary_extractor(dev_material_aurc),
        "dev_material_aurc": dev_material_aurc,
        "thresholds": {m: {a: (None if math.isinf(t) else t) for a, t in v.items()}
                       for m, v in thresholds.items()},
        "dev_results": dev_results,
    }
    utils.write_json(project_dir / "outputs" / "extraction" / "dev_tuning.json", summary)
    return summary


def run_evaluation(cfg: Mapping[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    """Tune thresholds on dev, evaluate on test, write thresholds, T2 and F2.

    Expects ``outputs/extraction/{model}/{dev,test}.jsonl`` for every model in
    ``cfg["extraction"]["models"]``. Run it once test gold exists.
    """
    project_dir = Path(cfg["project_dir"])
    boot = dict(n_resamples=cfg["metrics"]["bootstrap_resamples"],
                ci=cfg["metrics"]["ci_level"], seed=cfg["metrics"]["seed"])
    tuned = tune_on_dev(cfg, manifest)
    results: dict[str, Any] = {}
    for model in cfg["extraction"]["models"]:
        test = scored_pairs(utils.read_jsonl(extract.extraction_path(project_dir, model, "test")),
                            manifest[manifest["split"] == "test"])
        thresholds = load_thresholds(thresholds_path(project_dir, model))
        results[model] = evaluate_model(test, thresholds, **boot)

    analysis_dir = project_dir / "outputs" / "analysis"
    write_table(table_t2(results), analysis_dir, "T2_extraction")
    plot_f2(results, analysis_dir)
    summary = {"primary_extractor": tuned["primary_extractor"],
               "dev_material_aurc": tuned["dev_material_aurc"], "results": results}
    utils.write_json(project_dir / "outputs" / "extraction" / "evaluation.json", summary)
    return summary
