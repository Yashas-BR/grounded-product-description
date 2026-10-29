"""Paper tables and figures from cached outputs (spec §13).

M9 part: T3, T4, T5, T7 and F3, F4, plus the paired tests and the RQ1 origin
test, written to ``outputs/analysis/`` as CSV + LaTeX (booktabs) and PDF + PNG,
and their headline numbers to ``numbers.json``. M11 adds T1, T6, T8 and F5.

Figures follow one palette: categorical slots in fixed order (validated for
colour-vision deficiency), thin marks, recessive grid, ink-coloured text, and
direct labels wherever a colour's contrast is below 3:1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from groundedgen import decompose, extract, utils
from groundedgen import taxonomy as tx
from groundedgen.eval_extraction import write_table
from groundedgen.generate import CONDITIONS

# Reference categorical palette, fixed order (validated: scripts/validate_palette.js).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e1"
CONDITION_COLOR = dict(zip(CONDITIONS, SERIES))
ORIGIN_COLOR = dict(zip(decompose.ORIGINS, SERIES))


def short(name: str | None) -> str:
    """Display name of a model id (``Qwen/Qwen2.5-7B-Instruct`` → ``Qwen2.5-7B-Instruct``)."""
    return extract.model_slug(name).split("__")[-1] if name else ""


def fmt(ci: Any, digits: int = 3) -> str:
    """``0.123 [0.100, 0.150]`` from a bootstrap dict (blank if absent)."""
    if not isinstance(ci, Mapping):
        return ""
    return f"{ci['estimate']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {c: i for i, c in enumerate(CONDITIONS)}
    attr = {a: i for i, a in enumerate(tx.ATTRIBUTES)}
    return frame.assign(_c=frame["condition"].map(order), _a=frame["attribute"].map(attr)) \
        .sort_values(["_c", "generator", "extractor", "_a"]).drop(columns=["_c", "_a"])


def _base_columns(r: pd.Series) -> dict[str, str]:
    return {"condition": r["condition"], "generator": short(r["generator"]),
            "extractor": short(r["extractor"]), "attribute": r["attribute"]}


def table_t3(rates: pd.DataFrame) -> pd.DataFrame:
    """T3: HR, coverage and precision per attribute for all seven conditions."""
    return pd.DataFrame([{**_base_columns(r), "N": r["N"], "HR": fmt(r["HR"]),
                          "coverage": fmt(r["coverage"]), "precision": fmt(r["precision"])}
                         for _, r in _ordered(rates).iterrows()])


def table_t4(rates: pd.DataFrame) -> pd.DataFrame:
    """T4: EP, GD, GF rates and extraction share for B, B-Abs and C."""
    sub = rates[rates["condition"].isin(["B", "B-Abs", "C"])]
    return pd.DataFrame([{**_base_columns(r), **{o: fmt(r[o]) for o in decompose.ORIGINS},
                          "extraction_share": fmt(r["extraction_share"])}
                         for _, r in _ordered(sub).iterrows()])


def table_t5(rates: pd.DataFrame) -> pd.DataFrame:
    """T5: leakage and inflation per LLM for B-Abs, APG and C-Mask."""
    sub = rates[rates["condition"].isin(["B-Abs", "APG", "C-Mask"])]
    return pd.DataFrame([{**_base_columns(r), "leakage": fmt(r["leakage"]),
                          "inflation": fmt(r["inflation"])}
                         for _, r in _ordered(sub).iterrows()])


def table_t7(pairs: pd.DataFrame, extraction_summary: Mapping[str, Any] | None,
             boot: Mapping[str, Any]) -> pd.DataFrame:
    """T7: western vs ethnic wear — extraction accuracy and HR per condition."""
    rows = []
    if extraction_summary:
        for model, by_attr in extraction_summary.get("results", {}).items():
            for attribute, m in by_attr.items():
                groups = m.get("accuracy_by_group", {})
                rows.append({"measure": "extraction accuracy", "condition": "", "model": short(model),
                             "attribute": attribute,
                             "western": f"{groups.get('western', float('nan')):.3f}",
                             "ethnic": f"{groups.get('ethnic', float('nan')):.3f}"})
    for (condition, generator, attribute), g in pairs.groupby(["condition", "generator", "attribute"]):
        cells = {}
        for flag, sub in g.groupby("is_ethnic"):
            hall = (sub["outcome"] == "hallucination").astype(float)
            res = decompose.metrics.bootstrap_ratio(hall, pd.Series(1.0, index=sub.index),
                                                    sub["item_id"], **boot)
            cells["ethnic" if flag else "western"] = fmt(res.as_dict())
        rows.append({"measure": "HR", "condition": condition, "model": short(generator),
                     "attribute": attribute, "western": cells.get("western", ""),
                     "ethnic": cells.get("ethnic", "")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def plot_f3(rates: pd.DataFrame, out_dir: str | Path, generator: str, extractor: str) -> None:
    """F3: hallucinations split into EP / GD / GF per attribute and condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["B", "B-Abs", "APG", "C", "C-Mask"]
    fig, axes = plt.subplots(1, len(tx.ATTRIBUTES), figsize=(10, 3.2), sharey=True,
                             facecolor=SURFACE)
    for ax, attribute in zip(axes, tx.ATTRIBUTES):
        _style(ax)
        sub = rates[(rates["attribute"] == attribute) & (rates["generator"] == generator)]
        for i, condition in enumerate(conditions):
            row = sub[(sub["condition"] == condition)
                      & ((sub["extractor"] == extractor) | (sub["extractor"] == ""))]
            if row.empty:
                continue
            bottom = 0.0
            for origin in decompose.ORIGINS:
                value = row.iloc[0][origin]["estimate"] if isinstance(row.iloc[0][origin], Mapping) else 0.0
                ax.bar(i, value, bottom=bottom, width=0.6, color=ORIGIN_COLOR[origin],
                       edgecolor=SURFACE, linewidth=2,
                       label=origin if attribute == tx.ATTRIBUTES[0] and i == 0 else None)
                bottom += value
        ax.set_xticks(range(len(conditions)), conditions, fontsize=8, color=INK_2)
        ax.set_title(attribute, fontsize=9, color=INK)
    axes[0].set_ylabel("share of scored pairs", fontsize=8, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORIGIN_COLOR[o]) for o in decompose.ORIGINS]
    labels = ["EP (vision: copied wrong evidence)", "GD (language: deviated)",
              "GF (language: filled an unknown)"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.06))
    _save(fig, Path(out_dir), "F3_origins")
    plt.close(fig)


def plot_f4(rates: pd.DataFrame, out_dir: str | Path, llm: str, vlm: str, extractor: str) -> None:
    """F4: coverage (x) vs precision (y), one point per condition, CI whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tx.ATTRIBUTES), figsize=(10, 3.4), sharey=True,
                             facecolor=SURFACE)
    for ax, attribute in zip(axes, tx.ATTRIBUTES):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        placed: list[tuple[float, float, list[str]]] = []  # label clusters
        for condition in CONDITIONS:
            gen = vlm if condition in ("A", "A-F") else llm
            sel = ((rates["condition"] == condition) & (rates["attribute"] == attribute)
                   & (rates["generator"] == gen))
            if condition in ("B", "B-Abs", "APG"):
                sel &= rates["extractor"] == extractor
            if not sel.any():
                continue
            r = rates[sel].iloc[0]
            cov, prec = r["coverage"], r["precision"]
            if not (isinstance(cov, Mapping) and isinstance(prec, Mapping)) or pd.isna(prec["estimate"]):
                continue
            color = CONDITION_COLOR[condition]
            ax.errorbar(cov["estimate"], prec["estimate"],
                        xerr=[[cov["estimate"] - cov["low"]], [cov["high"] - cov["estimate"]]],
                        yerr=[[prec["estimate"] - prec["low"]], [prec["high"] - prec["estimate"]]],
                        fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0,
                        markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            x, y = cov["estimate"], prec["estimate"]
            near = next((c for c in placed if abs(c[0] - x) < 0.04 and abs(c[1] - y) < 0.04), None)
            if near:
                near[2].append(condition)  # coincident points share one label
            else:
                placed.append((x, y, [condition]))
        for x, y, names in placed:
            right_edge = x > 0.8
            ax.annotate(", ".join(names), (x, y), xytext=(-5 if right_edge else 5, 5),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        ha="right" if right_edge else "left")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.12)  # headroom so labels at precision 1.0 clear the title
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(attribute, fontsize=9, color=INK, pad=8)
        ax.set_xlabel("coverage (attribute stated)", fontsize=8, color=INK_2)
    axes[0].set_ylabel("precision (stated = gold)", fontsize=8, color=INK_2)
    _save(fig, Path(out_dir), "F4_tradeoff")
    plt.close(fig)


# --------------------------------------------------------------------------
# M9 entry point
# --------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, float) and value != value:
        return None
    return value


def run_evaluation(cfg: Mapping[str, Any], split: str = "test") -> dict[str, Any]:
    """M9: pairs, rates, tests, T3/T4/T5/T7, F3/F4 and numbers.json for ``split``.

    On test, ``outputs/frozen.json`` must exist (nothing in M9 may run on test
    before M8).
    """
    from groundedgen import generate

    project_dir = Path(cfg["project_dir"])
    if split == "test" and not generate.frozen_path(project_dir).exists():
        raise RuntimeError("M9 on test needs outputs/frozen.json (M8)")
    boot = dict(n_resamples=cfg["metrics"]["bootstrap_resamples"], ci=cfg["metrics"]["ci_level"],
                seed=cfg["metrics"]["seed"])
    out_dir = project_dir / "outputs" / "analysis"
    summary_path = project_dir / "outputs" / "extraction" / "evaluation.json"
    extraction_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    tuning_path = project_dir / "outputs" / "extraction" / "dev_tuning.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8")) if tuning_path.exists() else {}
    primary_extractor = ((extraction_summary or {}).get("primary_extractor")
                         or tuning.get("primary_extractor") or cfg["extraction"]["models"][0])
    llms = [cfg["generation"]["llms"][r] for r in ("primary", "secondary")]
    vlm = cfg["generation"]["vlms"]["primary"]

    pairs = decompose.load_pairs(cfg, split)
    decompose.write_pairs(cfg, pairs, split)
    rates = decompose.group_rates(pairs, boot)
    descriptions = decompose.description_rates(pairs, boot)
    tests = decompose.planned_tests(pairs, primary_extractor, llms, vlm, cfg["metrics"]["alpha"])
    origin = decompose.origin_test(pairs, primary_extractor, llms, boot)

    suffix = "" if split == "test" else f"_{split}"
    write_table(table_t3(rates), out_dir, f"T3_main{suffix}")
    write_table(table_t4(rates), out_dir, f"T4_origins{suffix}")
    write_table(table_t5(rates), out_dir, f"T5_abstention{suffix}")
    write_table(table_t7(pairs, extraction_summary, boot), out_dir, f"T7_ethnic{suffix}")
    tests.to_csv(out_dir / f"tests_mcnemar{suffix}.csv", index=False)
    plot_f3(rates, out_dir, llms[0], primary_extractor)
    plot_f4(rates, out_dir, llms[0], vlm, primary_extractor)

    numbers = {
        "split": split,
        "primary_extractor": primary_extractor,
        "rates": {f"{r['condition']}|{short(r['generator'])}|{short(r['extractor'])}|{r['attribute']}":
                  _plain({k: r[k] for k in ("N", "HR", "coverage", "precision", "EP", "GD", "GF",
                                            "extraction_share", "leakage", "inflation", "lucky")
                          if isinstance(r[k], Mapping) or k == "N"})
                  for _, r in rates.iterrows()},
        "description_any_hallucination": {
            f"{r['condition']}|{short(r['generator'])}|{short(r['extractor'])}": _plain(r["any_hallucination"])
            for _, r in descriptions.iterrows()},
        "rq1_gd_diff_B_minus_C": {f"{short(r['llm'])}|{r['attribute']}": _plain(r["gd_diff_B_minus_C"])
                                  for _, r in origin.iterrows()},
    }
    numbers_path = out_dir / f"numbers{suffix}.json"
    utils.write_json(numbers_path, numbers)
    return {"n_pairs": int(len(pairs)), "numbers": str(numbers_path)}


# --------------------------------------------------------------------------
# M11: T1, T6, T8, F5 and the one-call build
# --------------------------------------------------------------------------

def table_t1(project_dir: str | Path) -> pd.DataFrame:
    """T1: per split, kappa (raw agreement) per attribute, gold counts per
    category / color / material and the material unknown share."""
    project_dir = Path(project_dir)
    report = json.loads((project_dir / "data" / "ground_truth_report.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(project_dir / "data" / "manifest_final.csv", keep_default_na=False)
    splits = [s for s in ("dev", "test") if s in set(manifest["split"])]
    rows = []
    for attribute in tx.ATTRIBUTES:
        cells = {}
        for s in splits:
            a = (report["agreement"].get(s) or {}).get(attribute)
            cells[s] = f"{a['kappa']:.2f} ({a['raw_agreement']:.2f})" if a else ""
        rows.append({"section": "kappa (raw agreement)", "label": attribute, **cells})
    for attribute in tx.ATTRIBUTES:
        for label in tx.gold_values(attribute):
            counts = {s: int((manifest.loc[manifest["split"] == s, f"gt_{attribute}"] == label).sum())
                      for s in splits}
            if any(counts.values()):
                rows.append({"section": attribute, "label": label,
                             **{s: str(v) for s, v in counts.items()}})
    rows.append({"section": "material", "label": "unknown share",
                 **{s: f"{report['counts'][s]['material_unknown_share']:.3f}" for s in splits}})
    rows.append({"section": "items", "label": "annotated",
                 **{s: str(int((manifest["split"] == s).sum())) for s in splits}})
    return pd.DataFrame(rows)


def _filled(sheet: Path) -> bool:
    """A returned labelling CSV (from the page) has no ``description`` column."""
    return sheet.exists() and "description" not in pd.read_csv(sheet, nrows=0).columns


def validation_numbers(cfg: Mapping[str, Any], split: str = "test") -> dict[str, Any]:
    """Claim-extractor and judge validation (when the sheets are back) and the
    APG fallback rate per LLM x extractor on ``split``."""
    from groundedgen import claims, data, judge

    project_dir = Path(cfg["project_dir"])
    out: dict[str, Any] = {}
    cdir = project_dir.joinpath(*claims.VALIDATION_DIR)
    if all(_filled(cdir / f"claims_validation_{w}.csv") for w in ("a", "b")):
        out["claims"] = claims.score_validation(
            cdir, target=cfg["claims_validation"]["target_precision_recall"])
    jdir = project_dir / "outputs" / "validation" / "judge"
    if all(_filled(jdir / f"judge_validation_{w}.csv") for w in ("a", "b")):
        out["judge"] = judge.score_validation(jdir, judge.judged_descriptions(cfg, split))
    ids = set(data.load_manifest(cfg, splits=[split])["item_id"])
    fallback = {}
    for path in sorted((project_dir / "outputs" / "generation" / "APG").glob("*.jsonl")):
        records = [r for r in utils.read_jsonl(path) if r["item_id"] in ids]
        if records:
            used = [r["apg"]["fallback_used"] for r in records]
            fallback[f"{short(records[0]['generator'])}|{short(records[0]['extractor'])}"] = {
                "n": len(used), "rate": sum(used) / len(used)}
    out["apg_fallback"] = fallback
    return out


def table_t8(numbers: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    if "claims" in numbers:
        for attribute in tx.ATTRIBUTES:
            v = numbers["claims"][attribute].get("vs_agreed") or numbers["claims"][attribute]["vs_a"]
            rows.append({"measure": "claim extractor", "scope": attribute,
                         "precision": f"{v['precision']:.3f}", "recall": f"{v['recall']:.3f}",
                         "n": numbers["claims"]["n_descriptions"]})
    if "judge" in numbers:
        for who in ("a", "b"):
            v = numbers["judge"][f"vs_{who}"]
            rows.append({"measure": "LLM judge", "scope": f"vs labeller {who}",
                         "precision": f"{v['precision']:.3f}", "recall": f"{v['recall']:.3f}",
                         "n": numbers["judge"]["n_descriptions"]})
        rows.append({"measure": "LLM judge", "scope": "human kappa (any claim)",
                     "precision": f"{numbers['judge']['human_kappa_any_claim']:.3f}", "recall": "",
                     "n": numbers["judge"]["n_descriptions"]})
    for scope, v in numbers.get("apg_fallback", {}).items():
        rows.append({"measure": "APG fallback rate", "scope": scope,
                     "precision": f"{v['rate']:.3f}", "recall": "", "n": v["n"]})
    return pd.DataFrame(rows, columns=["measure", "scope", "precision", "recall", "n"])


def _segments(text: str, spans: Sequence[Sequence[int]]) -> list[tuple[str, bool]]:
    """Split ``text`` into (piece, is_error) runs."""
    out, pos = [], 0
    for start, end in sorted(spans):
        if start < pos:
            continue
        if start > pos:
            out.append((text[pos:start], False))
        out.append((text[start:end], True))
        pos = end
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _wrap_segments(segments, width: int) -> list[list[tuple[str, bool]]]:
    """Greedy word wrap that keeps the error flag of every word."""
    words = [(token, err) for piece, err in segments
             for token in re.split(r"(\s+)", piece) if token]
    lines, line, n = [], [], 0
    for token, err in words:
        if token.isspace():
            if line:
                line.append((" ", False))
                n += 1
            continue
        if n + len(token) > width and line:
            while line and line[-1][0] == " ":
                line.pop()
            lines.append(line)
            line, n = [], 0
        line.append((token, err))
        n += len(token)
    if line:
        lines.append(line)
    return lines


def choose_examples(pairs: pd.DataFrame, vlm: str, llm: str, extractor: str, k: int = 3) -> list[str]:
    """Items where the image-only description hallucinates and APG does not,
    with one ethnic-wear item if there is one."""
    scored = pairs[pairs["mask_attr"].isna()]

    def errors(condition, generator, ext):
        sub = scored[(scored["condition"] == condition) & (scored["generator"] == generator)]
        if ext is not None:
            sub = sub[sub["extractor"] == ext]
        return sub.groupby("item_id")["outcome"].apply(lambda s: int((s == "hallucination").sum()))

    af, apg = errors("A-F", vlm, None), errors("APG", llm, extractor)
    good = sorted(i for i in af.index if af[i] > 0 and apg.get(i, 1) == 0) or sorted(af.index)
    ethnic = set(scored.loc[scored["is_ethnic"].astype(str) == "True", "item_id"])
    picked = [i for i in good if i in ethnic][:1]
    picked += [i for i in good if i not in picked][: k - len(picked)]
    return picked


def plot_f5(cfg: Mapping[str, Any], pairs: pd.DataFrame, out_dir: str | Path,
            vlm: str, llm: str, extractor: str, k: int = 3, name: str = "F5_examples") -> list[str]:
    """F5: image, evidence and the A-F / B-Abs / APG descriptions with
    hallucinated attribute mentions marked in red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from groundedgen import claims, data, generate

    project_dir = Path(cfg["project_dir"])
    items = choose_examples(pairs, vlm, llm, extractor, k)
    manifest = data.load_manifest(cfg).set_index("item_id")
    sources = {"A-F": (vlm, None), "B-Abs": (llm, extractor), "APG": (llm, extractor)}
    records = {}
    for condition, (generator, ext) in sources.items():
        path = generate.generation_path(project_dir, condition, generator, ext)
        records[condition] = {r["item_id"]: r for r in utils.read_jsonl(path) if r.get("mask_attr") is None}
    halluc = pairs[(pairs["outcome"] == "hallucination") & pairs["mask_attr"].isna()]
    fig = plt.figure(figsize=(10, 3.4 * len(items)), facecolor=SURFACE)
    grid = fig.add_gridspec(len(items), 2, width_ratios=[1, 3.2], hspace=0.25, wspace=0.04)
    for row, item in enumerate(items):
        ax_img = fig.add_subplot(grid[row, 0])
        ax_img.imshow(Image.open(project_dir / manifest.at[item, "image_path"]))
        ax_img.set_axis_off()
        gold = ", ".join(str(manifest.at[item, f"gt_{a}"]) for a in tx.ATTRIBUTES)
        ax_img.set_title(f"{item}\ngold: {gold}", fontsize=7.5, color=INK_2)
        ax = fig.add_subplot(grid[row, 1])
        ax.set_axis_off()
        evidence = records["B-Abs"].get(item, {}).get("evidence") or {}
        lines = [([("Evidence (B-Abs, APG): " + ", ".join(f"{a}={v}" for a, v in evidence.items()), False)],
                  "bold", INK_2)]
        for condition, (generator, ext) in sources.items():
            record = records[condition].get(item)
            if record is None:
                continue
            bad = halluc[(halluc["item_id"] == item) & (halluc["condition"] == condition)
                         & (halluc["generator"] == generator)]
            if ext is not None:
                bad = bad[bad["extractor"] == ext]
            wrong = set(bad["attribute"])
            spans = [(m.start, m.end) for m in claims.find_mentions(record["text"])
                     if m.attribute in wrong and m.counted
                     and m.label != manifest.at[item, f"gt_{m.attribute}"]]
            lines.append(([(f"{condition}:", False)], "bold", CONDITION_COLOR.get(condition, INK)))
            lines += [(w, "normal", INK) for w in _wrap_segments(_segments(record["text"], spans), 95)[:6]]
        y = 1.0
        for segments, weight, color in lines:
            anchor = None
            for piece, err in segments:
                style = dict(fontsize=8, fontweight="bold" if err else weight,
                             color="#c0262d" if err else color)
                if err:
                    style["bbox"] = dict(boxstyle="square,pad=0.05", fc="#fde2e1", ec="none")
                if anchor is None:
                    anchor = ax.text(0.0, y, piece, transform=ax.transAxes, va="top", **style)
                else:
                    anchor = ax.annotate(piece, xy=(1, 1), xycoords=anchor, va="top", **style)
            y -= 0.075 if weight == "normal" else 0.085
    fig.text(0.01, 0.005, "Red: attribute mention whose value differs from the gold label (hallucination).",
             fontsize=7.5, color=INK_2)
    _save(fig, Path(out_dir), name)
    plt.close(fig)
    return items


def build_all(cfg: Mapping[str, Any], split: str = "test", with_extraction: bool = True) -> dict[str, Any]:
    """M11: regenerate every table and figure (T1-T8, F2-F5) and numbers.json
    from cached outputs in one call."""
    from groundedgen import claims, data, eval_extraction, human_eval

    project_dir = Path(cfg["project_dir"])
    out_dir = project_dir / "outputs" / "analysis"
    claims.rebuild_all(project_dir)  # current lexicon rules for every description
    if with_extraction and split == "test":
        eval_extraction.run_evaluation(cfg, data.load_manifest(cfg))  # T2, F2
    m9 = run_evaluation(cfg, split)  # T3, T4, T5, T7, F3, F4
    suffix = "" if split == "test" else f"_{split}"
    numbers_path = out_dir / f"numbers{suffix}.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8"))

    write_table(table_t1(project_dir), out_dir, "T1_data")  # covers both splits
    report = json.loads((project_dir / "data" / "ground_truth_report.json").read_text(encoding="utf-8"))
    numbers["annotation"] = _plain({"agreement": report["agreement"], "counts": report["counts"]})

    validation = validation_numbers(cfg, split)
    write_table(table_t8(validation), out_dir, f"T8_validation{suffix}")
    numbers["validation"] = _plain(validation)

    built = {"T6": False}
    raters = [r for r in human_eval.RATERS if (human_eval.out_dir(cfg) / f"ratings_{r}.csv").exists()]
    if split == "test" and len(raters) >= 2:  # alpha needs at least two raters
        numbers["human_eval"] = _plain(human_eval.run_analysis(cfg, raters))
        built["T6"] = True
    pairs = pd.read_csv(out_dir / f"pairs_{split}.csv", keep_default_na=False).replace({"": None})
    built["F5_items"] = plot_f5(cfg, pairs, out_dir, cfg["generation"]["vlms"]["primary"],
                                cfg["generation"]["llms"]["primary"], numbers["primary_extractor"],
                                name=f"F5_examples{suffix}")
    utils.write_json(numbers_path, numbers)
    return {**m9, "built": built}


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.analysis [extra.yaml ...] --split test``."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--m9-only", action="store_true", help="only T3/T4/T5/T7, F3/F4")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    result = run_evaluation(cfg, args.split) if args.m9_only else build_all(cfg, args.split)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
