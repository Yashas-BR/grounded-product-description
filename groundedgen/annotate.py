"""ipywidgets annotation tool for gold labels (spec §6).

The logic (``AnnotationStore``, ``AnnotationSession``) is plain Python and
unit-tested; ``launch`` only wires it to widgets in Colab.

Output: ``annotations/{split}_{annotator}.csv`` with columns
``item_id, annotator, category, color, material, notes, timestamp``,
rewritten after every save. On launch the tool resumes at the first
unannotated item.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from groundedgen import taxonomy as tx
from groundedgen import utils

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "item_id", "annotator", "category", "color", "material", "notes", "timestamp",
)


def annotation_path(project_dir: str | Path, split: str, annotator: str) -> Path:
    """``annotations/{split}_{annotator}.csv`` under ``project_dir``."""
    return Path(project_dir) / "annotations" / f"{split}_{annotator}.csv"


def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read an annotation CSV (empty frame with the right columns if missing)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(ANNOTATION_COLUMNS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class AnnotationStore:
    """One annotator's labels for one split, persisted as CSV after each save."""

    def __init__(self, path: str | Path, annotator: str):
        self.path = Path(path)
        self.annotator = annotator
        frame = read_annotations(self.path)
        self.rows: dict[str, dict[str, str]] = {
            r["item_id"]: dict(r) for r in frame.to_dict("records")
        }

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.rows

    def get(self, item_id: str) -> dict[str, str] | None:
        return self.rows.get(item_id)

    def save(self, item_id: str, labels: Mapping[str, str], notes: str = "") -> dict[str, str]:
        """Validate, store and rewrite the CSV atomically; return the saved row."""
        for attribute in tx.ATTRIBUTES:
            tx.validate_gold(attribute, labels[attribute])
        row = {
            "item_id": item_id,
            "annotator": self.annotator,
            **{a: labels[a] for a in tx.ATTRIBUTES},
            "notes": notes.strip(),
            "timestamp": utils.utc_now(),
        }
        self.rows[item_id] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        pd.DataFrame(list(self.rows.values()), columns=list(ANNOTATION_COLUMNS)).to_csv(
            tmp, index=False
        )
        os.replace(tmp, self.path)
        return row


class AnnotationSession:
    """Navigation, pre-filling and validation over one split's items.

    Args:
        items: manifest rows for the split, in annotation order.
        store: where labels are saved.
    """

    def __init__(self, items: pd.DataFrame, store: AnnotationStore):
        self.items = items.reset_index(drop=True)
        self.store = store
        self.index = self.first_unannotated()

    def __len__(self) -> int:
        return len(self.items)

    @property
    def item(self) -> pd.Series:
        return self.items.iloc[self.index]

    @property
    def n_done(self) -> int:
        return sum(item_id in self.store for item_id in self.items["item_id"])

    def first_unannotated(self) -> int:
        """Index of the first item without a saved label (last item if all done)."""
        for i, item_id in enumerate(self.items["item_id"]):
            if item_id not in self.store:
                return i
        return max(len(self.items) - 1, 0)

    def prefill(self) -> dict[str, str | None]:
        """Initial dropdown values for the current item.

        A saved annotation is shown as saved. Otherwise category and color
        come from metadata (for verification) and material starts blank.
        """
        saved = self.store.get(self.item["item_id"])
        if saved:
            return {a: saved[a] for a in tx.ATTRIBUTES} | {"notes": saved.get("notes", "")}
        return {
            "category": self.item["meta_category"],
            "color": self.item["meta_color"],
            "material": None,
            "notes": "",
        }

    @staticmethod
    def missing(labels: Mapping[str, str | None]) -> list[str]:
        """Attributes not yet set; saving is blocked while this is non-empty."""
        return [a for a in tx.ATTRIBUTES if not labels.get(a)]

    def save_and_next(self, labels: Mapping[str, str | None], notes: str = "") -> bool:
        """Save the current item and advance; ``False`` (nothing saved) if incomplete."""
        if self.missing(labels):
            return False
        self.store.save(self.item["item_id"], labels, notes)  # type: ignore[arg-type]
        self.next()
        return True

    def next(self) -> None:
        self.index = min(self.index + 1, len(self.items) - 1)

    def prev(self) -> None:
        self.index = max(self.index - 1, 0)

    def skip(self) -> None:
        """Move on without saving (the item stays unannotated)."""
        self.next()


# --------------------------------------------------------------------------
# Widget UI (Colab)
# --------------------------------------------------------------------------

def launch(cfg: Mapping[str, Any], split: str, annotator: str):
    """Show the annotation form for ``split`` and ``annotator`` in Colab.

    Returns the ``AnnotationSession`` so progress can be inspected.
    """
    import ipywidgets as w
    from IPython.display import display

    try:
        from google.colab import output

        output.enable_custom_widget_manager()
    except ImportError:
        pass

    from groundedgen import data

    project_dir = Path(cfg["project_dir"])
    items = data.load_manifest(cfg, splits=[split])
    store = AnnotationStore(annotation_path(project_dir, split, annotator), annotator)
    session = AnnotationSession(items, store)

    image = w.Image(format="jpeg", width=320)
    header = w.HTML()
    status = w.HTML()
    dropdowns = {
        a: w.Dropdown(options=list(tx.gold_values(a)), description=a.capitalize())
        for a in tx.ATTRIBUTES
    }
    notes = w.Textarea(description="Notes", layout=w.Layout(width="320px"))
    prev_btn = w.Button(description="Prev")
    save_btn = w.Button(description="Save & Next", button_style="success")
    skip_btn = w.Button(description="Skip")

    def render() -> None:
        item = session.item
        image.value = (project_dir / item["image_path"]).read_bytes()
        mark = "saved" if item["item_id"] in store else "not annotated"
        header.value = (
            f"<b>{split} · {annotator}</b> — item {session.index + 1}/{len(session)} "
            f"({item['item_id']}, {mark}) · {session.n_done} done"
        )
        values = session.prefill()
        for a, dd in dropdowns.items():
            dd.value = values[a]
        notes.value = values["notes"] or ""
        status.value = ""

    def on_save(_) -> None:
        labels = {a: dd.value for a, dd in dropdowns.items()}
        missing = session.missing(labels)
        if missing:
            status.value = f"<span style='color:#b00'>Set {', '.join(missing)} before saving.</span>"
            return
        session.save_and_next(labels, notes.value)
        render()

    def on_prev(_) -> None:
        session.prev()
        render()

    def on_skip(_) -> None:
        session.skip()
        render()

    save_btn.on_click(on_save)
    prev_btn.on_click(on_prev)
    skip_btn.on_click(on_skip)

    form = w.VBox([header, *dropdowns.values(), notes,
                   w.HBox([prev_btn, save_btn, skip_btn]), status])
    display(w.HBox([image, form]))
    render()
    return session
