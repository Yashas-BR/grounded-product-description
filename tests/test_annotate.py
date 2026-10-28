from __future__ import annotations

import pandas as pd
import pytest

from groundedgen import annotate


def _items(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": [f"fpi_{i}" for i in range(n)],
        "image_path": [f"data/images/dev/fpi_{i}.jpg" for i in range(n)],
        "meta_category": ["shirt", "kurta", "jeans", "dress"][:n],
        "meta_color": ["blue", "red", "blue", "black"][:n],
    })


LABELS = {"category": "shirt", "color": "blue", "material": "cotton"}


def test_store_writes_csv_after_every_save(tmp_path):
    path = annotate.annotation_path(tmp_path, "dev", "a")
    store = annotate.AnnotationStore(path, "a")
    store.save("fpi_0", LABELS, notes=" matte weave ")
    frame = annotate.read_annotations(path)
    assert list(frame.columns) == list(annotate.ANNOTATION_COLUMNS)
    assert frame.iloc[0].to_dict() | {"timestamp": ""} == {
        "item_id": "fpi_0", "annotator": "a", "category": "shirt", "color": "blue",
        "material": "cotton", "notes": "matte weave", "timestamp": "",
    }
    store.save("fpi_0", LABELS | {"material": "unknown"})  # re-save overwrites
    frame = annotate.read_annotations(path)
    assert len(frame) == 1 and frame.iloc[0]["material"] == "unknown"


def test_store_rejects_invalid_labels(tmp_path):
    store = annotate.AnnotationStore(tmp_path / "x.csv", "a")
    with pytest.raises(ValueError):
        store.save("fpi_0", LABELS | {"material": "exclude"})
    with pytest.raises(ValueError):
        store.save("fpi_0", LABELS | {"color": "teal"})


def test_session_resumes_at_first_unannotated(tmp_path):
    path = tmp_path / "dev_a.csv"
    store = annotate.AnnotationStore(path, "a")
    store.save("fpi_0", LABELS)
    store.save("fpi_1", LABELS)
    session = annotate.AnnotationSession(_items(), annotate.AnnotationStore(path, "a"))
    assert session.index == 2 and session.n_done == 2


def test_session_prefill_and_blocked_save(tmp_path):
    session = annotate.AnnotationSession(_items(), annotate.AnnotationStore(tmp_path / "s.csv", "a"))
    values = session.prefill()
    assert values["category"] == "shirt" and values["color"] == "blue"
    assert values["material"] is None  # material starts blank
    assert not session.save_and_next(values)  # blocked: material missing
    assert session.index == 0 and session.n_done == 0
    assert session.save_and_next(values | {"material": "unknown"})
    assert session.index == 1
    session.prev()
    assert session.prefill()["material"] == "unknown"  # saved values shown


def test_session_skip_and_bounds(tmp_path):
    session = annotate.AnnotationSession(_items(2), annotate.AnnotationStore(tmp_path / "s.csv", "a"))
    session.prev()
    assert session.index == 0
    session.skip()
    session.skip()
    assert session.index == 1 and session.n_done == 0


def test_recheck_items_and_start_at_first(tmp_path):
    path = annotate.annotation_path(tmp_path, "dev", "a")
    store = annotate.AnnotationStore(path, "a")
    for i, mat in enumerate(["cotton", "denim", "synthetic", "unknown"]):
        store.save(f"fpi_{i}", LABELS | {"material": mat})
    ids = annotate.recheck_items(tmp_path, "dev", "a")
    assert ids == ["fpi_0", "fpi_2"]
    assert annotate.recheck_items(tmp_path, "dev", "b") == []  # other annotator untouched
    items = _items()
    session = annotate.AnnotationSession(items[items["item_id"].isin(ids)],
                                         annotate.AnnotationStore(path, "a"), start_at_first=True)
    assert len(session) == 2 and session.index == 0
    assert session.prefill()["material"] == "cotton"  # shows own saved label
    session.save_and_next(session.prefill() | {"material": "unknown"})
    assert annotate.read_annotations(path).set_index("item_id").at["fpi_0", "material"] == "unknown"
