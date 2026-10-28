# Decisions

Choices made where the build spec was silent or ambiguous (spec §0.7). Newest
last. Each entry: what, why, where.

## M1 — Foundations

1. **Category raw values match exactly; colors case-insensitively.** The spec
   says baseColour is case-insensitive and says nothing for articleType, so
   articleType is matched after stripping whitespace only. Colors are also
   whitespace-normalised ("Navy  Blue" → "navy blue"). `taxonomy.py`.
2. **Lexicon additions.** "gray" (US spelling) is added as a surface form of
   grey, and "multicolour", "multicoloured", "multicolored", "multi-colored",
   "multi-coloured" as surface forms of `other`, because LLMs write these
   spellings and the spec's intent is to count them. `taxonomy.py`.
3. **Known false-positive risk in the color lexicon.** Raw values such as
   "rose", "steel", "tan" and "cream" are kept as the spec requires; they may
   match non-color uses ("rose print", "steel buttons"). The M7 claim-extractor
   validation is where this is caught and fixed (dev only).
4. **Styling triggers and hedge cues live in `taxonomy.py`.** They are lexical
   rules, and spec §0.4 asks for every label set and rule in one place.
5. **Cohen's kappa lives in `metrics.py`** and `agreement.py` will reuse it.
   When expected agreement is 1 (both annotators used a single label), kappa
   is undefined and returned as NaN; raw agreement is always reported beside it.
6. **AURC handles ties.** Items with equal confidence are accepted or rejected
   together, so the risk at each position inside a tie group is the group's
   risk. AURC is the mean risk over all n positions.
7. **Bootstrap statistics that are undefined in a resample** (zero denominator)
   are NaN and ignored by the percentile computation.
8. **`krippendorff` is imported lazily** inside `metrics.krippendorff_alpha`, so
   the rest of the package and its tests work without it.
9. **Judge `max_new_tokens` is 256**, not 160. The 160 limit is for
   descriptions; a judge listing several claims as JSON can exceed it and would
   then always fail to parse. `configs/default.yaml`.
10. **Tests import the package through `tests/conftest.py`** (adds the repo root
    to `sys.path`), so no install step is needed locally or in Colab.
11. **JSONL reading skips malformed lines with a warning**, which covers a
    line truncated by a Colab disconnect; appending first restores a trailing
    newline so the next record starts on its own line.

## M2 — Data

1. **Too-small images are replaced, not just skipped.** Skipping an image
   after sampling would leave the splits short of 40/200 and unbalanced. Image
   size is therefore checked while drawing: a rejected candidate is replaced
   by the next item in its category's color-diverse order, and a category
   that runs out has its shortfall redistributed. Only candidate images are
   decoded. `data.draw_split`, `data.ImageChecker`.
2. **Ties in allocation go to categories in taxonomy order.** When the split
   size does not divide evenly (200 / 13), the first categories in
   `taxonomy.CATEGORIES` get one extra item. Deterministic, independent of
   seed. `data.allocate`.
3. **Color groups and their order are drawn from one RNG per split**,
   processing categories in taxonomy order, so a split is reproducible from
   its seed alone. `data.draw_split`.
4. **Dropped colors are reported in two groups**: explicitly excluded raw
   colors and other unmapped values (missing values appear as `<missing>`).
   Only rows whose articleType maps are counted. `data.filter_metadata`.
5. **The data stage is skipped when `data/manifest.csv` exists** (resumable);
   `overwrite=True` rebuilds it. Existing JPEGs are never rewritten.
6. **`data.hf_revision` is in the config** so the dataset can be pinned to a
   commit; it is logged in the dataset report and `run_info.json`.
7. **Images come from `benitomartin/fashion-product-images-small-384x512`**,
   pinned to commit `2bb740f8dddd4ee43738bb2b6b4ed4dbb07a3bbe`. Every image in
   the spec's `ashraq/fashion-product-images-small` is 60×80 px, so the 64 px
   rule would drop all of them, and material cues (weave, knit loops) are not
   visible at that size. The mirror has the same 44,072 rows, ids and
   metadata at 384×512, and a sampled image shows real detail rather than an
   upscaled thumbnail. 384×512 matches SigLIP's 384 px input, downloads in
   about 2 GB on Colab, and keeps the 64 px rule meaningful. Its card gives no
   provenance or license, so the paper cites the original Kaggle Fashion
   Product Images dataset and releases item ids only. The 900×1200 mirror and
   the 25 GB Kaggle original were rejected as heavier for no gain at CLIP's
   224 px and SigLIP's 384 px input sizes.
