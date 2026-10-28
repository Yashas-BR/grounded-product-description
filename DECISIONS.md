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
8. **M2 runs locally, not on Colab.** Colab would not assign a backend from
   the team's location, and the data stage needs no GPU. The local machine
   has about 2 GB of free disk, too little for the `datasets` cache (~4 GB),
   so `data.loader: remote_parquet` (set in `configs/local.yaml`) reads the
   Hub's parquet shards over HTTP: metadata columns by byte range, and image
   bytes only for the row groups that hold candidates, in memory. Images are
   the original parquet bytes; the Hub's rows API was rejected because it
   re-encodes JPEGs (49 KB → 18 KB, pixel RMS 3.1). Row order equals
   `datasets.load_dataset` order, so both loaders give the same splits.
   `python -m groundedgen.data configs/local.yaml` runs the stage. First
   real run (2026-10-28): 17,162 items after filtering, 200 test + 40 dev,
   category spread 1 in both, no image under 64 px, 25 row groups fetched.
10. **The project lives on D: (`D:/NLP_CV`)** and model downloads go to
    `D:/hf_cache` (`hf_home` in `configs/local.yaml`, applied by
    `utils.apply_runtime_env` before any Hugging Face import).
9. **Machine-specific settings live in `configs/local.yaml`** (project_dir,
   code_dir, loader), merged over `default.yaml` via `load_config(extra_paths=...)`.
   Every label set, prompt and threshold rule stays in `default.yaml`.

## M3 — Annotation

1. **Annotator ids are `a` and `b`** (`annotation.annotators` in the config);
   files are `annotations/{split}_a.csv` and `{split}_b.csv`.
2. **A separate `notebooks/annotation.ipynb`** runs the tool and the agreement
   step. The spec lists five numbered notebooks but no home for annotation,
   which runs on a CPU runtime, by different people, alongside M5–M7.
3. **Skip leaves an item unannotated**; it is not written to the CSV and the
   tool resumes there next time.
4. **Unresolved pairs get empty gold** (treated as missing, so not scored)
   and are listed in `ground_truth_report.json`.
5. **Items labelled by only one annotator** take that label with
   `gt_source = annotated_single`; items nobody labelled keep metadata gold
   with `gt_source = metadata`. `gt_source` is per item.
6. **A stored `final_value` is kept only while both labels are unchanged**, so
   re-labelling dev after a guideline revision forces a fresh adjudication.
7. **The material-kappa check is a flag, not a stop**: the report sets
   `material_kappa_below_threshold`; a NaN kappa does not set it.
8. **Material unknown share** is computed over items whose material gold is
   set (unresolved and unlabelled items are left out).
9. **Annotation runs locally in JupyterLab** (Colab is unavailable). The
   notebook detects Colab and otherwise works from the repo root with
   `configs/local.yaml`; tested end to end in a browser on a scratch copy
   (blocked incomplete save, CSV written, resume). Annotator instructions:
   `ANNOTATING.md`.

## M4 — Extraction

1. **On Windows, `groundedgen/__init__.py` imports torch first.** Importing
   pyarrow (pulled in by pandas) before torch makes `from_pretrained`
   segfault on the local machine (native DLL clash; reproduced in isolation).
   Linux/Colab are unaffected and skip the import.
2. **Scoring runs in NumPy float32 behind a small `DualEncoder` interface**, so
   every rule (ensembling, softmax, margins, caching) is tested with a fake
   encoder; only `HFDualEncoder` touches torch.
3. **`margin` is top-1 minus top-2 probability.** The spec lists the field
   without defining it; it is stored but not used for thresholds.
4. **Infinite thresholds are stored as `null`** in `thresholds.json` (JSON has
   no infinity) and read back as `math.inf` (always abstain).
5. **Threshold candidates are the observed dev confidences**, so "lowest t
   with selective accuracy ≥ 0.90" is searched exactly; accuracy is not
   monotone in t, so the scan is exhaustive rather than a binary search.
6. **At the dev-tuned threshold, coverage and selective accuracy are computed
   over all scored pairs**; the abstention rate is over non-determinable
   pairs only (gold `unknown`).
7. **The `from_pretrained` dtype keyword is chosen by transformers version**
   (`dtype` from 4.56, `torch_dtype` before), because the local machine has
   transformers 5.x and Colab may not.
8. **`sentencepiece` is a requirement** (SigLIP's tokenizer needs it).
9. **Extraction ran locally on CPU** (fp32), as the spec's fp16-on-GPU rule
   allows for CPU. CLIP takes about 3 s per image; the model cache is on D:.
10. **SigLIP downloads avoid the Xet transfer locally.** The Xet download
    stalled at 1.48 of 3.5 GB; resuming with `HF_HUB_DISABLE_XET=1` (plain
    HTTP) finished in under 5 minutes. Revisions used: CLIP
    `32bd6428…`, SigLIP `9fdffc58…` (full hashes in `outputs/run_info.json`).

## M5 — Generation

1. **Prompt line breaks.** The PDF wraps prompt lines; each paragraph is one
   line, with a newline after "Product attributes:" and a blank line between
   the evidence JSON and the instruction. `prompts.py`.
2. **APG prompt when category is unknown** replaces the `{category}` clause
   with "Use each placeholder at most once, and refer to the product as
   \"this piece\"."; with **no known attribute** no slots are offered and the
   text only says to refer to "this piece". `prompts.apg_user`.
3. **The judge prompt lives in `prompts.py`** with the generation templates,
   so every prompt is hashed and frozen together.
4. **`prompt_sha256` per record** is one hash over the templates the condition
   uses (`prompts.condition_sha256`); per-template hashes go in `frozen.json`.
5. **A chat template "rejects" the system role** if it raises *or* silently
   drops the system text. Then the system text is prepended to the user
   message and the generator logs `system_merged` in `run_info.json`. Checked
   on the real templates: Qwen2.5-7B and Qwen2-VL keep the system message;
   Mistral-7B-v0.3 accepts it (placed in the user turn); **LLaVA-1.5 drops it,
   so A-F on LLaVA uses the merged form.**
6. **Output cleaning** strips whitespace, matching outer quotes (straight or
   curly) and a leading "Description:" / "Product description:" /
   "Description for …:" label, optionally in `**bold**`; up to 3 passes for
   nested forms. The raw output is always kept.
7. **`seconds` is the batch wall time divided by the batch size.**
8. **B-Abs and APG are skipped** for an extractor without dev thresholds (the
   state before annotation) instead of failing the whole run.
9. **The test split is guarded in code**: `run_matrix(..., "test")` raises
   unless `outputs/frozen.json` exists and matches the current prompt hashes
   and thresholds (spec §0.6, M8).
10. **The HF token** is read from `HF_TOKEN`, Colab secrets or Kaggle secrets
    into the process environment only (`utils.load_hf_token`).
11. **BLIP-2 fallback** (if Qwen2-VL fails) has no chat template; it gets
    "Question: {system} {user} Answer:" and counts as a merged system prompt.
12. **Not yet verified on a GPU**: model loading, batching and the 15 GB peak
    memory criterion. The local machine has no GPU and Colab is unavailable.

## M6 — APG (and the claims matcher it needs)

1. **`claims.py` attribute matching was built in M6**, ahead of M7, because
   APG's verify step runs the claim extractor. The judge stays in M7.
2. **Hyphens between letters are matched as spaces** ("navy-blue" →
   "navy blue", "off-white" → "off white", "t-shirt" → "t shirt"), which gives
   the spec's hyphenated-compound rule while keeping spans aligned with the
   original text. Lexicon forms and exclusion phrases are normalised the same way.
3. **Plurals** are an optional "s"/"es" on any surface form.
4. **A styling clause starts at the earliest trigger in a sentence.**
5. **APG verification checks every counted mention**, not only the first
   category mention, so a kept sentence agrees with the evidence however it is
   read. Mentions inside styling clauses are ignored, as in evaluation.
6. **Prune also drops sentences with a stray `{` or `}`** (e.g. an unclosed
   `{category`), so no raw placeholder can survive.
7. **Fallback when only material is known** (category and color unknown):
   "This piece is made of {material}." The spec covers the other cases only.
8. **Plural-only garments** (jeans, trousers, shorts; `taxonomy.PLURAL_CATEGORIES`):
   "a {category}" fills as "a pair of jeans", and the fallback takes no article
   ("Blue jeans in denim.").
9. **`placeholders_used`** lists the distinct slots in sentences that survive
   verification.
10. **The section 9 property is tested on 28 hand-written adversarial outputs
    plus 3,000 seeded random outputs** (`tests/test_apg.py`).

## M7 — Claims and judge

1. **Judge JSON parsing** strips code fences, then tries the whole answer and
   the outermost `{...}` substring. Valid means `{"claims": [...]}` with a
   string `text` and a known `type` per item (types are lower-cased). An
   invalid answer is retried once with "Answer with valid JSON only."; a
   second failure is stored with `parse_error = true` and left out of UCR
   (the parse-error rate is reported alongside).
2. **Judge evidence** is the condition's own evidence; for A and A-F it is the
   gold JSON with `exclude`/missing shown as `unknown`, as in condition C.
3. **Claims are extracted for dev and test generation files together**;
   generation files hold both splits, and later stages select by item id.
4. **Claims validation sheets** list descriptions only (blind to condition);
   each labeller writes the stated labels comma-separated. The extractor is
   scored per attribute at value level against each labeller and against the
   labels both agree on; the 0.95 target applies to every comparison.
5. **Judge validation** uses blind sheets where labellers write each
   unsupported claim as `words | type`. Judge and human claims match if one
   span contains the other or their word Jaccard is at least 0.5 (greedy
   one-to-one). Subjective claims are excluded from precision and recall.
   Human–human Cohen's kappa is computed per description on "has at least one
   non-subjective unsupported claim", because claim-level spans have no fixed
   unit to compare.

## GPU host — Kaggle

1. **GPU stages run on Kaggle** (T4 ×2), because Colab will not assign a
   backend from our location and the laptop has no GPU. `notebooks/kaggle_gpu.ipynb`
   runs generation, claims and the judge for one split; `KAGGLE.md` is the
   step-by-step guide. Notebooks 01/02 still work on Colab where available.
2. **Project state moves as zips** (`groundedgen/kaggle.py`): `pack` zips
   data, annotations and outputs; `restore` merges every copy under
   `/kaggle/input` (the uploaded dataset plus any earlier run outputs) into the
   writable `/kaggle/working/grounded-hallucination`; `merge` brings a
   downloaded results zip back. Missing files are copied, JSONL files are
   unioned line by line, and for other existing files the destination wins,
   so no finished record is ever lost or duplicated.
3. **`configs/kaggle.yaml`** puts model downloads in `/tmp/hf_cache` so the
   multi-GB checkpoints are not saved as notebook output.
4. **T4 ×2, not P100**: bitsandbytes 4-bit kernels are not reliable on Pascal
   GPUs. With two T4s, `device_map="auto"` may split a model across both;
   the 15 GB check reads device 0.

## M8 — Freeze

1. **`generate.freeze(cfg)`** (or `python -m groundedgen.generate configs/local.yaml --freeze`)
   writes `outputs/frozen.json` with every template hash, per-condition
   hashes, the dev thresholds of *every* extractor, the primary extractor and
   the git commit. It refuses if an extractor has no thresholds or the file
   already exists. `check_frozen` also requires the same set of extractors.

## M9 — Evaluation

1. **A fifth outcome, `silent`**: for A/A-F with gold `unknown` and nothing
   stated. The spec's four outcomes leave this case undefined; it is neither
   an error nor an abstention against evidence.
2. **A correct statement after a gap** (e unknown, D = {g}) counts as correct
   but still counts toward leakage, and is not "lucky" (lucky needs label
   evidence).
3. **Leakage in C-Mask** uses the masked attribute only; non-masked pairs of
   C-Mask still enter HR, coverage and precision.
4. **Planned McNemar tests**: A and A-F use the primary VLM; text conditions use
   the family's LLM and the primary extractor. Items are paired on
   (item, attribute); C-Mask is not in any family.
5. **The origin test** (RQ1) uses B with the primary extractor against C, per LLM
   and attribute.
6. **`analysis.py` holds the M9 tables and figures** (T3, T4, T5, T7, F3, F4) and a
   first `numbers.json`; M11 extends it. Dev runs write `*_dev` files; test
   runs need `frozen.json`.
7. **Figure style**: reference categorical palette in fixed order (validated
   for CVD: worst adjacent ΔE 9.1); slots below 3:1 contrast get direct labels
   (F4) or a legend plus table view (F3, with T4). F4 merges labels of points
   closer than 0.04.
13. **Peak GPU memory is reset before each model and summed over GPUs.** The
    first Kaggle dev run reported an identical 3.09 GB for all four models
    because the counter was never reset and only GPU 0 was read; with
    `device_map="auto"` on T4 ×2 the sum is the single-T4 equivalent.
5. **`run_info.json` is deep-merged when results come back from Kaggle**:
   keys missing locally (e.g. LLM/VLM revisions, `system_merged`) are added,
   local values win on conflict, and the remote environment snapshot is kept
   under `remote_env`. The first dev run confirmed `system_merged = true` only
   for LLaVA-1.5.
10. **Dev round 1 → guidelines v1.1.** Round 1 kappa: category 1.00, color
    0.70, material 0.34. Most material disagreements came from one annotator
    never using `unknown` (cotton chosen from garment type), linen over-use,
    and synthetic vs cotton on slight sheen; colour disagreements from printed
    single-colour garments and very dark shades. v1.1 adds 11 decision rules
    (material defaults to `unknown`; prints keep the base colour; dark shades
    keep their hue) without changing the v1.0 text. Round-1 labels, the
    disagreement sheet and the derived gold are archived in
    `annotations/dev_round1/`; both annotators re-label dev independently.
11. **Round-2 re-check of cotton/synthetic.** Round 2 reached material kappa
    0.71 but neither annotator used `unknown` (0/40). Before adjudication, each
    annotator independently revisits only their own cotton/synthetic labels
    (`annotate.recheck_items`, notebook "Re-check" cell) and keeps a fabric only
    if the cue is visible. Kappa is recomputed afterwards; both values are
    reported.
12. **Guidelines v1.1 frozen with dev material kappa 0.54 (deviation from the
    0.6 rule).** Dev kappa history for material: round 1 0.34; round 2 0.71
    (but 0/40 `unknown`, suggesting shared garment-type priors); after the
    visibility re-check 0.54 (95% bootstrap CI 0.35–0.71; raw agreement 0.625;
    15 `unknown` labels). After the re-check, 11 of 15 disagreements are
    "unknown vs a fabric" and, where both annotators named a fabric, they chose
    the same one on 23 of 27 items (85%): the remaining disagreement is about
    *whether* the fabric is visible, an inherently graded judgement at
    384×512. No further dev round was run because the CI includes 0.6 and the
    annotators had seen the 40 dev images three times (a further round would
    partly measure recall). Test (200 fresh items) gives the main agreement
    estimate. Category 1.00, color 0.84. All values are reported in the paper.
13. **Dev adjudication** uses `annotations/dev_adjudication.html` (image, both
    labels, a dropdown of all allowed values), which writes the filled
    `disagreements_dev.csv`. The adjudicator is a third team member and may
    choose a value neither annotator chose.

## Dev gold and thresholds (after adjudication)

1. **Dev gold**: 40/40 items annotated, 0 unresolved after adjudication (the
   adjudicator sided with a 9×, b 11×, and chose a third value once). Material
   gold: 8/40 `unknown` (20%).
2. **Dev thresholds without touching test**: `eval_extraction.tune_on_dev`
   tunes and saves thresholds, picks the primary extractor and reports dev
   metrics (`outputs/extraction/dev_tuning.json`). `run_evaluation` (T2, F2 on
   test) runs only once test gold exists. Primary extractor: SigLIP (dev
   material AURC 0.53 vs CLIP 0.85). Material thresholds: CLIP ∞ (always
   abstain), SigLIP 0.986 (3% coverage), i.e. B-Abs/APG almost always receive
   `unknown` material.
3. **Stale-evidence purge** (`generate.purge_stale`, run at the start of every
   `run_matrix`): cache keys do not include evidence, so C/C-Mask records made
   with pre-annotation (metadata) gold would otherwise be kept forever. Records
   whose evidence differs from what the current gold/extraction/thresholds
   give are dropped with their claims and judge records and regenerated. The
   first purge removed 34/40 C and 67/80 C-Mask dev records per LLM.
