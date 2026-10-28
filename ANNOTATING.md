# How to annotate (local laptop)

Two people (annotator **a** and annotator **b**) each label all 240 images,
**independently**: never look at the other person's labels or CSV until both
have finished a split. Read [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md)
before you start.

## 1. Open the tool

In a terminal:

```
cd D:\NLP_CV\ghd
python -m jupyter lab notebooks/annotation.ipynb
```

A browser tab opens with the notebook. Run the first code cell (setup), then
in the **Label** cell set:

```python
ANNOTATOR = "a"   # or "b"
SPLIT = "dev"
```

and run it (Shift+Enter). The form appears under the cell.

## 2. Label

For each image:

- **Category** and **Color** are pre-filled from the catalog. Check them
  against the image and change them if they are wrong. Use `exclude` when the
  image is not a single garment of that kind (category) or the color is
  multicolor, a print, color blocks or a near-equal split (color).
- **Material** starts blank and must be chosen. Pick a fabric only if you can
  *see* the cue (weave, knit loops, sheen, sheerness, denim fading…);
  otherwise choose `unknown`. Never guess from the garment type
  ("t-shirts are usually cotton") — that guessing is what the study measures.
- **Notes** (optional): anything odd about the image.
- **Save & Next** saves and moves on. **Prev** goes back (your saved labels
  are shown, and saving again overwrites them). **Skip** moves on without
  saving; the item stays unlabelled.

Every save is written straight to
`D:\NLP_CV\grounded-hallucination\annotations\{split}_{annotator}.csv`.
You can close the browser at any time; re-running the Label cell resumes at
the first unlabelled item.

Dev has 40 images (about 15–20 minutes); test has 200.

## 3. Order of work

1. **a** and **b** both label **dev**.
2. Run the **Agreement** cell. If material kappa is below 0.6, the team
   discusses the disagreements, refines `ANNOTATION_GUIDELINES.md` (bump the
   version), and both re-label dev.
3. Freeze the guidelines, then **a** and **b** both label **test**.
4. Run the **Agreement** cell again. A third team member opens
   `annotations/disagreements_{split}.csv` and fills the `final_value`
   column (allowed values are listed in the guidelines), then the cell is
   run once more. It writes `data/manifest_final.csv`, which every later
   stage uses.

## Annotating on a different laptop

Copy the whole `D:\NLP_CV` folder (code + `grounded-hallucination\data`,
about 10 MB of images). If it ends up at a different path, edit
`project_dir` and `code_dir` in `ghd\configs\local.yaml`. When done, send
back only your `annotations\{split}_{annotator}.csv` and place it in the
same folder on the main laptop.
