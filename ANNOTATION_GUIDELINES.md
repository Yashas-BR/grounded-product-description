# Annotation Guidelines

Version 1.0 (copied verbatim from build spec §6; bump the version whenever
the guidelines are refined after the dev round).

- **Category:** the main garment being sold; `exclude` if the image shows
  something else.
- **Color:** the dominant color of the main garment in the 12 basic terms;
  ignore skin, background and other garments; `exclude` for multicolor
  prints, color blocks or near-equal splits.
- **Material:** choose a fabric only if visible cues support it; otherwise
  `unknown`. Cues: denim (twill weave, fading, contrast stitching), knit
  (visible loops or ribbing), leather (grain, sheen, stiff drape), silk
  (strong sheen, fluid drape), chiffon (sheer, light), linen (slub texture,
  creasing), cotton (matte weave clearly visible), synthetic (glossy technical
  or athletic fabric, mesh).

Never use the product name, brand, price or general knowledge ("t-shirts are
usually cotton"). That prior-based inference is what the study measures in
models.

## Process

1. Both annotators label dev. Compute kappa per attribute.
2. If material kappa is below 0.6, discuss disagreements, refine the
   guidelines (bump the version), and re-label dev.
3. Freeze the guidelines, then both annotators label test.
4. Write `annotations/disagreements_{split}.csv` (`item_id, attribute,
   annotator_a, annotator_b, final_value`). A third team member fills
   `final_value`.
5. Gold = the agreed value, else the adjudicated value; anything else is
   unresolved and reported. Validate that every gold value is in the allowed
   set.
6. Write `manifest_final.csv` and `data/ground_truth_report.json` (kappa, raw
   agreement, counts, unresolved).

## Label sets

| Attribute | Options |
|-----------|---------|
| Category | t-shirt, shirt, top, kurta, saree, jeans, trousers, shorts, dress, skirt, sweater, sweatshirt, jacket, `exclude` |
| Color | black, white, grey, blue, red, green, yellow, orange, pink, purple, brown, beige, `exclude` |
| Material | chiffon, cotton, denim, knit, leather, linen, silk, synthetic, `unknown` |

Category and color are pre-filled from catalog metadata for verification;
change them if the image disagrees. Material starts blank and must be chosen.
