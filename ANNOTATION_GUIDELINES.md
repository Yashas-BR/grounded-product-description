# Annotation Guidelines

Version 1.1 (used from dev round 2). The rules below are build spec §6
verbatim (v1.0); the "Clarifications (v1.1)" section adds decision rules after
dev round 1 (kappa: category 1.00, color 0.70, material 0.34 — below the 0.6
bar). Round-1 labels are archived in `annotations/dev_round1/`.

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

## Clarifications (v1.1)

Written after dev round 1, where most disagreements came from the cases below.
They refine the v1.0 rules and do not change them.

**Material — the default is `unknown`.**

1. Start every item at `unknown`. Change it only if you can point to the cue
   in the image (zoom in). If you would have to say "it's probably…", keep
   `unknown`.
2. If you hesitate between two fabrics, choose `unknown`.
3. The garment type is never a cue. A plain t-shirt, shirt, sweatshirt or
   jacket with no visible weave, loops or sheen is `unknown`, not cotton.
4. **Cotton** only when a matte woven texture is actually visible (e.g. the
   weave of a shirt cloth at full size). Smooth, flat, untextured fabric is
   `unknown`.
5. **Knit** when loops or ribbing are visible (sweaters, ribbed cuffs and
   collars on the body of the garment). A smooth jersey t-shirt is `unknown`.
6. **Denim** for twill weave, fading/whiskering or contrast stitching on jeans,
   jackets or skirts. Jeans with these cues are denim, never linen.
7. **Linen** needs both slub texture (uneven thick/thin threads) and creasing.
   Creases alone are not enough.
8. **Synthetic** only for clearly glossy technical or athletic fabric, or mesh.
   A slight sheen on a printed kurta or saree is not enough → `unknown`.
9. **Silk / chiffon**: silk for strong sheen with fluid drape; chiffon when the
   fabric is visibly sheer (you can see through it).

**Color.**

10. A single-color garment with a print, logo or graphic is **not** multicolor:
    label its base color. Use `exclude` only when no single color covers clearly
    more than half of the garment.
11. Dark shades: choose `black` only when no hue is visible. Dark navy → blue,
    charcoal → grey, dark purple → purple, maroon → red.

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
