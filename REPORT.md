# Extract, Abstain, Generate: Decomposing the Origins of Attribute Hallucination in Grounded E-Commerce Product Descriptions

Team 10, RV University. Guide: Prof. Lokesh J K.

All numbers are on the frozen **test split (200 items)**, with 95% item-cluster bootstrap
confidence intervals. Tables and figures are in [`paper/`](paper/) and regenerate from cached
outputs with `python -m groundedgen.finish --rebuild configs/local.yaml`. Every departure from
the build specification is listed in [Limitations](#7-limitations-and-deviations) and in
[`DECISIONS.md`](DECISIONS.md).

## Abstract

Product descriptions generated from images often state attributes that are wrong. We split
the pipeline into an extraction stage (zero-shot CLIP/SigLIP, with abstention thresholds tuned
on dev) and a language stage (a 7B LLM writing from the extracted JSON). Then we trace every
category, color and material mention to its origin. Two results stand out.

- **The language stage adds almost no attribute errors.** Every hallucination in the grounded
  pipeline B comes from extraction (GD = 0 for both Qwen2.5-7B and Mistral-7B).
- **Abstention survives the LLM.** When the extractor abstains, the LLM almost never fills
  in a guess (leakage ≤ 0.5%).

Abstaining on low-confidence material cuts the material hallucination rate from 0.58 to 0.00.
Color goes from 0.23 to 0.15, at 85% color coverage. Image-only VLM baselines hallucinate
category and color in about half of descriptions. Our proposed Abstention-Preserving
Generation (APG) has the fewest unsupported non-attribute claims (UCR 0.17 vs 0.57 for B-Abs).
Human raters, however, find its descriptions too terse: fluency 1.67 vs 4.30 for B-Abs.

## 1. Research questions

| RQ | Question | Comparison |
|---|---|---|
| RQ1 | Where does attribute hallucination originate: vision stage or language stage? | B vs. C |
| RQ2 (headline) | Does abstention survive the LLM? Leakage (unknown replaced by a guess) and inflation (known attribute dropped or hedged) | B vs. B-Abs; C vs. C-Mask |
| RQ3 | Does APG remove language-stage attribute hallucination without hurting fluency or usefulness? | APG vs. A, A-F, B-Abs; human ratings |

## 2. Conditions

| ID | Input to the generator | Role |
|---|---|---|
| A | Image → VLM, generic prompt | Conventional baseline |
| A-F | Image → VLM, faithfulness prompt | Strong baseline |
| B | Forced extraction JSON → LLM, free text | Grounding without abstention |
| B-Abs | Thresholded extraction JSON → LLM, free text | Grounding with abstention |
| APG | Thresholded extraction JSON → LLM with slots → renderer | Proposed method |
| C | Gold JSON → LLM, free text | Oracle |
| C-Mask | Gold JSON, one attribute masked → LLM, free text | Leakage probe |

*Figure F1 (pipeline diagram) is drawn by hand, as the specification requires, and is not generated.*

**Models.**
- VLMs: llava-1.5-7b (primary), Qwen2-VL-2B.
- LLMs: Qwen2.5-7B-Instruct (primary), Mistral-7B-Instruct-v0.3.
- Extractors: SigLIP-so400m (primary), CLIP ViT-L/14.

7B models are 4-bit NF4 on a T4. Decoding is greedy, with seed 42. Prompts and thresholds were
frozen on dev before any test run (`outputs/frozen.json`).

## 3. Data and annotation (T1)

- **Items:** 40 dev and 200 test images from the Fashion Product Images dataset (384×512 copy),
  13 categories balanced to within 1, including kurtas and sarees (ethnic wear).
- **Labels:** two annotators labelled every item independently, and a third team member
  adjudicated all disagreements.

Test agreement (Cohen's kappa):
- category: 1.00
- color: 0.86
- material: 0.50 (raw agreement 0.60)

Material is the hardest attribute to judge from a photo. 15.5% of test items have gold material
`unknown`.

## 4. Measurement

- **Attribute claims.** A deterministic lexicon extractor reads each description. It applies
  longest-match, styling-clause and hedge rules, and it ignores colors of logos, prints and trim.
  Validated against labels both labellers agree on (60 dev descriptions):
  - category: P 1.00 / R 0.95 (recall is limited by the spec's first-mention rule)
  - color: 1.00 / 1.00
  - material: 1.00 / 1.00
- **Outcome per attribute.** Each mention is scored as correct, hallucination, or not stated.
  A hallucination is then traced to its origin:
  - EP: the extractor passed a wrong value
  - GD: the generator deviated from correct evidence
  - GF: the generator fabricated an attribute the evidence left unknown
- **Non-attribute claims (UCR).** A Qwen2.5-7B judge lists unsupported claims, such as fit,
  function or occasion. UCR counts the non-subjective ones per description; subjective praise
  is counted separately.
- **Statistics.**
  - Exact McNemar with Holm correction within each (attribute, LLM) family.
  - Wilcoxon signed-rank for human ratings.
  - Krippendorff's alpha for rater agreement.

## 5. Results

### 5.1 Extraction (T2, F2)

| Extractor | Category acc. | Color acc. | Material acc. | Coverage at threshold (cat / col / mat) |
|---|---|---|---|---|
| CLIP ViT-L/14 | 0.84 | 0.66 | 0.49 | 92% / 30% / 0% |
| SigLIP so400m | **0.91** | **0.77** | 0.50 | 97% / 85% / 2% |

Zero-shot material recognition is near chance for both models. The dev-tuned threshold has
SigLIP abstain on material for 98% of items, and CLIP never asserts it. SigLIP is the primary
extractor.

### 5.2 Hallucination rate by condition (T3, primary configuration)

| Condition | Category HR | Color HR | Material HR |
|---|---|---|---|
| A | 0.45 [0.38, 0.52] | 0.52 [0.45, 0.59] | 0.00 |
| A-F | 0.46 [0.39, 0.53] | 0.45 [0.38, 0.53] | 0.00 |
| B | 0.09 [0.05, 0.13] | 0.23 [0.17, 0.29] | **0.58** [0.51, 0.65] |
| B-Abs | 0.07 [0.04, 0.11] | 0.15 [0.10, 0.20] | 0.00 |
| APG | 0.07 [0.04, 0.11] | 0.15 [0.10, 0.20] | 0.00 |
| C (oracle) | 0.00 | 0.00 | 0.00 |
| C-Mask | 0.01 [0.00, 0.02] | 0.00 | 0.00 |

- **The image-only VLM is wrong about category or color in about half its descriptions.**
  - llava calls many garments by the wrong category, especially ethnic wear (see 5.6).
  - The faithfulness prompt (A-F) only reduces color HR, from 0.52 to 0.45 (McNemar Holm p = 0.034).
  - The VLMs almost never mention material, so their material HR of 0 reflects silence, not
    accuracy. llava's material coverage is 0.5%.
- **Grounding on extracted attributes (B) removes most category and color errors.** But
  because B forces the extractor to commit, it states a material for every item, and 58% of
  those are wrong.
- **Abstention (B-Abs) removes every material error and a third of the color errors.**
  B vs. B-Abs: material −0.58 (Holm p = 1.2e-34), color −0.08 (p = 1.8e-4). Category does not
  change significantly (p = 0.38). The price is coverage: color is stated for 85% of items, and
  material for 2%.
- **Swapping the LLM changes nothing.** Mistral-7B gives the same hallucination rates as Qwen2.5-7B
  (T3), and their per-pair outcomes agree on 99.3% of pairs. The language stage copies the evidence.

### 5.3 RQ1: Origins (T4, F3)

In B, **every** attribute hallucination comes from the extraction stage. For the primary LLM:

- category: EP 0.09, GD 0.00, GF 0.00
- color: EP 0.23, GD 0.00, GF 0.00
- material: EP 0.58, GD 0.00, GF 0.00

The GD rate is also indistinguishable from the oracle C. GD(B) − GD(C) is 0.000 for every
attribute with Qwen2.5, and at most 0.005 [0.000, 0.015] with Mistral. **Answer to RQ1:**
attribute hallucination in a grounded pipeline originates in the vision stage. A 7B
instruction-tuned LLM given structured evidence does not add attribute errors.

### 5.4 RQ2: Does abstention survive the LLM? (T5)

| Condition | Leakage (unknown replaced by a guess) | Inflation (known attribute dropped or hedged) |
|---|---|---|
| B-Abs, Qwen2.5 | 0.00 for all attributes | 0.00 for all attributes |
| B-Abs, Mistral | ≤ 0.005 | category 0.03 [0.01, 0.06] |
| APG, Qwen2.5 | 0.00 | 0.00 |
| APG, Mistral | 0.00 | color 0.03 [0.01, 0.06] |
| C-Mask, Qwen2.5 | category **0.05** [0.02, 0.09], material 0.006 | 0.00 |
| C-Mask, Mistral | category 0.02 [0.01, 0.04] | category 0.05, color 0.03 |

**Answer to RQ2: yes.** When the evidence says `unknown`, both LLMs almost always keep it unknown.

The one exception is C-Mask category. When the category is masked, Qwen2.5 still names a
garment type in 5% of descriptions. It infers this from the remaining attributes: all 10 leaks say "jeans",
which it infers from denim material.

Mistral shows a small amount of inflation: it sometimes drops or hedges an attribute it was given.

### 5.5 RQ3: APG and human judgement

**Non-attribute claims (judge, UCR per description, primary configuration).**

| A | A-F | B | B-Abs | **APG** | C | C-Mask |
|---|---|---|---|---|---|---|
| 1.02 | 1.02 | 0.69 | 0.57 | **0.17** | 0.68 | 0.71 |

APG keeps the attribute-level guarantees of B-Abs, with the same hallucination rate and zero
leakage. It also cuts unsupported claims such as fit, occasion or function by about 70%.

Its slot renderer fell back to a template for 6% of items with Qwen2.5 and 1–2% with Mistral (T8).

**Human evaluation (T6).** Two raters scored 30 test items (8 ethnic wear), with descriptions
from A-F, B-Abs and APG shuffled and unlabelled. Scores are 1–5.

| Condition | Fluency | Faithfulness | Usefulness | Publish as-is |
|---|---|---|---|---|
| A-F | 2.15 [1.97, 2.33] | 3.03 [2.77, 3.28] | 2.12 [1.88, 2.37] | 0.08 |
| APG | 1.67 [1.30, 2.10] | 2.17 [1.77, 2.58] | 1.72 [1.38, 2.07] | 0.23 |
| **B-Abs** | **4.30** [4.07, 4.52] | **4.37** [4.20, 4.53] | **4.15** [3.90, 4.38] | **0.93** |

Rater agreement (Krippendorff's alpha):
- fluency: 0.88
- usefulness: 0.81
- publish: 0.66
- faithfulness: 0.57

B-Abs beats APG on every scale (Wilcoxon, Holm p < 1e-4). APG is also rated less faithful than
A-F (p = 0.012), but its fluency and usefulness are not significantly different from A-F
(p = 0.10).

**Answer to RQ3: partly.**
- **What APG achieves:** it removes language-stage attribute hallucination. That was already
  zero in B-Abs (5.3), so APG adds no gain there. It also removes most unsupported free-text claims.
- **What it costs:** APG's outputs are often a single templated sentence, such as "Kurta in red."
  Raters judged these much worse than the fluent B-Abs text. The low faithfulness score, despite
  zero attribute errors, suggests raters penalise descriptions that say almost nothing, or that
  repeat a wrong extracted attribute without any context to soften it.
- **Practical recommendation:** for a deployable system, B-Abs is the better trade-off. Use
  grounding plus abstention and let the LLM write freely.

### 5.6 Western vs. ethnic wear (T7)

| Measure | Western | Ethnic |
|---|---|---|
| SigLIP color accuracy | 0.80 | **0.62** |
| llava A category HR | 0.39 | **0.75** |
| llava A-F category HR | 0.38 | **0.88** |
| B-Abs color HR (primary) | 0.07 | 0.16 |

The image-only VLM is badly wrong about kurtas and sarees: 22 of its 24 category errors on kurtas and sarees call
them a dress.
Extraction and grounding close most of this gap for category, where SigLIP is 1.00 on ethnic
items. Multi-colored ethnic garments are still harder for color.

### 5.7 Measurement validation (T8)

- **Claim extractor:** see section 4 (P = 1.00, R ≥ 0.95 on every attribute).
- **LLM judge vs. one human labeller** (70 test descriptions, 10 per condition, non-subjective claims):
  - precision 0.31, recall 0.68
  - 49 judge claims vs. 22 human claims
  - the judge over-counts by roughly 2×, so absolute UCR values are upper bounds
  - **the ranking is preserved:** human and judge both give APG the fewest claims (0.2 and 0.3
    per description), with the other conditions at 0.3–0.9 (human) and 0.8–1.1 (judge)

## 6. Discussion

1. **Fix the extractor, not the LLM.** In a grounded pipeline, attribute hallucination is an
   extraction problem. Better or calibrated vision models are where the gains are. Swapping or
   constraining the LLM changes nothing at the attribute level.
2. **Abstention is cheap and propagates.** Thresholds tuned on 40 dev items removed all material
   errors on test, and the LLMs respected `unknown` almost perfectly. Coverage is the cost:
   material is effectively never stated, because zero-shot extractors cannot read fabric from a
   photo.
3. **Hard constraints cost fluency.** APG's slot-and-render design guarantees faithfulness but
   reads as mechanical. Humans strongly preferred free-text B-Abs, which already has the same
   attribute hallucination rate.

## 7. Limitations and deviations

- **Material agreement is low.** Annotation kappa is 0.50, below the 0.6 target. Guidelines
  were refined once on dev (v1.1) and frozen before test; test was not re-labelled.
- **Category and color labels were pre-filled.** The annotation tool pre-filled them from the
  dataset metadata for annotators to verify; only material was labelled from scratch. Their
  kappa therefore overstates fully independent agreement.
- **Claim extractor changed once.** One rule was added on dev (colors of logos, prints,
  embroidery and trim are not garment colors) and re-validated.
- **Judge scope.**
  - It was run on the primary configuration only.
  - It was validated on 70 test descriptions by one labeller (spec: 20% of descriptions, two
    labellers), so there is no human–human agreement for the judge.
  - It is the same model family as one generator (Qwen2.5).
- **Human evaluation was smaller than specified.** 30 items and 2 raters (spec: 50 items,
  3 raters). Faithfulness agreement is only moderate (alpha 0.57).
- **Data source.** Images come from a 384×512 copy of the dataset, not the specification's
  60×80 version, for legibility.
- **Compute.** Runs used Kaggle T4 GPUs instead of Colab.
- **Small sample.** 200 test items with 13 categories: per-category and ethnic-wear estimates
  have wide intervals.

## 8. Reproducibility

- **Code:** [`groundedgen/`](groundedgen/), MIT licence. 334 unit tests run on CPU without network.
- **Labels and item IDs:** [`release/`](release/). Images are not redistributed; download them
  with `python -m groundedgen.data`.
- **Frozen settings:** prompt hashes, thresholds, seeds, package versions, GPU and model revisions
  are in `outputs/frozen.json` and `outputs/run_info.json`.
- **GPU stages:** run on Kaggle with `notebooks/kaggle_gpu.ipynb` ([`KAGGLE.md`](KAGGLE.md)).
- **Paper assets:** every table and figure regenerates with `python -m groundedgen.finish --rebuild`.
