# Research Design

## Questions

| RQ | Question | Comparison |
|----|----------|------------|
| RQ1 | Where does attribute hallucination originate: vision stage or language stage? | B vs. C |
| RQ2 (headline) | Does abstention survive the LLM? Leakage (unknown replaced by a guess) and inflation (known attribute dropped or hedged). | B vs. B-Abs; C vs. C-Mask |
| RQ3 | Does APG remove language-stage attribute hallucination without hurting fluency or usefulness? | APG vs. A, A-F, B-Abs; human ratings |

Attributes: category, color, material. Human evaluation: three raters on
buyer usefulness, fluency and faithfulness (no seller study).

## Conditions

| ID | Input to the generator | Role |
|----|------------------------|------|
| A | Image → VLM, generic prompt | Conventional baseline |
| A-F | Image → VLM, faithfulness prompt | Strong baseline |
| B | Forced extraction JSON → LLM, free text | Grounding without abstention |
| B-Abs | Thresholded extraction JSON → LLM, free text | Grounding with abstention |
| APG | Thresholded extraction JSON → LLM with slots → renderer | Proposed method |
| C | Gold JSON → LLM, free text | Oracle |
| C-Mask | Gold JSON, one attribute masked → LLM, free text | Leakage probe |

## Unit of analysis

One (description, attribute) pair; pairs with gold `exclude` or missing are
not scored. Notation: `g` gold (label or `unknown`), `e` evidence value given
to the generator (absent for A/A-F), `D` set of values stated, `h` hedged.

| Outcome | Rule |
|---------|------|
| Correct statement | D = {g} and g is a label |
| Hallucination | D non-empty and D ≠ {g} (any statement when g = unknown) |
| Omission | D empty and e is a label (A/A-F: g is a label) |
| Preserved abstention | D empty and e = unknown |

| Origin | Rule | Stage |
|--------|------|-------|
| EP, extraction-propagated | e label, D = {e}, e ≠ g | Vision |
| GD, generation deviation | e label, D ≠ {e} | Language |
| GF, gap-fill | e = unknown, D non-empty | Language |

"Lucky" statements (D ≠ {e} but D = {g}) count as correct and are reported
separately.

## Rates

Per condition × generator × extractor × attribute, N = scored pairs:
HR = hallucinations / N; coverage = pairs with D non-empty / N; precision =
correct / statements; EP, GD, GF rates = counts / N; extraction share =
EP / hallucinations; leakage = pairs with e = unknown and D non-empty / pairs
with e = unknown (C-Mask: masked attribute only, headline RQ2 number);
inflation = pairs with e a label and (D empty or h) / pairs with e a label.
Description level: share with ≥ 1 attribute hallucination; unsupported claim
rate (UCR). A and A-F report HR, coverage, precision and description-level
rates only.

## Statistics

95% percentile bootstrap (2,000 resamples, items resampled as clusters, seed
42); exact McNemar on paired hallucination indicators with Holm correction per
family (per attribute, primary extractor, each LLM: A vs. A-F; A-F vs. B-Abs;
A-F vs. APG; B vs. B-Abs; B-Abs vs. APG); RQ1 origin test as the bootstrap CI
of GD(B) − GD(C); Cohen's kappa for annotation; Krippendorff's alpha
(ordinal) for human ratings; risk–coverage curves and AURC for selective
prediction. Effect sizes with CIs accompany every p-value.
