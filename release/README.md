# Released labels and item IDs

Released under the release plan in the build specification: labels and item IDs only.
Images are not redistributed; `python -m groundedgen.data configs/local.yaml` downloads them
from the pinned Hugging Face dataset revision and rebuilds the same split.

| File | Contents |
|---|---|
| `manifest_final.csv` | 240 items (40 dev, 200 test): source IDs, metadata and adjudicated gold category/color/material |
| `ground_truth_report.json`, `dataset_report.json` | Agreement and split statistics |
| `annotations/{dev,test}_{a,b}.csv` | Independent labels from annotators a and b |
| `annotations/disagreements_{dev,test}.csv` | Disagreements and the adjudicated final value |
| `human_eval/sheet.csv`, `key.csv` | Blinded rating sheet and the condition key |
| `human_eval/ratings_r{1,2}.csv` | Human ratings (fluency, faithfulness, usefulness 1–5; publish yes/no) |
| `validation/claims_validation_*.csv` | Claim-extractor validation labels (60 dev descriptions) |
| `validation/judge_validation_*.csv` | Judge validation labels (70 test descriptions, labeller a) |

Annotators are identified only as a, b and raters as r1, r2.
