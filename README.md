# Extract, Abstain, Generate

Code for *Extract, Abstain, Generate: Decomposing the Origins of Attribute
Hallucination in Grounded E-Commerce Product Descriptions* (Team 10, RV
University; guide Prof. Lokesh J K).

The study traces every category, color and material stated in a generated
product description back to its origin (vision stage or language stage),
measures whether an `unknown` from the vision stage survives the language
stage, and proposes Abstention-Preserving Generation (APG). See
[RESEARCH_DESIGN.md](RESEARCH_DESIGN.md) for the design and
[DECISIONS.md](DECISIONS.md) for choices made beyond the build spec.

## Results

The written report with all test results is [REPORT.md](REPORT.md); tables and figures are in
[`paper/`](paper/), released labels in [`release/`](release/).

## Setup

Local (CPU, tests only):

```bash
pip install numpy pandas scipy pyyaml pytest krippendorff
python -m pytest tests
```

Colab (T4 GPU):

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/Yashas-BR/grounded-product-description /content/ghd
%cd /content/ghd
!pip install -q -r requirements.txt
!python -m pytest -q tests
```

Local CPU stages (data splits; later evaluation and figures) run without a
notebook. `configs/local.yaml` sets a local `project_dir` and a low-disk
loader that streams the dataset instead of caching it:

```bash
python -m groundedgen.data configs/local.yaml
```

GPU stages (generation, judge) run on Kaggle: see [KAGGLE.md](KAGGLE.md).

Gated checkpoints (Mistral) need an `HF_TOKEN` in Colab secrets; it is never
written to disk or logs.

## Quick start

```python
from groundedgen import utils
cfg = utils.load_config()                 # configs/default.yaml
utils.set_seed(cfg["seed"])
utils.update_run_info(utils.project_path(cfg, "outputs", "run_info.json"),
                      {"env": utils.capture_env()})
```

All data, annotations and outputs live under `project_dir` on Google Drive;
every stage caches JSONL outputs and resumes where it stopped.

## Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Foundations: utils, taxonomy, metrics, config, tests | done |
| M2 | Data: `data.py`, notebook 01 part 1 | done (run locally) |
| M3 | Annotation: `annotate.py`, `agreement.py`, guidelines, `notebooks/annotation.ipynb` | done: dev + test labelled by a and b, adjudicated (0 unresolved) |
| M4 | Extraction: `extract.py`, `eval_extraction.py` | done: dev-tuned thresholds, T2, F2 on test |
| M5 | Generation: `models.py`, `prompts.py`, `generate.py`, notebook 02 | done on dev (Kaggle T4); test run on Kaggle |
| M6 | APG: `apg.py` (+ `claims.py` matcher) | done; dev fallback rate 0–10% |
| M7 | Claims and judge: `claims.py`, `judge.py` | done; validation pages ready, hand labels pending |
| M8 | Freeze prompt hashes and thresholds | done (`outputs/frozen.json`, 2026-10-29) |
| M9 | Evaluation: `decompose.py`, `analysis.py`, notebook 03 | done on dev; test after the Kaggle run |
| M10 | Human evaluation: `human_eval.py`, notebook 04 | done (mock ratings run end to end); ratings pending |
| M11 | Paper assets: `analysis.py` (`build_all`), notebook 05 | done: T1–T8, F2–F5, numbers.json in one call |

Nothing in M9 runs on the test split before M8 is complete.

## License

Code: MIT. Labels and item IDs are released; images only if the dataset
license allows redistribution, otherwise via a download script.
