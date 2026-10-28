# Running the GPU stages on Kaggle

Generation (M5), APG on real outputs (M6) and the judge (M7) need a GPU.
Colab is unavailable from our location and the laptop has none, so these run
in a free Kaggle notebook (T4 ×2, about 30 GPU hours a week).

## One-time setup

1. **Kaggle account.** Sign up at kaggle.com and **verify your phone number**
   (Settings → Phone verification). Without it, GPU and Internet stay disabled.
2. **Hugging Face.**
   - Accept the terms of `mistralai/Mistral-7B-Instruct-v0.3` on its model page.
   - Create a read token (huggingface.co → Settings → Access Tokens).
3. **Upload the project as a Kaggle dataset.** Locally:
   ```
   cd D:\NLP_CV\ghd
   python -m groundedgen.kaggle pack configs/local.yaml
   ```
   This writes `D:\NLP_CV\kaggle_upload\ghd-project.zip`, which holds the
   images, manifest, annotations and extraction outputs (a few MB). On Kaggle,
   go to **Datasets → New Dataset**, upload that zip, name it `ghd-project`,
   and keep it **Private**.
4. **Create the notebook.** Choose **Code → New Notebook → File → Import Notebook**,
   then paste the GitHub URL of `notebooks/kaggle_gpu.ipynb`:
   `https://github.com/Yashas-BR/grounded-product-description/blob/main/notebooks/kaggle_gpu.ipynb`
5. **Notebook settings** (right-hand panel):
   - Accelerator: **GPU T4 x2**. Avoid P100: 4-bit loading is not reliable on it.
   - Internet: **On**.
   - **Add Input → Your Datasets → `ghd-project`**.
   - **Add-ons → Secrets → Add** `HF_TOKEN` (your token), and tick it for this notebook.

## Each run

1. Set `SPLIT` in the first cell. Use `"dev"` until M8 has frozen the prompts
   and thresholds; the code refuses `"test"` before that.
2. Choose **Save Version → Save & Run All (Commit)**. It runs in the background,
   so you can close the browser. Each 7B model takes a few minutes to download
   the first time.
3. When the run finishes, open the version's **Output** tab and download
   `ghd_outputs.zip`.
4. Merge it into the local project:
   ```
   cd D:\NLP_CV\ghd
   python -m groundedgen.kaggle merge path\to\ghd_outputs.zip configs/local.yaml
   ```
   Merging only adds records, so nothing local is overwritten.

## If a run stops (12-hour limit, error, quota)

Everything is cached. In the notebook, choose **Add Input → Notebook Output** and pick
the previous version of this notebook, then run again. The restore cell merges
its outputs, and finished records are skipped.

## When local data changes (new annotations, thresholds)

Run `pack` again and upload the new zip as a **new version** of the
`ghd-project` dataset (Dataset page → New Version). Then re-run the notebook.
