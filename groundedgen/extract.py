"""Zero-shot attribute extraction with dual encoders (spec §7).

Raw extraction stores the full probability distribution per attribute;
abstention is applied afterwards by :func:`to_evidence` with dev-tuned
thresholds. Scoring is NumPy float32 so it can be tested with a fake encoder;
only :class:`HFDualEncoder` touches torch and transformers.

Output: ``outputs/extraction/{model_slug}/{split}.jsonl``, one record per item:
``{"item_id", "model", "attributes": {attr: {"pred", "confidence", "margin",
"probs": {label: p}}}}``.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from groundedgen import taxonomy as tx
from groundedgen import utils

LOGGER = logging.getLogger(__name__)


def model_slug(model_name: str) -> str:
    """``openai/clip-vit-large-patch14`` → ``openai__clip-vit-large-patch14``."""
    return model_name.replace("/", "__")


def extraction_path(project_dir: str | Path, model_name: str, split: str) -> Path:
    return Path(project_dir) / "outputs" / "extraction" / model_slug(model_name) / f"{split}.jsonl"


class DualEncoder(Protocol):
    """What extraction needs from a model: embeddings and the logit scale."""

    name: str
    logit_scale: float

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


# --------------------------------------------------------------------------
# Hugging Face encoder (GPU or CPU)
# --------------------------------------------------------------------------

def _as_tensor(output: Any, key: str):
    """Embeddings from ``get_*_features``, which may return an output object."""
    if hasattr(output, "shape"):
        return output
    for attr in (key, "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot find embeddings in {type(output).__name__}")


def dtype_kwarg(dtype) -> dict[str, Any]:
    """``from_pretrained`` dtype argument: ``dtype`` from transformers 4.56,
    ``torch_dtype`` before (Colab and local versions differ)."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


class HFDualEncoder:
    """CLIP or SigLIP via ``AutoModel`` / ``AutoProcessor``.

    fp16 on GPU, fp32 on CPU (never bf16). SigLIP text uses
    ``padding="max_length"``; CLIP uses ``padding=True``.
    """

    def __init__(self, name: str, device: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision, **dtype_kwarg(self.dtype))
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.padding: str | bool = "max_length" if "siglip" in name.lower() else True
        self.logit_scale = float(self.model.logit_scale.detach().float().exp().item())

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        inputs = self.processor(text=list(texts), padding=self.padding,
                                truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        return _as_tensor(out, "text_embeds").float().cpu().numpy()

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        return _as_tensor(out, "image_embeds").float().cpu().numpy()

    def unload(self) -> None:
        """Free the model and empty the CUDA cache (one large model at a time)."""
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def class_embeddings(encoder: DualEncoder) -> dict[str, tuple[list[str], np.ndarray]]:
    """Per attribute: labels and their prompt-ensembled embeddings [C, D].

    Each class's 4 templates are encoded, L2-normalised, averaged and
    re-normalised. Computed once per model.
    """
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for attribute in tx.ATTRIBUTES:
        prompts = tx.clip_prompts(attribute)
        labels = list(prompts)
        n_templates = len(tx.CLIP_TEMPLATES[attribute])
        flat = [p for label in labels for p in prompts[label]]
        emb = _l2_normalise(encoder.encode_text(flat))
        emb = emb.reshape(len(labels), n_templates, emb.shape[-1])
        out[attribute] = (labels, _l2_normalise(emb.mean(axis=1)))
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float32."""
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def score_embeddings(
    image_emb: np.ndarray,
    classes: Mapping[str, tuple[list[str], np.ndarray]],
    logit_scale: float,
) -> list[dict[str, dict[str, Any]]]:
    """Attribute predictions for a batch of image embeddings [N, D].

    logits = logit_scale × cosine similarity; probabilities = softmax over the
    attribute's classes. ``margin`` is top-1 minus top-2 probability.
    """
    img = _l2_normalise(image_emb)
    results: list[dict[str, dict[str, Any]]] = [{} for _ in range(len(img))]
    for attribute, (labels, cls) in classes.items():
        probs = softmax(logit_scale * img @ cls.T)
        for i, p in enumerate(probs):
            order = np.argsort(-p, kind="stable")
            results[i][attribute] = {
                "pred": labels[order[0]],
                "confidence": float(p[order[0]]),
                "margin": float(p[order[0]] - p[order[1]]),
                "probs": {label: float(v) for label, v in zip(labels, p)},
            }
    return results


def run_extraction(
    cfg: Mapping[str, Any],
    encoder: DualEncoder,
    split: str,
    manifest: pd.DataFrame,
    load_image=None,
) -> Path:
    """Extract every pending item of ``split``; finished item ids are skipped.

    Args:
        encoder: a loaded :class:`HFDualEncoder` (or a fake in tests).
        manifest: rows with ``item_id``, ``split`` and ``image_path``.
        load_image: ``image_path -> PIL image``; defaults to opening the file
            under ``project_dir`` as RGB.
    """
    project_dir = Path(cfg["project_dir"])
    out_path = extraction_path(project_dir, encoder.name, split)
    done = {key[0] for key in utils.done_keys(out_path, ["item_id"])}
    pending = manifest[(manifest["split"] == split) & ~manifest["item_id"].isin(done)]
    if pending.empty:
        LOGGER.info("%s %s: all %d items done", encoder.name, split, len(done))
        return out_path

    if load_image is None:
        from PIL import Image

        def load_image(rel: str):
            with Image.open(project_dir / rel) as im:
                return im.convert("RGB")

    classes = class_embeddings(encoder)
    batch_size = int(cfg["extraction"]["batch_size"])
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start:start + batch_size]
        emb = encoder.encode_images([load_image(p) for p in batch["image_path"]])
        scored = score_embeddings(emb, classes, encoder.logit_scale)
        utils.append_jsonl(out_path, (
            {"item_id": item_id, "model": encoder.name, "attributes": attrs}
            for item_id, attrs in zip(batch["item_id"], scored)
        ))
        LOGGER.info("%s %s: %d/%d", encoder.name, split, start + len(batch), len(pending))
    return out_path


# --------------------------------------------------------------------------
# Evidence JSON
# --------------------------------------------------------------------------

def to_evidence(
    record: Mapping[str, Any], thresholds: Mapping[str, float | None] | None = None
) -> dict[str, str]:
    """``{"category", "color", "material"}`` from an extraction record.

    With ``thresholds`` (B-Abs, APG) an attribute whose confidence is below its
    threshold becomes ``unknown``; a threshold of ``None`` or infinity always
    abstains. Without thresholds (condition B) every attribute is the top
    prediction.
    """
    evidence: dict[str, str] = {}
    for attribute in tx.ATTRIBUTES:
        attr = record["attributes"][attribute]
        if thresholds is None:
            evidence[attribute] = attr["pred"]
            continue
        t = thresholds[attribute]
        t = math.inf if t is None else float(t)
        evidence[attribute] = attr["pred"] if attr["confidence"] >= t else tx.UNKNOWN
    return evidence


# --------------------------------------------------------------------------
# Command line / notebook entry point
# --------------------------------------------------------------------------

def run_all(cfg: Mapping[str, Any], splits: Sequence[str] = ("dev", "test")) -> None:
    """Run every configured model on ``splits``, one model in memory at a time,
    and record each checkpoint's revision in ``outputs/run_info.json``."""
    from groundedgen import data

    manifest = data.load_manifest(cfg)
    run_info = utils.project_path(cfg, "outputs", "run_info.json")
    for name in cfg["extraction"]["models"]:
        pending = [s for s in splits
                   if len(utils.read_jsonl(extraction_path(cfg["project_dir"], name, s)))
                   < int((manifest["split"] == s).sum())]
        if not pending:
            LOGGER.info("%s: nothing pending", name)
            continue
        encoder = HFDualEncoder(name)
        utils.update_run_info(run_info, {"model_revisions": {name: encoder.revision}})
        for split in pending:
            run_extraction(cfg, encoder, split, manifest)
        encoder.unload()


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.extract [extra.yaml ...] [--evaluate]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("configs", nargs="*", help="YAML files merged over the default")
    parser.add_argument("--evaluate", action="store_true",
                        help="also tune thresholds on dev and write T2/F2 (needs annotated gold)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    utils.apply_runtime_env(cfg)
    utils.set_seed(cfg["seed"])
    run_all(cfg)
    if args.evaluate:
        from groundedgen import data, eval_extraction

        summary = eval_extraction.run_evaluation(cfg, data.load_manifest(cfg))
        print(json.dumps({k: summary[k] for k in ("primary_extractor", "dev_material_aurc")},
                         indent=2))


if __name__ == "__main__":
    main()
