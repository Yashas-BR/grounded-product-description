"""Model loaders for generation and judging (spec §2, §8).

* 7B models load in 4-bit NF4 with fp16 compute; Qwen2-VL-2B loads in fp16.
* Greedy decoding, ``max_new_tokens`` from the config, left padding.
* At most one large model is in memory: load, process everything, ``unload``.
* If a chat template rejects the system role, the system text is prepended to
  the first user message and the generator records ``system_merged = True``.

Everything that needs torch/transformers is imported lazily; the chat-message
logic is plain Python and unit-tested.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from groundedgen.prompts import ChatPrompt

LOGGER = logging.getLogger(__name__)


class Generator(Protocol):
    """Anything that turns prompts into raw output strings."""

    name: str
    system_merged: bool

    def generate(self, prompts: Sequence[ChatPrompt]) -> list[str]: ...

    def unload(self) -> None: ...


def generator_slug(name: str) -> str:
    """``Qwen/Qwen2.5-7B-Instruct`` → ``Qwen__Qwen2.5-7B-Instruct``."""
    return name.replace("/", "__")


# --------------------------------------------------------------------------
# Chat messages
# --------------------------------------------------------------------------

def chat_messages(prompt: ChatPrompt, merge_system: bool = False,
                  image_token: bool = False) -> list[dict[str, Any]]:
    """Messages for ``apply_chat_template``.

    Args:
        merge_system: put the system text in front of the user text instead of
            sending a system message (for templates that reject the role).
        image_token: use the multimodal content format with one image slot.
    """
    user = prompt.user
    messages: list[dict[str, Any]] = []
    if prompt.system:
        if merge_system:
            user = f"{prompt.system}\n\n{user}"
        else:
            messages.append({"role": "system", "content": _content(prompt.system, False)})
    messages.append({"role": "user", "content": _content(user, image_token)})
    return messages


def _content(text: str, image: bool) -> Any:
    if not image:
        return text
    return [{"type": "image"}, {"type": "text", "text": text}]


def render_chat(apply_template: Callable[..., str], prompt: ChatPrompt,
                image_token: bool = False) -> tuple[str, bool]:
    """Render ``prompt`` with a chat template, merging the system text if needed.

    Returns the rendered string and whether the system text had to be merged.
    A template "rejects" the system role if it raises, or if it silently drops
    the system text (both happen with some Mistral versions).
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if prompt.system:
        try:
            text = apply_template(chat_messages(prompt, image_token=image_token), **kwargs)
            if prompt.system in text:
                return text, False
        except Exception as exc:  # template-specific errors (jinja TemplateError etc.)
            LOGGER.info("Chat template rejected the system role (%s); merging", exc)
    text = apply_template(chat_messages(prompt, merge_system=True, image_token=image_token),
                          **kwargs)
    return text, bool(prompt.system)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def quantization_config(cfg: Mapping[str, Any]):
    """``BitsAndBytesConfig`` for 4-bit NF4 with fp16 compute (never bf16)."""
    import torch
    from transformers import BitsAndBytesConfig

    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )


def _load_kwargs(cfg: Mapping[str, Any], quantize: bool) -> dict[str, Any]:
    import torch

    from groundedgen.extract import dtype_kwarg

    kwargs: dict[str, Any] = {"device_map": "auto", **dtype_kwarg(torch.float16)}
    if quantize:
        kwargs["quantization_config"] = quantization_config(cfg)
    return kwargs


def _revision(model) -> str | None:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def free_memory() -> None:
    """Collect garbage and empty the CUDA cache."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_memory() -> None:
    """Reset the peak-memory counters of every GPU (call before loading a model)."""
    try:
        import torch
    except ImportError:
        return
    # Before CUDA is initialised the counters are zero, and newer torch raises
    # "Invalid device argument" when asked to reset them.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except RuntimeError as exc:
                LOGGER.debug("Could not reset peak memory on GPU %d: %s", device, exc)


def peak_gpu_gb() -> float | None:
    """Peak allocated memory since the last reset, summed over all GPUs, in GB.

    ``device_map="auto"`` can split a model across GPUs (Kaggle T4 ×2), so the
    sum is what a single 16 GB T4 would have had to hold. ``None`` on CPU.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30


def _decode_new(tokenizer, output_ids, input_len: int) -> list[str]:
    return tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)


class HFTextGenerator:
    """Instruction-tuned LLM in 4-bit (B, B-Abs, APG, C, C-Mask, judge)."""

    def __init__(self, name: str, cfg: Mapping[str, Any], max_new_tokens: int | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.max_new_tokens = max_new_tokens or cfg["generation"]["max_new_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, **_load_kwargs(cfg, quantize=True))
        self.model.eval()
        self.revision = _revision(self.model)
        self.system_merged = False

    def generate(self, prompts: Sequence[ChatPrompt]) -> list[str]:
        import torch

        texts = []
        for p in prompts:
            text, merged = render_chat(self.tokenizer.apply_chat_template, p)
            self.system_merged |= merged
            texts.append(text)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return _decode_new(self.tokenizer, out, inputs["input_ids"].shape[1])

    def unload(self) -> None:
        del self.model
        free_memory()


class HFVisionGenerator:
    """VLM for conditions A and A-F (LLaVA-1.5 4-bit, Qwen2-VL-2B fp16, BLIP-2)."""

    def __init__(self, name: str, cfg: Mapping[str, Any], project_dir: str | Path):
        from transformers import AutoProcessor

        self.name = name
        self.project_dir = Path(project_dir)
        self.max_new_tokens = cfg["generation"]["max_new_tokens"]
        self.kind = ("qwen2vl" if "qwen2-vl" in name.lower()
                     else "blip2" if "blip2" in name.lower() else "llava")
        self.processor = AutoProcessor.from_pretrained(name)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.model = self._load_model(name, cfg)
        self.model.eval()
        self.revision = _revision(self.model)
        self.system_merged = False

    def _load_model(self, name: str, cfg: Mapping[str, Any]):
        if self.kind == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration as Model
            return Model.from_pretrained(name, **_load_kwargs(cfg, quantize=False))
        if self.kind == "blip2":
            from transformers import Blip2ForConditionalGeneration as Model
            return Model.from_pretrained(name, **_load_kwargs(cfg, quantize=False))
        from transformers import LlavaForConditionalGeneration as Model
        return Model.from_pretrained(name, **_load_kwargs(cfg, quantize=True))

    def _text(self, prompt: ChatPrompt) -> str:
        if self.kind == "blip2":  # no chat template: plain question/answer prompt
            body = f"{prompt.system} {prompt.user}" if prompt.system else prompt.user
            self.system_merged |= bool(prompt.system)
            return f"Question: {body} Answer:"
        text, merged = render_chat(self.processor.apply_chat_template, prompt, image_token=True)
        self.system_merged |= merged
        return text

    def generate(self, prompts: Sequence[ChatPrompt]) -> list[str]:
        import torch
        from PIL import Image

        images = []
        for p in prompts:
            with Image.open(self.project_dir / p.image_path) as im:
                images.append(im.convert("RGB"))
        texts = [self._text(p) for p in prompts]
        inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                padding=True).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        return _decode_new(self.processor, out, inputs["input_ids"].shape[1])

    def unload(self) -> None:
        del self.model
        free_memory()


def load_vlm(name: str, cfg: Mapping[str, Any], fallback: str | None = None):
    """Load a VLM; if it fails and ``fallback`` is given, load that instead.

    Returns ``(generator, used_fallback)``. The spec's fallback is BLIP-2 for
    Qwen2-VL; callers record it in DECISIONS.md and ``run_info.json``.
    """
    try:
        return HFVisionGenerator(name, cfg, cfg["project_dir"]), False
    except Exception:
        if fallback is None:
            raise
        LOGGER.exception("Loading %s failed; falling back to %s", name, fallback)
        free_memory()
        return HFVisionGenerator(fallback, cfg, cfg["project_dir"]), True
