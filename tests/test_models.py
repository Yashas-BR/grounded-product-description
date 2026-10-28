"""Chat-message logic of models.py (no model is loaded)."""

from __future__ import annotations

from groundedgen import models
from groundedgen.prompts import ChatPrompt

P = ChatPrompt("SYSTEM TEXT", "USER TEXT")


def _template(messages, tokenize=False, add_generation_prompt=True):
    return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def _rejects_system(messages, **kwargs):
    if any(m["role"] == "system" for m in messages):
        raise ValueError("Conversation roles must alternate user/assistant")
    return _template(messages)


def _drops_system(messages, **kwargs):
    return _template([m for m in messages if m["role"] != "system"])


def test_chat_messages_shapes():
    assert models.chat_messages(P) == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    assert models.chat_messages(P, merge_system=True) == [
        {"role": "user", "content": "SYSTEM TEXT\n\nUSER TEXT"},
    ]
    image = models.chat_messages(ChatPrompt(None, "U", "x.jpg"), image_token=True)
    assert image == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "U"}]}]


def test_render_chat_keeps_supported_system_role():
    text, merged = models.render_chat(_template, P)
    assert text == "<system>SYSTEM TEXT<user>USER TEXT<assistant>" and not merged


def test_render_chat_merges_when_template_rejects_system():
    text, merged = models.render_chat(_rejects_system, P)
    assert merged and text == "<user>SYSTEM TEXT\n\nUSER TEXT<assistant>"


def test_render_chat_merges_when_template_drops_system():
    text, merged = models.render_chat(_drops_system, P)
    assert merged and "SYSTEM TEXT" in text


def test_render_chat_without_system():
    text, merged = models.render_chat(_rejects_system, ChatPrompt(None, "U"))
    assert text == "<user>U<assistant>" and not merged


def test_generator_slug():
    assert models.generator_slug("Qwen/Qwen2.5-7B-Instruct") == "Qwen__Qwen2.5-7B-Instruct"


def test_reset_peak_memory_skips_uninitialised_cuda(monkeypatch):
    # Kaggle's torch raises "Invalid device argument" if CUDA isn't initialised yet.
    import torch

    def boom(device=None):
        raise RuntimeError("Invalid device argument")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", boom)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    models.reset_peak_memory()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    models.reset_peak_memory()  # still swallowed after initialisation
