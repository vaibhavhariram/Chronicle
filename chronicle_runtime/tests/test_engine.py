"""Tests for batched inference engine."""

import pytest

from chronicle_runtime.runtime.engine import batch_generate


def test_batch_generate_returns_texts_for_2_prompts():
    """batch_generate returns one text per prompt for a batch of 2."""
    prompts = ["Hello", "The capital of France is"]
    max_new_tokens_list = [5, 5]
    texts, tokens_per_req, prefill_ms, decode_ms = batch_generate(
        prompts, max_new_tokens_list, seed=42
    )
    assert len(texts) == 2
    assert len(texts[0]) > 0
    assert len(texts[1]) > 0
    assert "Hello" in texts[0]
    assert "The capital of France is" in texts[1]
    assert len(tokens_per_req) == 2
    assert prefill_ms >= 0
    assert decode_ms >= 0


def test_batch_generate_deterministic_with_seed():
    """Same seed produces identical output."""
    prompts = ["Once upon a time"]
    max_new_tokens_list = [10]
    texts1, _, _, _ = batch_generate(prompts, max_new_tokens_list, seed=123)
    texts2, _, _, _ = batch_generate(prompts, max_new_tokens_list, seed=123)
    assert texts1 == texts2


def test_batch_generate_respects_per_request_max_tokens():
    """Different max_new_tokens per request produces different lengths."""
    prompts = ["Hi", "Hello"]
    max_new_tokens_list = [2, 8]
    texts, tokens_per_req, _, _ = batch_generate(
        prompts, max_new_tokens_list, seed=42
    )
    assert len(texts) == 2
    assert tokens_per_req[0] <= tokens_per_req[1]
