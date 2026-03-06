"""Batched inference engine with KV-cache reuse."""

from typing import Optional

import torch

from chronicle_runtime.runtime.model import load_model


def _set_seed(seed: int = 42) -> None:
    """Set deterministic seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_generate(
    prompts: list[str],
    max_new_tokens_list: list[int],
    seed: Optional[int] = 42,
) -> list[str]:
    """
    Batched greedy decoding with KV-cache reuse.

    - Tokenize prompts as batch with left padding
    - Prefill: one forward pass with use_cache=True
    - Decode token-by-token up to max(max_new_tokens)
    - Per-request limits and EOS handling
    """
    if not prompts:
        return []
    if len(prompts) != len(max_new_tokens_list):
        raise ValueError("prompts and max_new_tokens_list must have same length")

    _set_seed(seed)
    tokenizer, model = load_model()
    device = next(model.parameters()).device

    # Ensure pad token for batch padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize with left padding (causal LM: last token = actual last of prompt)
    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    max_len = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024)
    )
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    tokenizer.padding_side = orig_padding
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Per-request state
    generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    max_steps = max(max_new_tokens_list)

    # Prefill
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    # Next token from last position
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)

    for i in range(batch_size):
        generated_ids[i].append(next_tokens[i].item())
        if (
            next_tokens[i].item() == eos_token_id
            or len(generated_ids[i]) >= max_new_tokens_list[i]
        ):
            finished[i] = True

    # Decode loop
    for _ in range(max_steps - 1):
        if all(finished):
            break

        # Input: last token per sequence; use pad for finished
        next_input = next_tokens.unsqueeze(-1).clone()
        for i in range(batch_size):
            if finished[i]:
                next_input[i] = pad_token_id

        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        next_tokens = torch.argmax(logits[:, 0, :], dim=-1)

        for i in range(batch_size):
            if not finished[i]:
                generated_ids[i].append(next_tokens[i].item())
                if (
                    next_tokens[i].item() == eos_token_id
                    or len(generated_ids[i]) >= max_new_tokens_list[i]
                ):
                    finished[i] = True

        for i in range(batch_size):
            if finished[i]:
                next_tokens[i] = pad_token_id

    # Decode: prompt tokens (non-pad) + generated
    texts = []
    for i in range(batch_size):
        prompt_len = attention_mask[i].sum().item()
        prompt_ids = input_ids[i, -int(prompt_len) :].tolist()
        full_ids = prompt_ids + generated_ids[i]
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        texts.append(text)

    return texts
