"""Synthetic prompt generation for benchmarking."""

from chronicle_runtime.runtime.model import load_model


def make_prompts(
    n: int,
    fixed_length_tokens: int | None = 64,
    varied: bool = False,
) -> list[str]:
    """
    Generate synthetic prompts.

    - fixed_length_tokens: if set, all prompts have ~this many tokens
    - varied: if True, use varied lengths (16, 32, 64, 128 tokens)
    """
    tokenizer, _ = load_model()
    prompts = []

    if varied:
        lengths = [16, 32, 64, 128]
        base = "The quick brown fox jumps over the lazy dog. " * 4
        for i in range(n):
            target = lengths[i % len(lengths)]
            # Pad or truncate to reach target
            tokens = tokenizer.encode(base, return_tensors=None)
            if len(tokens) < target:
                tokens = tokens * (target // len(tokens) + 1)
            tokens = tokens[:target]
            prompts.append(tokenizer.decode(tokens))
    else:
        target = fixed_length_tokens or 64
        base = "The quick brown fox jumps over the lazy dog. " * 8
        tokens = tokenizer.encode(base, return_tensors=None)
        if len(tokens) < target:
            tokens = tokens * (target // len(tokens) + 1)
        template_tokens = tokens[:target]
        template = tokenizer.decode(template_tokens)
        for i in range(n):
            prompts.append(template)

    return prompts
