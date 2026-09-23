"""Fail-fast padding stubs used only with ``--no_flash_attn``."""


def unpad_input(*args, **kwargs):
    raise RuntimeError(
        "FlashAttention is disabled in the T4 profile; keep --no_flash_attn enabled"
    )


def pad_input(*args, **kwargs):
    raise RuntimeError(
        "FlashAttention is disabled in the T4 profile; keep --no_flash_attn enabled"
    )
