"""Fail-fast FlashAttention stub used only with ``--no_flash_attn``."""


def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
    raise RuntimeError(
        "FlashAttention is disabled in the T4 profile; keep --no_flash_attn enabled"
    )
