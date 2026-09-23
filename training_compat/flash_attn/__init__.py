"""Import-only compatibility shim for Turing GPUs.

The laboratory launcher always passes ``--no_flash_attn``. Simple-TAD still
imports the package at module load time, so these stubs keep that import valid
without pretending FlashAttention is available on a T4.
"""
