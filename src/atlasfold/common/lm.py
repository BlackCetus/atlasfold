"""Central registry mapping ``lm_name`` to a backbone and a tokenizer alphabet.

Both the folding model (``atlasfold.model.model.AtlasFold``) and the data
pipeline (``atlasfold.train.*.dataset``) must agree on which protein language
model is used. This module is the single source of truth:

* :func:`build_lm` returns the frozen backbone (``atlaslm.model.AtlasLM`` or a
  :class:`atlaslm.adapters.PLMAdapter` such as ``ProtT5Adapter``).
* :func:`get_lm_alphabet` returns the matching tokenizer alphabet used by
  ``atlasfold.common.featurize.featurize``.

Heavy / optional imports (``transformers``) are deferred into the ProtT5 branch
so ESM-3-only usage needs no extra dependencies.
"""

from pathlib import Path

import torch


def _is_prott5(lm_name: str) -> bool:
    n = lm_name.lower()
    return "prott5" in n or "prot_t5" in n


def get_lm_alphabet(lm_name: str = "atlaslm-3b", *, cache_dir: str | Path | None = None):
    """Return the tokenizer alphabet matching ``lm_name`` (for featurization)."""
    if _is_prott5(lm_name):
        from transformers import T5Tokenizer

        from atlaslm.adapters.prott5 import PROTT5_NAME, ProtT5Alphabet

        name = lm_name if "/" in lm_name else PROTT5_NAME
        tokenizer = T5Tokenizer.from_pretrained(name, cache_dir=cache_dir)
        return ProtT5Alphabet(tokenizer)

    from atlaslm.alphabet import Alphabet

    return Alphabet()


def build_lm(
    lm_name: str = "atlaslm-3b",
    lm_path: str | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
):
    """Build the frozen language-model backbone selected by ``lm_name``."""
    if _is_prott5(lm_name):
        from atlaslm.adapters.prott5 import PROTT5_NAME, ProtT5Adapter

        source = lm_path or (lm_name if "/" in lm_name else PROTT5_NAME)
        return ProtT5Adapter.from_pretrained(
            source, dtype=dtype, device=device, cache_dir=cache_dir
        )

    from atlaslm.model import AtlasLM

    source = Path(lm_path) if lm_path is not None else lm_name
    return AtlasLM.from_pretrained(
        source, dtype=dtype, device=device, cache_dir=cache_dir
    )
