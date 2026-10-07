"""PLM adapters that make third-party protein language models look like AtlasLM.

Each adapter duck-types ``atlaslm.model.AtlasLM`` so that AtlasFold's
``run_lm_embedder`` and folding trunk run unchanged. See :mod:`.base`.
"""

from .base import AdapterAlphabet, PLMAdapter
from .prott5 import PROTT5_MASK_TOKEN, PROTT5_NAME, ProtT5Adapter, ProtT5Alphabet

__all__ = [
    "AdapterAlphabet",
    "PLMAdapter",
    "ProtT5Adapter",
    "ProtT5Alphabet",
    "PROTT5_MASK_TOKEN",
    "PROTT5_NAME",
]
