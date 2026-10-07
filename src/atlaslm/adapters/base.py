"""Base class for protein-language-model adapters used by AtlasFold.

AtlasFold consumes its language model through a very specific, internals-level
contract (see ``atlasfold.model.model.AtlasFold.run_lm_embedder``). It does *not*
call a single ``forward``; instead it:

1. reads ``lm.d_model``, ``lm.n_heads``, ``lm.n_layers`` to size its learned
   adapters (``lm_layer_weights``, ``proj_lm_attn``, ``lm_emb_to_s_lm`` ...),
2. reads ``lm.alphabet`` (``aa_idxs``, ``mask_idx``, ``pad_idx``) for MLM masking,
3. uses ``lm.embed(input_ids)`` as the layer-0 representation, then
4. iterates ``lm.transformer.blocks`` and calls each block as
   ``block(x, seq_id, pos_id, return_attn_logits=True) -> (hidden, attn_logits)``
   where ``attn_logits`` are the *pre-softmax* attention logits of shape
   ``[B, n_heads, S, S]``.

A ``PLMAdapter`` subclass makes an arbitrary PLM look exactly like
``atlaslm.model.AtlasLM`` so that ``run_lm_embedder`` and the whole folding trunk
stay byte-for-byte unchanged. This is deliberately faithful to AtlasFold's
workflow rather than convenient: every subclass must expose per-layer residual
hidden states *and* per-layer pre-softmax attention logits, honour ``pos_id`` and
``seq_id``, and keep the backbone frozen and deterministic.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class AdapterAlphabet(Protocol):
    """The subset of ``atlaslm.alphabet.Alphabet`` that AtlasFold relies on.

    ``run_lm_embedder`` uses ``aa_idxs`` and ``mask_idx``; the data featurizer
    uses ``pad_idx``/``eos_idx``/``bos_idx``/``encode``. ``n_prefix``/``n_suffix``
    describe how many special tokens wrap the residues (e.g. AtlasLM uses
    ``<cls>`` + residues + ``<eos>`` => (1, 1); ProtT5 uses residues + ``</s>``
    => (0, 1)), which the crop/``seq_tok_idx`` logic needs.
    """

    pad_idx: int
    mask_idx: int
    aa_idxs: list[int]
    n_prefix: int
    n_suffix: int

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]: ...


class PLMAdapter(nn.Module):
    """Duck-types ``atlaslm.model.AtlasLM`` for a third-party PLM.

    Subclasses must set the integer attributes ``d_model``, ``n_heads`` and
    ``n_layers``, assign an :class:`AdapterAlphabet` to ``self.alphabet``, expose
    a ``self.transformer`` object whose ``.blocks`` is an iterable of length
    ``n_layers`` of callables, and implement :meth:`embed`.

    Each block callable must accept ``(x, seq_id, pos_id, return_attn_logits)``
    and return ``(hidden_state, attn_logits_or_None)`` with shapes
    ``[B, S, d_model]`` and ``[B, n_heads, S, S]`` respectively.
    """

    d_model: int
    n_heads: int
    n_layers: int
    alphabet: AdapterAlphabet

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the layer-0 token embedding of shape ``[B, S, d_model]``."""
        raise NotImplementedError
