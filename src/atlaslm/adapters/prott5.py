"""ProtT5 adapter.

Makes ``Rostlab/prot_t5_xl_uniref50`` (a T5 encoder) look exactly like
``atlaslm.model.AtlasLM`` so AtlasFold's ``run_lm_embedder`` and folding trunk
run unchanged. See :mod:`atlaslm.adapters.base` for the contract.

Fidelity decisions (mirror AtlasFold, not convenience):

* **Pre-softmax attention logits.** AtlasFold consumes raw pre-softmax logits
  (it clamps to +-100 and divides by 100). We capture T5's own pre-softmax
  scores ``q.kT * scaling + position_bias + mask`` (``scaling == 1.0`` for T5),
  computed from the pre-norm hidden state exactly as the real block does.
* **Residual-stream hidden states, no final norm.** Layer 0 is the raw token
  embedding; layers 1..N are the post-block residual stream. The final
  ``T5Stack.final_layer_norm`` is *not* applied, matching AtlasLM which never
  applies its own final norm inside ``run_lm_embedder``.
* **``pos_id`` / ``seq_id`` honoured.** T5's relative-position bias is recomputed
  from ``pos_id`` differences (true residue separation across crops), and the
  attention mask is ``seq_id[i] == seq_id[j]`` with ``-inf`` at masked entries so
  AtlasFold's ``nan_to_num(neginf=0)`` zeroes them, identical to AtlasLM.
* **Frozen and deterministic.** The encoder is forced into ``eval`` (dropout off)
  and kept there even when the parent model is in ``train`` mode.
* **Masking token.** ProtT5 ships no ``[MASK]`` (both ``[MASK]`` and ``<mask>``
  fold to ``<unk>``); its only reserved single-token candidates are ``<unk>`` and
  ``<extra_id_N>``. We use ``<extra_id_0>`` as a dedicated, non-colliding mask.
"""

from pathlib import Path

import torch

from .base import PLMAdapter

# SentencePiece metaspace prefix on every ProtT5 residue piece (e.g. "_M").
_META = "▁"
# Residue pieces present in the ProtT5 vocab (standard 20 + X, B, O, U, Z).
_RESIDUES = "ALGVSRETIDPKFQNYMHWCXBOUZ"
# ProtT5 has no [MASK]; <extra_id_0> is the reserved surrogate (see module docstring).
PROTT5_MASK_TOKEN = "<extra_id_0>"
PROTT5_NAME = "Rostlab/prot_t5_xl_uniref50"


class ProtT5Alphabet:
    """``atlaslm.alphabet.Alphabet``-shaped view over the ProtT5 tokenizer."""

    def __init__(self, tokenizer, mask_token: str = PROTT5_MASK_TOKEN) -> None:
        self.tokenizer = tokenizer
        self.pad_idx: int = tokenizer.pad_token_id
        self.eos_idx: int = tokenizer.eos_token_id
        self.unk_idx: int = tokenizer.unk_token_id
        self.bos_idx: int | None = None  # ProtT5 has no BOS/CLS token.

        self.res_to_idx: dict[str, int] = {}
        for aa in _RESIDUES:
            tid = tokenizer.convert_tokens_to_ids(_META + aa)
            if tid != self.unk_idx:
                self.res_to_idx[aa] = tid
        self.aa_idxs: list[int] = sorted(set(self.res_to_idx.values()))

        self.mask_idx: int = tokenizer.convert_tokens_to_ids(mask_token)
        if self.mask_idx == self.unk_idx:
            raise ValueError(
                f"Mask token {mask_token!r} is not in the ProtT5 vocab "
                "(resolved to <unk>); pick a real reserved token."
            )

        # Residues carry no wrapping prefix; a single </s> is appended.
        self.n_prefix: int = 0
        self.n_suffix: int = 1

    def __len__(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        """Encode residue-by-residue (avoids SentencePiece merging unspaced input)."""
        ids = [self.res_to_idx.get(aa, self.unk_idx) for aa in sequence]
        if add_special_tokens:
            ids = ids + [self.eos_idx]  # no BOS; </s> suffix
        return ids


class _T5BlockCaller:
    """Callable with AtlasLM's block signature, backed by one real ``T5Block``."""

    def __init__(self, adapter: "ProtT5Adapter", idx: int) -> None:
        self.adapter = adapter
        self.idx = idx

    def __call__(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
        return_attn: bool = False,
        return_attn_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.adapter._run_block(
            self.idx, x, seq_id, pos_id, return_attn or return_attn_logits
        )


class _TransformerView:
    """Lightweight stand-in for ``AtlasLM.transformer`` exposing ``.blocks``."""

    def __init__(self, blocks: list[_T5BlockCaller]) -> None:
        self.blocks = blocks


class ProtT5Adapter(PLMAdapter):
    def __init__(self, encoder, tokenizer, *, mask_token: str = PROTT5_MASK_TOKEN) -> None:
        super().__init__()
        cfg = encoder.config
        # Force eager attention so the pre-softmax capture is arithmetically
        # identical to the hidden-state path (config is shared by all submodules).
        cfg._attn_implementation = "eager"

        self.encoder = encoder  # transformers.T5EncoderModel (registered submodule)
        self.encoder.eval()

        self.d_model: int = cfg.d_model
        self.n_heads: int = cfg.num_heads
        self.n_layers: int = cfg.num_layers
        self.key_value_proj_dim: int = cfg.d_kv

        self.alphabet = ProtT5Alphabet(tokenizer, mask_token=mask_token)
        self.transformer = _TransformerView(
            [_T5BlockCaller(self, i) for i in range(self.n_layers)]
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _stack(self):
        return self.encoder.encoder  # transformers.T5Stack (encoder-only)

    @property
    def _rab(self):
        # Shared relative-position-bias source lives on block 0's self-attention.
        # Accessed as a property (not stored) to avoid double-registering params.
        return self._stack().block[0].layer[0].SelfAttention

    def train(self, mode: bool = True) -> "ProtT5Adapter":
        super().train(mode)
        self.encoder.eval()  # keep the frozen LM deterministic (dropout off)
        return self

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._stack().embed_tokens(input_ids)

    def _position_bias(self, pos_id: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """T5 relative-position bias from explicit positions -> [B, H, Sq, Sk]."""
        sa = self._rab
        # memory_position - context_position, matching T5Attention.compute_bias.
        rel = pos_id[:, None, :] - pos_id[:, :, None]  # [B, Sq, Sk]
        bucket = sa._relative_position_bucket(
            rel,
            bidirectional=True,  # encoder self-attention
            num_buckets=sa.relative_attention_num_buckets,
            max_distance=sa.relative_attention_max_distance,
        )
        values = sa.relative_attention_bias(bucket)  # [B, Sq, Sk, H]
        return values.permute(0, 3, 1, 2).to(dtype)  # [B, H, Sq, Sk]

    def _additive_mask(self, seq_id: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``seq_id[i] == seq_id[j]`` block mask with -inf at masked -> [B, 1, S, S]."""
        allowed = seq_id[:, :, None] == seq_id[:, None, :]  # [B, S, S]
        add = torch.zeros(allowed.shape, dtype=dtype, device=seq_id.device)
        add.masked_fill_(~allowed, float("-inf"))
        return add[:, None]

    def _run_block(
        self,
        idx: int,
        x: torch.Tensor,
        seq_id: torch.Tensor | None,
        pos_id: torch.Tensor | None,
        return_attn_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S, _ = x.shape
        device = x.device
        if seq_id is None:
            seq_id = torch.ones(B, S, dtype=torch.long, device=device)
        if pos_id is None:
            pos_id = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        pos_id = pos_id.long()

        rel_bias = self._position_bias(pos_id, x.dtype)  # [B, H, S, S]
        add_mask = self._additive_mask(seq_id, x.dtype)  # [B, 1, S, S]

        block = self._stack().block[idx]
        # Real block forward: scores = q.kT*scaling + position_bias + attention_mask.
        hidden = block(x, attention_mask=add_mask, position_bias=rel_bias)[0]

        attn_logits = None
        if return_attn_logits:
            sa = block.layer[0].SelfAttention
            normed = block.layer[0].layer_norm(x)  # T5 is pre-norm
            shape = (B, S, self.n_heads, self.key_value_proj_dim)
            q = sa.q(normed).view(shape).transpose(1, 2)  # [B, H, S, Dh]
            k = sa.k(normed).view(shape).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(2, 3)) * sa.scaling  # scaling == 1.0
            attn_logits = scores + rel_bias + add_mask  # [B, H, S, S], pre-softmax

        return hidden, attn_logits

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path = PROTT5_NAME,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
        mask_token: str = PROTT5_MASK_TOKEN,
    ) -> "ProtT5Adapter":
        from transformers import T5EncoderModel, T5Tokenizer

        name = str(name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(name, cache_dir=cache_dir)
        encoder = T5EncoderModel.from_pretrained(
            name, cache_dir=cache_dir, attn_implementation="eager"
        )
        model = cls(encoder, tokenizer, mask_token=mask_token)
        if dtype is not None:
            model = model.to(dtype)
        model = model.to(device)
        model.encoder.eval()
        return model
