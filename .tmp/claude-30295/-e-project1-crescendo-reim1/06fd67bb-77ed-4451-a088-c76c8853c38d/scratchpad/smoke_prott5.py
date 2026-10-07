"""CPU-only smoke test for ProtT5Adapter using a TINY random T5 encoder.

No network, no GPU, no multi-GB weights. Validates that the adapter's
block-by-block iteration reproduces the real T5 encoder, that pos_id-aware
bias matches compute_bias, and that attention-logit capture has the right shape.
"""

import torch
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel

import sys
sys.path.insert(0, "src")
from atlaslm.adapters.prott5 import ProtT5Adapter


class FakeTok:
    """Minimal tokenizer stub so ProtT5Alphabet can build on a random model."""
    vocab_size = 130
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def convert_tokens_to_ids(self, tok):
        # Map residue pieces to ids 3.., mask token to a reserved slot.
        table = {"▁" + a: 3 + i for i, a in enumerate("ALGVSRETIDPKFQNYMHWCXBOUZ")}
        table["<extra_id_0>"] = 127
        return table.get(tok, self.unk_token_id)


def main() -> None:
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=130, d_model=32, d_kv=8, d_ff=64, num_layers=3, num_heads=4,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
        is_encoder_decoder=False, feed_forward_proj="gated-gelu", dropout_rate=0.0,
    )
    enc = T5EncoderModel(cfg).eval()
    adapter = ProtT5Adapter(enc, FakeTok()).eval()

    B, S = 2, 7
    ids = torch.randint(3, 28, (B, S))
    ones = torch.ones(B, S, dtype=torch.long)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)

    # (1) embed == encoder token embedding
    emb = adapter.embed(ids)
    ref_emb = adapter._stack().embed_tokens(ids)
    assert torch.equal(emb, ref_emb), "embed mismatch"

    # (2) pos_id-aware bias (arange) == T5.compute_bias
    sa = adapter._stack().block[0].layer[0].SelfAttention
    cb = sa.compute_bias(S, S)                       # [1, H, S, S]
    mine = adapter._position_bias(torch.arange(S).unsqueeze(0), torch.float32)
    assert torch.allclose(mine[0], cb[0], atol=1e-6), "relative bias mismatch"

    # (3) block iteration reproduces the real encoder (+ attn logit shape)
    with torch.no_grad():
        x = adapter.embed(ids)
        for blk in adapter.transformer.blocks:
            x, attn = blk(x, seq_id=ones, pos_id=pos, return_attn_logits=True)
            assert attn.shape == (B, adapter.n_heads, S, S), attn.shape
        x = adapter._stack().final_layer_norm(x)
        ref = enc(input_ids=ids, attention_mask=ones).last_hidden_state
    max_err = (x - ref).abs().max().item()
    assert max_err < 1e-4, f"hidden-state reproduction error too large: {max_err}"

    # (4) masking (padding present) stays finite, no all -inf rows
    seq_id = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        x = adapter.embed(ids)
        for blk in adapter.transformer.blocks:
            x, attn = blk(x, seq_id=seq_id, pos_id=pos, return_attn_logits=True)
        # masked entries are -inf (as AtlasLM expects before nan_to_num)
        assert torch.isinf(attn).any(), "expected -inf at masked entries"
    assert torch.isfinite(x).all(), "hidden states went non-finite under masking"

    print(f"OK  n_layers={adapter.n_layers} n_heads={adapter.n_heads} "
          f"d_model={adapter.d_model}  hidden-repro max_err={max_err:.2e}")
    print(f"alphabet: pad={adapter.alphabet.pad_idx} eos={adapter.alphabet.eos_idx} "
          f"mask={adapter.alphabet.mask_idx} aa_idxs={adapter.alphabet.aa_idxs}")


if __name__ == "__main__":
    main()
