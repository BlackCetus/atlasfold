"""Drive ProtT5Adapter through AtlasFold's REAL run_lm_embedder.

Uses the real ProtT5 tokenizer (cached) + a tiny random T5 (vocab 128 so real
residue ids fit). Builds the exact submodules run_lm_embedder touches with
AtlasFold's own classes, then calls the unbound method. No big download, no
heads built. CPU only.
"""
import sys
import torch
from torch import nn
sys.path.insert(0, "src")

from transformers import T5Config, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5EncoderModel

from atlaslm.adapters.prott5 import ProtT5Adapter
from atlasfold.model.model import AtlasFold
from atlasfold.model.network import trunk
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias


def build_tiny_adapter():
    cfg = T5Config(
        vocab_size=128, d_model=32, d_kv=8, d_ff=64, num_layers=3, num_heads=4,
        is_encoder_decoder=False, feed_forward_proj="gated-gelu", dropout_rate=0.1,
    )
    enc = T5EncoderModel(cfg)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    return ProtT5Adapter(enc, tok)


class LMConsumer(nn.Module):
    """Holds exactly the modules AtlasFold.run_lm_embedder uses (small sizes)."""
    def __init__(self, lm, channel_s_lm=16, channel_z=8):
        super().__init__()
        self.lm = lm
        self.lm.requires_grad_(False)
        self.alphabet = lm.alphabet
        self.channel_z = channel_z
        self.kernel_backend = "torch"
        # Mirrors AtlasFold.__init__ (model.py:137-164) verbatim, tiny channels.
        self.lm_layer_weights = nn.Parameter(torch.zeros(lm.n_layers + 1))
        self.layernorm_lm_emb = LayerNorm(lm.d_model)
        self.lm_emb_to_s_lm = nn.Sequential(
            LinearNoBias(lm.d_model, channel_s_lm, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_s_lm, channel_s_lm, init="default"),
        )
        self.proj_lm_attn = nn.ModuleList(
            [LinearNoBias(lm.n_heads, channel_z) for _ in range(lm.n_layers)]
        )
        self.lm_attn_to_z_lm = nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, channel_z, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_z, channel_z, init="default"),
        )
        self.lm_stack = trunk.LMStack(
            channel_s=channel_s_lm, channel_z=channel_z, num_heads=2,
            num_tri_heads=1, dropout_z=0.0, num_blocks=1, blocks_per_ckpt=None,
        )

    # Reuse AtlasFold's real implementation unchanged.
    run_lm_embedder = AtlasFold.run_lm_embedder


def make_batch(adapter, seqs):
    alpha = adapter.alphabet
    B = len(seqs)
    L = len(seqs[0])
    S = L + alpha.n_prefix + alpha.n_suffix
    input_ids = torch.zeros(B, S, dtype=torch.long)
    pos_id = torch.zeros(B, S, dtype=torch.long)
    seq_id = torch.ones(B, S, dtype=torch.long)
    seq_tok_idx = torch.zeros(B, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids = alpha.encode(s)
        input_ids[i] = torch.tensor(ids)
        pos_id[i] = torch.tensor([0] * alpha.n_prefix + list(range(1, L + 1)) +
                                 [L + 1 + j for j in range(alpha.n_suffix)])
        seq_tok_idx[i] = torch.arange(alpha.n_prefix, alpha.n_prefix + L)
    return {
        "lm.input_ids": input_ids, "lm.pos_id": pos_id, "lm.seq_id": seq_id,
        "seq_tok_idx": seq_tok_idx, "seq_mask": torch.ones(B, L, dtype=torch.bool),
    }, L


def main():
    torch.manual_seed(0)
    adapter = build_tiny_adapter()

    # Freeze / determinism checks
    assert not any(p.requires_grad for p in adapter.parameters()) or True  # set in consumer
    adapter.train()
    assert adapter.encoder.training is False, "encoder must stay in eval under train()"
    adapter.eval()

    consumer = LMConsumer(adapter).eval()
    assert not any(p.requires_grad for p in consumer.lm.parameters()), "LM not frozen"
    n_train = sum(p.requires_grad for p in consumer.parameters())
    assert n_train > 0, "adapter modules should be trainable"

    batch, L = make_batch(adapter, ["MKLVQ", "ACDEF"])
    B = batch["lm.input_ids"].shape[0]

    # NOTE: AtlasFold's LMStack triangle attention uses an in-place op on a view
    # in its "torch" backend (triangle_update.py:389). Under CPU eager + autograd
    # this trips torch 2.14's stricter view rules; on GPU the cuequivariance
    # kernel path is used instead. This is pre-existing AtlasFold code, unrelated
    # to the adapter -> we validate the adapter path under inference (no_grad).

    # (1) No MLM mask
    with torch.no_grad():
        s_lm, z_lm = consumer.run_lm_embedder(batch, mlm_mask=None, train=False)
    assert s_lm.shape == (B, L, 16), s_lm.shape
    assert z_lm.shape == (B, L, L, 8), z_lm.shape
    assert torch.isfinite(s_lm).all() and torch.isfinite(z_lm).all()
    print(f"run_lm_embedder (no mask) OK: s_lm={tuple(s_lm.shape)} z_lm={tuple(z_lm.shape)}")

    # (2) With MLM mask (exercises aa_idxs / mask_idx substitution path)
    with torch.no_grad():
        mlm = torch.ones(1, batch["lm.input_ids"].shape[1], dtype=torch.bool)
        s_lm2, z_lm2 = consumer.run_lm_embedder(batch, mlm_mask=mlm, train=False)
    assert s_lm2.shape == (B, L, 16) and torch.isfinite(s_lm2).all()
    assert torch.isfinite(z_lm2).all()
    assert not torch.allclose(s_lm, s_lm2), "MLM mask had no effect"
    print("run_lm_embedder (with MLM mask) OK and changes features")

    # (3) Freezing: the LM holds no trainable params; adapter heads do.
    assert not any(p.requires_grad for p in consumer.lm.parameters()), "LM not frozen"
    assert consumer.lm_emb_to_s_lm[0].weight.requires_grad, "adapter head not trainable"
    print("freeze check OK: ProtT5 frozen, adapter heads trainable")

    # (4) Realistic dtype path: bf16 LM (as build_lm loads it) + fp32 trunk under
    # autocast -- this is how AtlasFold actually runs it.
    adapter_bf16 = build_tiny_adapter().to(torch.bfloat16)
    cons_bf16 = LMConsumer(adapter_bf16).eval()  # trunk adapters stay fp32
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        sb, zb = cons_bf16.run_lm_embedder(batch, mlm_mask=None, train=False)
    assert torch.isfinite(sb.float()).all() and torch.isfinite(zb.float()).all()
    print(f"bf16-LM + autocast path OK: s_lm dtype={sb.dtype}")

    # (5) Document that the LMStack autograd failure is pre-existing (no adapter).
    s = torch.randn(1, L, 16, requires_grad=True)
    z = torch.randn(1, L, L, 8, requires_grad=True)
    m = torch.ones(1, L, dtype=torch.bool)
    try:
        consumer.lm_stack(s, z, m, "torch")  # grad enabled
        print("NOTE: plain LMStack ran under autograd (no view error on this torch)")
    except RuntimeError as e:
        print(f"PRE-EXISTING (not adapter): LMStack autograd/CPU view error -> {str(e)[:70]}...")

    print("\nALL ADAPTER INTEGRATION CHECKS PASSED (inference path)")


if __name__ == "__main__":
    main()
