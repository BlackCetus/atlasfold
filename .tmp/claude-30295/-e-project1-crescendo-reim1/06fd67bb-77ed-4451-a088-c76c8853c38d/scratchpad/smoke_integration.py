"""Integration diagnostic (CPU, no model download — tokenizer is already cached)."""
import sys
import numpy as np
sys.path.insert(0, "src")

from atlaslm.alphabet import Alphabet, VOCAB
from atlasfold.common import featurize as F
from atlasfold.common.lm import get_lm_alphabet
from atlasfold.train.monomer.dataset import TrainingDataset

VOCAB_TO_IDX = {t: i for i, t in enumerate(VOCAB)}
SEQ = "MKLVQ"

# --- 1) ESM-3 regression: new featurize == old hardcoded layout ---
esm = Alphabet()
feat = F.featurize(SEQ)  # default alphabet
L = len(SEQ)
exp_ids = np.array([VOCAB_TO_IDX["<cls>"]] + [VOCAB_TO_IDX[a] for a in SEQ] + [VOCAB_TO_IDX["<eos>"]])
exp_pos = np.concatenate(([0], np.arange(1, L + 1), [L + 1]))
exp_tok = np.arange(1, L + 1)
assert np.array_equal(feat["lm.input_ids"], exp_ids), feat["lm.input_ids"]
assert np.array_equal(feat["lm.pos_id"], exp_pos), feat["lm.pos_id"]
assert np.array_equal(feat["seq_tok_idx"], exp_tok), feat["seq_tok_idx"]
assert np.array_equal(feat["lm.seq_id"], np.ones(L + 2)), feat["lm.seq_id"]
print("ESM-3 regression OK:", feat["lm.input_ids"].tolist())

# --- 2) ProtT5 featurization (cached tokenizer, no weights) ---
p5 = get_lm_alphabet("prott5-xl")
assert type(p5).__name__ == "ProtT5Alphabet"
featp = F.featurize(SEQ, alphabet=p5)
# residues + </s>, no BOS
exp_ids_p = np.array([p5.res_to_idx[a] for a in SEQ] + [p5.eos_idx])
exp_pos_p = np.concatenate((np.arange(1, L + 1), [L + 1]))
exp_tok_p = np.arange(0, L)
assert np.array_equal(featp["lm.input_ids"], exp_ids_p), featp["lm.input_ids"]
assert np.array_equal(featp["lm.pos_id"], exp_pos_p), featp["lm.pos_id"]
assert np.array_equal(featp["seq_tok_idx"], exp_tok_p), featp["seq_tok_idx"]
assert featp["lm.input_ids"].shape[0] == L + 1
print("ProtT5 featurize OK:", featp["lm.input_ids"].tolist(),
      "| n_prefix/suffix:", p5.n_prefix, p5.n_suffix, "| mask_idx:", p5.mask_idx)

# --- 3) Full crop -> seq_tok_idx pathway for ProtT5 (n_prefix=0, n_suffix=1) ---
seqlen = 100
long_seq = ("ACDEFGHIKLMNPQRSTVWY" * 5)  # length 100, standard residues
featL = F.featurize(long_seq, alphabet=p5)          # lm arrays length 101
res_idx = np.arange(1, seqlen + 1)                   # fold res_idx
crop = np.arange(10, 31)                             # contiguous crop (21 residues)
lm_crop = TrainingDataset._expand_crop_indices_for_lm(
    crop, seqlen, max_seq_length=50, n_prefix=p5.n_prefix, n_suffix=p5.n_suffix
)
assert len(lm_crop) <= 50
assert lm_crop.min() >= 0 and lm_crop.max() <= seqlen + p5.n_prefix + p5.n_suffix - 1
lm_pos_cropped = featL["lm.pos_id"][lm_crop]
is_in_crop = np.isin(lm_pos_cropped, res_idx[crop])
seq_tok_idx = np.where(is_in_crop)[0]
# residue tokens recovered must equal the cropped residues, in order
recovered_ids = featL["lm.input_ids"][lm_crop][seq_tok_idx]
expected_ids = np.array([p5.res_to_idx[long_seq[i]] for i in crop])
assert np.array_equal(recovered_ids, expected_ids), (recovered_ids, expected_ids)
assert len(seq_tok_idx) == len(crop)
print(f"ProtT5 crop pathway OK: lm_crop_len={len(lm_crop)} residues_recovered={len(seq_tok_idx)}")

# --- 4) ESM-3 crop expansion unchanged (default n_prefix/suffix = 1,1) ---
lm_crop_esm = TrainingDataset._expand_crop_indices_for_lm(crop, seqlen, max_seq_length=50)
assert lm_crop_esm.max() <= seqlen + 1  # +2 tokens total, indices 0..101
print(f"ESM-3 crop expansion OK: len={len(lm_crop_esm)} max_idx={lm_crop_esm.max()}")

print("\nALL INTEGRATION CHECKS PASSED")
