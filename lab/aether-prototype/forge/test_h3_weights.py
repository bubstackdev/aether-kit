"""CPU regression for the original converter's raw QKV input contract."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

from h3_weights import convert_raw_transformer_key

lab = Path(os.environ["AETHER_H3_LAB"])
sys.dont_write_bytecode = True
sys.path[:0] = [str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
import torch

path = lab / "forge/vendor/diffusers/scripts/convert_minimax_h3_to_diffusers.py"
spec = importlib.util.spec_from_file_location("upstream_h3_conversion", path)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class RawQkvContract(unittest.TestCase):
    def test_head_and_projection_sentinels_survive_raw_checkpoint_conversion(self):
        # Two heads, two rows per projection. Rows carry unique values, making
        # the raw Q0,K0,V0,Q1,K1,V1 layout unambiguous independently of weights.
        raw = torch.arange(36, dtype=torch.float32).reshape(12, 3)
        config = dict(num_attention_heads=2, attention_head_dim=2)
        expected_rows = {"to_q": [0, 1, 6, 7], "to_k": [2, 3, 8, 9], "to_v": [4, 5, 10, 11]}
        for prefix, target_prefix in [("blocks.0", "transformer_blocks.0"),
                                      ("token_refiner.blocks.1", "token_refiner.refiner_blocks.1")]:
            name = prefix + ".attn.qkv_proj.weight"
            actual = dict(convert_raw_transformer_key(converter, name, raw, config))
            wrong = dict(converter.convert_transformer_key(name, raw, config))
            for projection, rows in expected_rows.items():
                key = target_prefix + ".attn." + projection + ".weight"
                self.assertTrue(torch.equal(actual[key], raw[rows]))
                self.assertFalse(torch.equal(wrong[key], raw[rows]))


if __name__ == "__main__":
    unittest.main()
