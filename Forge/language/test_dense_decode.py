"""Weight-free differential tests against pinned Transformers; run with Forge's Python."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from olmo2_decode import Olmo2Decoder, load_weights as load_olmo2
from qwen3_decode import Qwen3Decoder, load_weights as load_qwen3


class DenseDecoderTests(unittest.TestCase):
    def test_upstream_parity_and_cache_growth(self):
        torch.set_num_threads(2)
        for config_type, upstream_type, decoder_type, loader in (
            (Olmo2Config, Olmo2ForCausalLM, Olmo2Decoder, load_olmo2),
            (Qwen3Config, Qwen3ForCausalLM, Qwen3Decoder, load_qwen3),
        ):
            for kv_heads in (2, 4):
                torch.manual_seed(73)
                cfg = config_type(vocab_size=128, hidden_size=64, intermediate_size=96,
                                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
                                  head_dim=16, rms_norm_eps=1e-6, rope_theta=500000., tie_word_embeddings=False)
                upstream = upstream_type(cfg).eval()
                # Nonuniform gains detect normalization over the wrong axis / misplaced residual norms.
                with torch.no_grad():
                    for name, param in upstream.named_parameters():
                        if 'norm' in name:
                            param.uniform_(0.6, 1.4)
                ids = torch.tensor([[7, 10, 9, 3, 2, 11, 5, 8, 1, 6, 4, 12, 13]])
                with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                    upstream.save_pretrained(directory)
                    expected = upstream(ids, use_cache=False).logits
                    for contract in ('running', 'bucketed'):
                        with self.subTest(family=cfg.model_type, kv_heads=kv_heads, contract=contract):
                            port = decoder_type(cfg.to_dict(), contract).eval()
                            loader(port, Path(directory))
                            kc, vc = port.empty_state(8)
                            start = 0
                            for width in (3, 4, 1, 2, 3):
                                end = start + width
                                if end > kc.shape[3]:
                                    nk, nv = port.empty_state(24)
                                    nk[:, :, :, :start] = kc[:, :, :, :start]
                                    nv[:, :, :, :start] = vc[:, :, :, :start]
                                    kc, vc = nk, nv
                                # Invisible future cache rows must never contribute to attention.
                                kc[:, :, :, start:] = 23
                                vc[:, :, :, start:] = -17
                                positions = torch.arange(start if contract == 'bucketed' else 0, end)
                                got = port.step(ids[:, start:end].int(), positions[None].int(), kc, vc)
                                torch.testing.assert_close(got, expected[0, end - 1], atol=3e-6, rtol=3e-5)
                                start = end

    def test_unsupported_olmo2_variants_refuse(self):
        cfg = Olmo2Config(hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                          num_attention_heads=4, num_key_value_heads=2).to_dict()
        for key, value in [('attention_bias', True), ('rope_scaling', {'type': 'linear', 'factor': 2}),
                           ('hidden_act', 'gelu'), ('model_type', 'olmo3')]:
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                Olmo2Decoder({**cfg, key: value}, 'bucketed')


if __name__ == '__main__':
    unittest.main()
