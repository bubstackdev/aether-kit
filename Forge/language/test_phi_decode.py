"""Differential tests against pinned Transformers, without downloaded weights."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import PhiConfig, PhiForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phi_decode import PhiDecoder, load_weights, rotate_prefix


class PhiTests(unittest.TestCase):
    def test_upstream_parity(self):
        torch.set_num_threads(2)
        for factor, heads, tied, qknorm in ((0.4, 4, False, False), (0.5, 2, True, True), (1.0, 4, True, False)):
            torch.manual_seed(73)
            cfg = PhiConfig(vocab_size=128, hidden_size=160, intermediate_size=192,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=heads,
                            partial_rotary_factor=factor, tie_word_embeddings=tied, qk_layernorm=qknorm,
                            max_position_embeddings=64, resid_pdrop=0, embd_pdrop=0, attention_dropout=0)
            upstream = PhiForCausalLM(cfg).eval()
            with torch.no_grad():
                for name, p in upstream.named_parameters():
                    if 'layernorm' in name and name.endswith('weight'):
                        p.uniform_(0.6, 1.4)
                    if name.endswith('bias'):
                        p.uniform_(-0.1, 0.1)
            ids = torch.tensor([[7, 10, 9, 3, 2, 11, 5, 8, 1, 6, 4, 12, 13]])
            with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                upstream.save_pretrained(directory)
                expected = upstream(ids, use_cache=False).logits
                for contract in ('running', 'bucketed'):
                    with self.subTest(factor=factor, heads=heads, tied=tied, qknorm=qknorm, contract=contract):
                        port = PhiDecoder(cfg.to_dict(), contract, max_capacity=64).eval()
                        load_weights(port, Path(directory))
                        kc, vc = port.empty_state(8)
                        start = 0
                        for width in (3, 4, 1, 2, 3):
                            end = start + width
                            if end > kc.shape[3]:
                                nk, nv = port.empty_state(24)
                                nk[:, :, :, :start] = kc[:, :, :, :start]
                                nv[:, :, :, :start] = vc[:, :, :, :start]
                                kc, vc = nk, nv
                            kc[:, :, :, start:] = 23
                            vc[:, :, :, start:] = -17
                            pos = torch.arange(start if contract == 'bucketed' else 0, end)[None].int()
                            got = port.step(ids[:, start:end].int(), pos, kc, vc)
                            torch.testing.assert_close(got, expected[0, end - 1], atol=3e-6, rtol=3e-5)
                            start = end

    def test_trained_context_boundary(self):
        torch.manual_seed(91)
        cfg = PhiConfig(vocab_size=128, hidden_size=80, intermediate_size=96,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        partial_rotary_factor=0.4, max_position_embeddings=2048)
        upstream = PhiForCausalLM(cfg).eval()
        ids = torch.arange(2048)[None] % 128
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            upstream.save_pretrained(directory)
            expected = upstream(ids, use_cache=False).logits[0, -1]
            for contract in ('running', 'bucketed'):
                with self.subTest(contract=contract):
                    port = PhiDecoder(cfg.to_dict(), contract).eval()
                    load_weights(port, Path(directory))
                    kc, vc = port.empty_state(2048)
                    for start in range(0, 2048, 256):
                        end = start + 256
                        pos = torch.arange(start if contract == 'bucketed' else 0, end)[None].int()
                        got = port.step(ids[:, start:end].int(), pos, kc, vc)
                    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)

    def test_partial_rotation_preserves_tail(self):
        x = torch.randn(1, 5, 4, 80)
        cos, sin = torch.randn(5, 32), torch.randn(5, 32)
        result = rotate_prefix(x, cos, sin)
        self.assertTrue(torch.equal(result[..., 32:], x[..., 32:]))

    def test_refuses_unimplemented_or_invalid_config(self):
        cfg = PhiConfig(hidden_size=160, num_attention_heads=4, num_hidden_layers=1).to_dict()
        for key, value in [('model_type', 'phi3'), ('hidden_act', 'relu'),
                           ('rope_scaling', {'type': 'linear', 'factor': 2}), ('attention_bias', True)]:
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                PhiDecoder({**cfg, key: value}, 'bucketed')
        for key, value in [('partial_rotary_factor', 0), ('partial_rotary_factor', 1.5),
                           ('partial_rotary_factor', 0.125), ('num_key_value_heads', 3)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                PhiDecoder({**cfg, key: value}, 'bucketed')
        with self.assertRaises(NotImplementedError):
            PhiDecoder(cfg, 'bucketed', attention_precision='float16')
        with self.assertRaises(ValueError):
            PhiDecoder(cfg, 'bucketed', max_capacity=4096)


if __name__ == '__main__':
    unittest.main()
