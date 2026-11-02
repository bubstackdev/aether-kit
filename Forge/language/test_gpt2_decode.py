"""Differential GPT-2 tests against pinned Transformers, with no downloaded weights."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_decode import GPT2Decoder, load_weights


class GPT2Tests(unittest.TestCase):
    def test_upstream_cache_contracts_and_position_limit(self):
        torch.set_num_threads(2)
        for tied in (True, False):
            torch.manual_seed(73)
            cfg = GPT2Config(vocab_size=128, n_embd=64, n_head=4, n_layer=2, n_positions=32,
                             n_inner=96, tie_word_embeddings=tied)
            upstream = GPT2LMHeadModel(cfg).eval()
            # Nonzero biases, nonuniform norms and absolute positions detect silent mapping errors.
            with torch.no_grad():
                for name, param in upstream.named_parameters():
                    if 'ln_' in name and 'weight' in name:
                        param.uniform_(0.6, 1.4)
                    elif name.endswith('bias'):
                        param.uniform_(-0.1, 0.1)
            ids = torch.randint(0, 128, (1, 32))
            with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                upstream.save_pretrained(directory)
                expected = upstream(ids, use_cache=False).logits
                for contract in ('running', 'bucketed'):
                    with self.subTest(tied=tied, contract=contract):
                        port = GPT2Decoder(cfg.to_dict(), contract, max_capacity=32).eval()
                        load_weights(port, Path(directory))
                        kc, vc = port.empty_state(8)
                        start = 0
                        for width in (3, 4, 1, 2, 21, 1):
                            end = start + width
                            if end > kc.shape[3]:
                                nk, nv = port.empty_state(32)
                                nk[:, :, :, :start] = kc[:, :, :, :start]
                                nv[:, :, :, :start] = vc[:, :, :, :start]
                                kc, vc = nk, nv
                            kc[:, :, :, start:] = 23
                            vc[:, :, :, start:] = -17
                            positions = torch.arange(start if contract == 'bucketed' else 0, end)[None].int()
                            got = port.step(ids[:, start:end].int(), positions, kc, vc)
                            torch.testing.assert_close(got, expected[0, end-1], atol=3e-6, rtol=3e-5)
                            start = end

    def test_legacy_checkpoint_and_strict_tensor_loading(self):
        from safetensors.torch import save_file
        cfg = GPT2Config(vocab_size=128, n_embd=64, n_head=4, n_layer=1, n_positions=32)
        upstream = GPT2LMHeadModel(cfg).eval()
        state = {name.removeprefix('transformer.'): tensor.detach().contiguous()
                 for name, tensor in upstream.state_dict().items() if name != 'lm_head.weight'}
        state['h.0.attn.bias'] = torch.ones(1, 1, 32, 32, dtype=torch.bool).tril()
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            file = Path(directory) / 'model.safetensors'
            save_file(state, file)
            port = GPT2Decoder(cfg.to_dict(), 'bucketed', max_capacity=32).eval()
            load_weights(port, Path(directory))
            ids = torch.tensor([[3, 7, 11]], dtype=torch.int32)
            got = port.step(ids, torch.arange(3)[None].int(), *port.empty_state(8))
            torch.testing.assert_close(got, upstream(ids).logits[0, -1], atol=3e-6, rtol=3e-5)
            state['unexpected.weight'] = torch.ones(2)
            save_file(state, file)
            with self.assertRaises(KeyError):
                load_weights(port, Path(directory))
            del state['unexpected.weight'], state['wpe.weight']
            save_file(state, file)
            with self.assertRaises(KeyError):
                load_weights(port, Path(directory))

    def test_unsupported_variants_refuse(self):
        cfg = GPT2Config(n_embd=64, n_head=4, n_layer=2, n_positions=32).to_dict()
        for key, value in [('model_type', 'gpt_bigcode'), ('activation_function', 'relu'),
                           ('scale_attn_weights', False), ('add_cross_attention', True),
                           ('scale_attn_by_inverse_layer_idx', True), ('reorder_and_upcast_attn', True),
                           ('pruned_heads', {0: [1]})]:
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                GPT2Decoder({**cfg, key: value}, 'bucketed', max_capacity=32)
        with self.assertRaises(ValueError):
            GPT2Decoder(cfg, 'bucketed', max_capacity=33)


if __name__ == '__main__':
    unittest.main()
