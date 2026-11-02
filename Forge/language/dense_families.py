"""Explicit exporter/reference registry. Add a family here without editing either CLI."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class DenseFamily:
    module: str
    decoder: str
    tiny_overrides: dict
    maximum_context: int = 16384
    prompt_style: str = 'chat'
    reference_rtol: float = 0.0

    def implementation(self):
        module = import_module(self.module)
        return getattr(module, self.decoder), module.load_weights


FAMILIES = {
    'qwen3': DenseFamily('qwen3_decode', 'Qwen3Decoder', {}),
    'olmo2': DenseFamily('olmo2_decode', 'Olmo2Decoder',
                         {'model_type': 'olmo2', 'rope_theta': 500000.0, 'tie_word_embeddings': False}),
    'gpt2': DenseFamily('gpt2_decode', 'GPT2Decoder',
                        {'model_type': 'gpt2', 'n_embd': 256, 'n_head': 4, 'n_layer': 2, 'n_positions': 1024,
                         'layer_norm_epsilon': 1e-5, 'activation_function': 'gelu_new'},
                        maximum_context=1024, prompt_style='completion', reference_rtol=1e-5),
}
