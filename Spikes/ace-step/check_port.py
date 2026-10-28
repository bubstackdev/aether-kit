"""CPU check, no weights: acestep_port's prompts, metas and schedule against the pinned upstream sources.

usage: check_port.py <ACE-Step-1.5 source dir @ ca1e85fe9430> <Ace-Step1.5 snapshot's acestep-v15-turbo dir>
Imports upstream's PromptMixin and MetadataMixin (they need only loguru and torch) and reads generate_audio's
SHIFT_TIMESTEPS table from the pinned model file, then compares them with the port.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import acestep_port as port  # noqa: E402

source, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(source))
from acestep import constants  # noqa: E402


def load(name):
    """One handler module by file: the handler package's __init__ imports every mixin (soundfile, MLX, the pipeline)."""
    import importlib.util
    path = source / "acestep/core/generation/handler" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"handler_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MetadataMixin = load("metadata_utils").MetadataMixin
PromptMixin = load("prompt_utils").PromptMixin


class Handler(PromptMixin, MetadataMixin):
    pass


h = Handler()
assert port.SFT_GEN_PROMPT == constants.SFT_GEN_PROMPT
assert port.DEFAULT_DIT_INSTRUCTION == constants.DEFAULT_DIT_INSTRUCTION
for duration, bpm in [(60, 82), (30, "N/A"), (240, 120)]:
    upstream_metas = h._dict_to_meta_string({"bpm": bpm, "duration": duration})
    assert port.metas(duration, bpm=bpm) == upstream_metas, (port.metas(duration, bpm=bpm), upstream_metas)
    caption, lyrics = h.build_dit_inputs("text2music", None, "warm lo-fi beat", "[Instrumental]",
                                         {"bpm": bpm, "duration": duration}, "unknown")
    assert port.caption_prompt("warm lo-fi beat", port.metas(duration, bpm=bpm)) == caption
    assert port.lyric_prompt("[Instrumental]", "unknown") == lyrics

tree = ast.parse((model_dir / "modeling_acestep_v15_turbo.py").read_text())
table = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(getattr(t, "id", None) == "SHIFT_TIMESTEPS" for t in node.targets))
assert ast.literal_eval(table) == port.SHIFT_TIMESTEPS
print("acestep_port matches upstream: prompt template, instruction, metas, caption and lyric prompts, turbo schedules")
