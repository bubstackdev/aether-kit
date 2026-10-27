"""Tokenize the fixed timing prompt with the Granite tokenizer (CPU)."""
import sys
from pathlib import Path

from transformers import AutoTokenizer

ckpt = next((Path.home() / "Library/Caches/aether/checkpoints/models--ibm-granite--granite-4.0-h-350m/snapshots")
            .iterdir())
text = ("The Roman Empire was the post-Republican state of ancient Rome. It is generally understood to mean the "
        "period and territory ruled by the Romans following Octavian's assumption of sole rule under the Principate "
        "in 27 BC. The empire")
ids = AutoTokenizer.from_pretrained(ckpt)(text)["input_ids"]
Path(sys.argv[1]).write_text(",".join(map(str, ids)))
print(len(ids))
