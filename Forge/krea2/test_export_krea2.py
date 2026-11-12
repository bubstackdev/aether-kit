"""Export metadata checks without converting graphs or reading checkpoint weights.

    Forge/.venv/bin/python -I Forge/krea2/test_export_krea2.py
"""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_krea2 import sample_manifest


class PrecisionMetadataTests(unittest.TestCase):
    def test_default_manifest_is_unchanged(self):
        fixture = Path(__file__).resolve().parents[2] / 'Tests/AetherCoreTests/Fixtures/krea2/realism-masked-256.json'
        self.assertEqual(sample_manifest('realism', text_rows=256), json.loads(fixture.read_text()))

    def test_fp16_decoder_is_distinguished_without_changing_encoder(self):
        old = sample_manifest('realism')
        new = sample_manifest('realism', vae_compute='float16')
        self.assertEqual(new['precision']['components']['vaeDecoder'], 'fp16 weights, fp16 compute')
        self.assertEqual(new['precision']['components']['vaeEncoder'], 'fp16 weights, fp32 compute')
        new['precision']['components']['vaeDecoder'] = old['precision']['components']['vaeDecoder']
        self.assertEqual(new, old)


if __name__ == '__main__':
    unittest.main()
