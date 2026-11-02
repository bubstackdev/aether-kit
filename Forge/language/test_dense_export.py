"""Core AI CPU regression: strict logits/cache parity for random-weight exports.

The Qwen3 seed-0 first row previously reversed tokens 211/398 in fp16 SDPA.
Float32 attention must keep the expected greedy choice; no near-tie waiver.
Run using Forge's pinned Python environment, without weights or a GPU lease.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DenseExportTests(unittest.TestCase):
    def test_strict_coreai_parity_with_fp32_attention(self):
        for family in ('qwen3', 'olmo2', 'gpt2', 'phi'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'bundle'
                result = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name(f'export_{family}.py')),
                    '--tiny', '--output', str(output), '--max-capacity', '512', '--max-prefill-chunk', '64'],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / 'export-report.json').read_text())
                self.assertEqual(report['attentionPrecision'], 'float32')
                self.assertEqual(report['dtype'], 'float16')
                self.assertTrue(report['selfCheck']['passed'])
                if family == 'qwen3':
                    self.assertEqual(report['selfCheck']['steps'][0]['referenceArgmax'], 211)
                    self.assertEqual(report['selfCheck']['steps'][0]['nativeArgmax'], 211)
                for step in report['selfCheck']['steps']:
                    self.assertTrue(step['argmaxEqual'])
                    self.assertEqual(step['lostCacheRows'], 0)


if __name__ == '__main__':
    unittest.main()
