#!/usr/bin/env python3
"""Export GPT-2 through the shared dense-language exporter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dense import main
if __name__ == '__main__':
    main(default_architecture='gpt2')
