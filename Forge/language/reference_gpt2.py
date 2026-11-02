#!/usr/bin/env python3
"""Independent GPT-2 completion reference through the shared dense tooling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dense import main
if __name__ == '__main__':
    main(default_architecture='gpt2')
