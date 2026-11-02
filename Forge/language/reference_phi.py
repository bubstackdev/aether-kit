#!/usr/bin/env python3
"""Phi entry point; CLI and cache contracts live in reference_dense."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dense import main
if __name__ == '__main__':
    main(default_architecture='phi')
