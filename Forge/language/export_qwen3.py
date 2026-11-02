#!/usr/bin/env python3
"""Compatibility entry point for Qwen3 exports; implementation shared with OLMo2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dense import *  # preserve helper imports used by development tooling
if __name__ == "__main__":
    main()
