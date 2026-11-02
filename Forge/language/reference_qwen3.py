#!/usr/bin/env python3
"""Compatibility entry point for the Qwen3 independent reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dense import *
if __name__ == "__main__":
    main()
