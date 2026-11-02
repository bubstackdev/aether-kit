#!/usr/bin/env python3
"""Compatibility entry point for shared dense-language bundle registration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_dense_bundle import *
if __name__ == "__main__":
    main()
