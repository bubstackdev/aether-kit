#!/usr/bin/env python3
"""Export OLMo2 using Aether's shared dense-language step contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dense import main
if __name__ == "__main__":
    main(default_architecture="olmo2")
