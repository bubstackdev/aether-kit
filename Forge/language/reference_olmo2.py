#!/usr/bin/env python3
"""Independent Transformers OLMo2 reference and port parity checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dense import main
if __name__ == "__main__":
    main(default_architecture="olmo2")
