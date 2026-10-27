"""Step 1c – check sustain_generator_inputs(): the model inputs for pySuStaIn's generator.

Run:  bash run.sh 01c
Expected: Z_vals (10, 3) with rows [2 5 10]; Z_max all 15; sequences (4, 30),
          S1 starting 0 10 1 20 11 2.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # project root (one level above code/)
sys.path.insert(0, str(ROOT))                       # so Python can find src/
from src.toy_data import sustain_generator_inputs

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
Z_vals, Z_max, sequences = sustain_generator_inputs(cfg)
print("Z_vals shape:", Z_vals.shape, "| first row:", Z_vals[0])
print("Z_max:", Z_max)
print("sequences shape:", sequences.shape, "| S1 starts:", sequences[0, :6])
