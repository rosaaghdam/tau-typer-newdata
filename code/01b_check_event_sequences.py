"""Step 1b – check the "wave" event sequences built from config.yaml.

Run from the project root (with the environment active):
    python code/01b_check_event_sequences.py
Expected: every subtype has 30 events, each exactly once ("True").
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # project root (one level above code/)
sys.path.insert(0, str(ROOT))                       # so Python can find src/
from src.toy_data import wave_event_sequence

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
regions, levels = cfg["regions"], cfg["z_events"]

for name, order in cfg["subtype_region_order"].items():
    seq = wave_event_sequence(order, regions, len(levels))
    each_once = sorted(seq) == list(range(len(regions) * len(levels)))
    first_five = [f"{regions[e % len(regions)]}->{levels[e // len(regions)]}" for e in seq[:5]]
    print(f"{name}: {len(seq)} events, each exactly once: {each_once}")
    print(f"   starts with: {first_five}")
