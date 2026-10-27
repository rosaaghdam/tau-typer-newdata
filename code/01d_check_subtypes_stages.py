"""Step 1d – check assign_true_subtypes_and_stages(): the answer key (true subtype + stage).

Run:  bash run.sh 01d
Expected (seed 42): -1 (tau-negative): 300, S1: 69, S2: 36, S3: 62, S4: 33;
          tau+ stages from 1 to 30; every tau-negative person at stage 0.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.toy_data import assign_true_subtypes_and_stages

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
subtypes, stages = assign_true_subtypes_and_stages(cfg, np.random.default_rng(cfg["seed"]))
values, counts = np.unique(subtypes, return_counts=True)
print("people per subtype (-1 = tau-negative):", dict(zip(values.tolist(), counts.tolist())))
print("tau+ stages from", stages[subtypes >= 0].min(), "to", stages[subtypes >= 0].max())
print("all tau-negative at stage 0:", bool(np.all(stages[subtypes == -1] == 0)))
