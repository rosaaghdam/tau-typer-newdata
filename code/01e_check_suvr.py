"""Step 1e – check simulate_regional_suvr(): noisy z-scores and SUVR per region.

Run:  bash run.sh 01e
Expected: shapes (500, 10); tau-negative z mean ~0 and SD ~1; tau-negative mean SUVR equal to
          the config's normal means; at late stages each subtype shows its own pattern.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.toy_data import assign_true_subtypes_and_stages, simulate_regional_suvr

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
np.set_printoptions(precision=1, suppress=True)
subtypes, stages = assign_true_subtypes_and_stages(cfg, np.random.default_rng(cfg["seed"]))
z, suvr = simulate_regional_suvr(cfg, subtypes, stages)
neg = subtypes == -1
print("shapes:", z.shape, suvr.shape, "| regions:", cfg["regions"])
print(f"tau-negative z: mean {z[neg].mean():.2f}, SD {z[neg].std():.2f}")
print("tau-negative mean SUVR:", suvr[neg].mean(axis=0).round(2).tolist())
for s, name in enumerate(cfg["subtype_region_order"]):
    late = (subtypes == s) & (stages >= 15)
    print(f"{name:20s} mean z at stage >= 15:", z[late].mean(axis=0))
