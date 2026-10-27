"""Step 1f – check simulate_covariates(): age, sex, diagnosis, amyloid.

Run:  bash run.sh 01f
Expected (seed 42): CN 350 / MCI 69 / AD 81, every MCI and AD amyloid-positive;
          about 55% female; age lower at later stages (correlation about -0.5).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.toy_data import assign_true_subtypes_and_stages, simulate_covariates

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
rng = np.random.default_rng(cfg["seed"])
subtypes, stages = assign_true_subtypes_and_stages(cfg, rng)
cov = pd.DataFrame(simulate_covariates(cfg, subtypes, stages, rng))
print(pd.crosstab(cov["diagnosis"], cov["amyloid_positive"], margins=True))
print("female fraction:", round((cov["sex"] == "F").mean(), 2))
tau_pos = subtypes >= 0
print("corr(stage, age) in tau+:", round(np.corrcoef(stages[tau_pos], cov["age"][tau_pos])[0, 1], 2))
