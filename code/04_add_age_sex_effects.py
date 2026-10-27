"""Add realistic age, sex and age x sex effects to the sample data (data/toy/sim_tau_400.csv).

Why: the simulated regional values had no age or sex effect at all, so the covariate adjustment of tab 2
had nothing to remove and nothing changed. In real tau-PET, SUVR rises a little with age also in healthy
people (off-target binding), and can differ between men and women. Here, in every person (healthy or not):

    SUVR += weight[region] * (0.004 * (age - 70) + 0.03 * male + 0.004 * (age - 70) * male)

so women gain 0.004 SUVR per year, men 0.008 per year (the age x sex interaction), and men are 0.03 higher
at age 70. weight = 1.5 for MTL, 1.2 temporal, 1.0 parietal, 0.8 occipital and frontal (strongest where
off-target binding is). The true subtype and stage do not change (truth file), so the Simulation check shows
whether the adjustment helps.

Also sets "group" to amyloid_neg in the truth file for the 5 MCI people that were made amyloid-negative.
Run once from the project folder:   python code/04_add_age_sex_effects.py
It stops if the effects are already there (it checks the age slope of the healthy group).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, TRUTH = ROOT / "data/toy/sim_tau_400.csv", ROOT / "data/toy/sim_tau_400_truth.csv"
WEIGHT = {"MTL": 1.5, "temporal": 1.2, "parietal": 1.0, "occipital": 0.8, "frontal": 0.8}
AGE_SLOPE, MALE, AGE_X_MALE = 0.004, 0.03, 0.004

data = pd.read_csv(DATA)
regions = [c for c in data.columns if c[:2] in ("L_", "R_")]
healthy = data["diagnosis"].isin(["CN", "SCD"]) & (data["amyloid_positive"] == 0)
slope = smf.ols(f"{regions[0]} ~ age", data=data[healthy]).fit().params["age"]
if slope > AGE_SLOPE / 2:
    raise SystemExit(f"The age effect is already there (slope {slope:.4f} per year in {regions[0]}) - nothing done.")

age, male = data["age"] - 70, (data["sex"] == "M").astype(float)
for region in regions:
    weight = WEIGHT[region.split("_", 1)[1]]
    data[region] = (data[region] + weight * (AGE_SLOPE * age + MALE * male + AGE_X_MALE * age * male)).round(4)
data.to_csv(DATA, index=False)

truth = pd.read_csv(TRUTH)
negative = data.set_index("subject_id")["amyloid_positive"].reindex(truth["subject_id"]).to_numpy() == 0
truth.loc[negative & (truth["group"] == "amyloid_pos_tau_neg"), "group"] = "amyloid_neg"
truth.to_csv(TRUTH, index=False)
print(f"Added age, sex and age x sex effects to {len(regions)} regions of {len(data)} people.")
