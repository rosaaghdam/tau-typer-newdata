"""Step 1h - add 10 invented clinical variables to the toy data (for testing tab 5 of the app).

Reads data/toy/toy_tau.csv and data/toy/toy_truth.csv (true subtype + stage of each toy person) and
writes data/toy/toy_tau_clinical.csv: the same people and columns + 10 clinical variables.
The rules are INVENTED, only so tab 5 has realistic-looking differences to find:
cognition gets worse with stage, and each subtype has its own weak domain (limbic -> memory,
MTL-sparing -> executive, lateral temporal -> language, posterior -> visuospatial).
A few values are left empty on purpose (config: toy_data.clinical_missing_fraction), and some
cognitively normal people get the diagnosis "SCD" (subjective cognitive decline; config:
toy_data.scd_fraction_of_cn) - the paper counts SCD as cognitively unimpaired.
Some tau-negative people are changed so the paper's filtering rule must REMOVE them
(config: toy_data.n_remove_young_cn, n_remove_amyloid_negative_impaired):
CN people younger than 40, and MCI/AD people who are amyloid-negative.

Run from the project folder:   python code/01h_make_toy_clinical.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
settings = yaml.safe_load(open(ROOT / "config.yaml"))["toy_data"]
rng = np.random.default_rng(settings["clinical_seed"])

data = pd.read_csv(ROOT / "data/toy/toy_tau.csv")
truth = pd.read_csv(ROOT / "data/toy/toy_truth.csv").set_index("subject_id").loc[data["subject_id"]]
stage = truth["true_stage"].to_numpy()
subtype = truth["true_subtype"].to_numpy()
amyloid = data["amyloid_positive"].to_numpy()
n = len(data)


def noise(sd):
    return rng.normal(0, sd, n)


def weak_domain(name):
    """Extra loss (in SD) for people of this subtype, growing with stage."""
    return np.where(subtype == name, 0.03 * stage, 0)


clinical = pd.DataFrame(index=data.index)
clinical["education_years"] = np.clip(np.round(14 + noise(3)), 6, 22)
clinical["APOE4_carrier"] = (rng.random(n) < np.where(amyloid == 1, 0.6, 0.25)).astype(int)
clinical["MMSE"] = np.clip(np.round(29.5 - 0.3 * stage + 0.1 * (clinical["education_years"] - 14) + noise(1.2)), 0, 30)
clinical["CDR_SB"] = np.clip(np.round((0.18 * stage + noise(0.6)) * 2) / 2, 0, 18)
clinical["ADAS_Cog13"] = np.clip(np.round(9 + 1.1 * stage + noise(4), 1), 0, 85)
clinical["memory_composite"] = np.round(0.3 - 0.06 * stage - weak_domain("S1_limbic") + noise(0.5), 2)
clinical["executive_composite"] = np.round(0.3 - 0.05 * stage - weak_domain("S2_MTL_sparing") + noise(0.5), 2)
clinical["language_composite"] = np.round(0.3 - 0.05 * stage - weak_domain("S4_lateral_temporal") + noise(0.5), 2)
clinical["visuospatial_composite"] = np.round(0.3 - 0.05 * stage - weak_domain("S3_posterior") + noise(0.5), 2)
clinical["plasma_ptau217"] = np.round(np.where(amyloid == 1, 0.6, 0.3) * np.exp(0.06 * stage + noise(0.25)), 3)

clinical = clinical + 0.0                                       # rounding can give -0.0; make it 0.0
for column in clinical.columns.drop("education_years"):         # a few empty cells, as in real data
    clinical.loc[rng.random(n) < settings["clinical_missing_fraction"], column] = np.nan

is_cn = data["diagnosis"] == "CN"
data.loc[is_cn & (rng.random(n) < settings["scd_fraction_of_cn"]), "diagnosis"] = "SCD"

# people the filtering rule must remove: picked among tau-negative (stage 0), amyloid-negative CN
candidates = data.index[(stage == 0) & (data["diagnosis"] == "CN") & (data["amyloid_positive"] == 0)]
picked = rng.choice(candidates, settings["n_remove_young_cn"] + settings["n_remove_amyloid_negative_impaired"],
                    replace=False)
young, impaired = picked[:settings["n_remove_young_cn"]], picked[settings["n_remove_young_cn"]:]
data.loc[young, "age"] = rng.uniform(30, 39.9, len(young)).round(1)             # CN, 40 or younger
data.loc[impaired, "diagnosis"] = rng.choice(["MCI", "AD"], len(impaired))        # MCI/AD, amyloid-negative

regions = [col for col in data.columns if col[:2] in ("L_", "R_")]
out = pd.concat([data.drop(columns=regions), clinical, data[regions]], axis=1)   # regions last
out.to_csv(ROOT / "data/toy/toy_tau_clinical.csv", index=False)
print(f"Saved data/toy/toy_tau_clinical.csv: {out.shape[0]} people x {out.shape[1]} columns")
print(out.drop(columns=regions).describe().round(2).T[["count", "mean", "min", "max"]])
print(out["diagnosis"].value_counts().to_string())
