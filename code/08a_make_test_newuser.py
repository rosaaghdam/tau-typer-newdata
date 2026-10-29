"""Make data/simulated/test_newuser.csv: the simulated example data as a NEW USER might send it, with column
names that differ from the model's 64 region names - to try step 1 (Region matching) in the app by hand.
The values are the same as df_sustain_true.csv, so the results must be the same once every region is found.
Run from the project folder:   python code/08a_make_test_newuser.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "simulated"
data = pd.read_csv(SIM / "df_sustain_true.csv")

renames = {
    # found automatically ("renamed"): other spelling of the same side + region
    "ctx-lh-entorhinal": "CTX_LH_ENTORHINAL",
    "ctx-rh-entorhinal": "ctx_rh_entorhinal",
    "ctx-lh-fusiform": "fusiform_L",
    "ctx-rh-fusiform": "fusiform_R",
    "ctx-lh-precuneus": "lh_precuneus_SUVR",
    "Right-Amygdala": "right_amygdala",
    "Right-Hippocampus": "Hippocampus (right)",
    # NOT found ("missing"): a short or different name - choose the column yourself in the app
    "Left-Hippocampus": "Left-Hippo",
    "ctx-lh-insula": "L_Insular_cortex",
    # clinical columns under other names
    "filename": "Subject", "study": "Site", "amyloid": "Amyloid_status", "diagnosis": "Dx",
}
new = data.rename(columns=renames)
new["rh_precuneus_SUVR"] = new["ctx-rh-precuneus"]       # two columns for one region -> "more than one"
new["ctx-lh-entorhinal_SIZE"] = 1500.0                     # a volume: must NOT be taken as entorhinal tau
new = new.drop(columns=["is_synthetic"])
new.to_csv(SIM / "test_newuser.csv", index=False)
print(f"wrote data/simulated/test_newuser.csv: {new.shape[0]} rows, {new.shape[1]} columns")
