"""Tab 1, piece 1 - does src/roi_names.match_regions find the model's 66 regions in differently named files?
Run from the project folder:   python code/06a_check_region_matching.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.roi_names import match_regions

expected = [str(r) for r in np.load(ROOT / "data" / "simulated" / "mod_cols.npy")]   # the model's 66 regions
columns = list(pd.read_csv(ROOT / "data" / "simulated" / "df_sustain_true.csv", nrows=0).columns)

def side_last(name):                                       # ctx-lh-entorhinal -> entorhinal_L
    side = "L" if "lh" in name or "Left" in name else "R"
    return name.replace("ctx-lh-", "").replace("ctx-rh-", "").replace("Left-", "").replace("Right-", "") + "_" + side

tests = {"as uploaded": columns,
         "CAPITALS_WITH_UNDERSCORES": [c.upper().replace("-", "_") for c in columns],
         "region_L style": [side_last(c) if c in expected else c for c in columns],
         "one missing, one doubled": [c for c in columns if c != "ctx-lh-cuneus"] + ["ctx_rh_precuneus",
                                                                                    "CTX_LH_ENTORHINAL_SIZE"]}
for name, cols in tests.items():
    table = match_regions(cols, expected)
    print(f"{name:28s}", table.status.value_counts().to_dict())
    print("   not found:", table.region[~table.status.isin(["same name", "renamed"])].tolist())
