"""New app's pipeline (src/trained_model.py + src/normalization.py) vs Jake's notebook steps on the simulated
data: lobes, z-scores, subtypes, stages and probabilities must equal the answer key (code/05b, 05c).
Run from the project folder:   python code/07a_check_pipeline_vs_jake.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.normalization import normalize
from src.roi_names import match_regions
from src.trained_model import (apply_model, clean_up, load_model_package, lobe_values, required_regions,
                               tau_positive)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["model"]
package = load_model_package(cfg, ROOT)
SIM = ROOT / "data" / "simulated"
new = pd.read_csv(SIM / "df_sustain_true.csv", index_col="filename")
needed = required_regions(package, cfg["cleanup"]["regions"])
matched = match_regions(list(new.columns), needed)
print("regions found:", matched.status.value_counts().to_dict())

lobes = lobe_values(new, package["regions"])
z, normals, _, notes = normalize("training", lobes, package, cfg["gmm"])
jake_z = pd.read_csv(SIM / "jake_steps_lobar_z.csv", index_col="filename")[package["lobes"]]
print("largest z difference from Jake's steps:", float(np.abs(z.values - jake_z.loc[z.index].values).max()))

table, _ = apply_model(z, package, cfg["n_samples"])
key = pd.read_csv(SIM / "expected_jake_steps.csv", index_col="filename").loc[table.index]
print("same subtype as Jake:", bool((table.subtype_number == key.subtype).all()),
      "| same stage:", bool((table.ml_stage == key.stage).all()),
      "| largest probability difference:", float(np.abs(table[[f"p_S{n}" for n in range(1, 5)]].values
                                                        - key[[f"p_S{n}" for n in range(1, 5)]].values).max()))
print("before cleanup:", table.subtype_number.value_counts().sort_index().to_dict())

p, tau_pos, notes = tau_positive(new, package["reference"], cfg["cleanup"]["regions"], cfg["cleanup"]["threshold"],
                                 cfg["gmm"])
cleaned = clean_up(table, tau_pos)
print("after cleanup: ", cleaned.subtype_number.value_counts().sort_index().to_dict(),
      "| people changed:", int(cleaned.cleaned.sum()), "| notes:", notes)
