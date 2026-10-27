"""S9 - full SuStaIn run with the paper's settings from config.yaml (25 start points, 10,000 MCMC).

Run in the Terminal (NOT inside Python), from the project folder with the environment active:
    python code/02_run_sustain_full.py
Takes roughly 30-60 minutes. Results go to results/toy/sustain_full/.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # save figures to files instead of opening windows
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent      # project root (one level above code/)
sys.path.insert(0, str(ROOT))
from src.sustain_model import sustain_inputs, run_sustain

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
p, s = cfg["pipeline"], cfg["sustain"]
z = pd.read_csv(ROOT / p["output_folder"] / "zscores.csv", index_col=p["id_column"])
data, Z_vals, Z_max = sustain_inputs(z, p["regions"], s["z_events"], s["z_max"])

start = time.time()
run_sustain(data, Z_vals, Z_max, p["regions"], s, str(ROOT / p["output_folder"] / "sustain_full"), p["seed"])
print(f"DONE in {(time.time() - start) / 60:.0f} minutes - results in {p['output_folder']}/sustain_full")
