"""Draws every learned subtype's 'Trajectory on the brain' (src/brain3d.py, yabplot) into the cache folder of
config.yaml (app: brain_cache), so the app shows them at once. Run it after a new model is set in config.yaml,
e.g. before putting the app online. Already drawn figures are kept.
Run from the project folder:   python code/09b_draw_brain_trajectories.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.brain3d import trajectory_png
from src.trained_model import event_stages, load_model_package

config = yaml.safe_load(open(ROOT / "config.yaml"))
package = load_model_package(config["model"], ROOT)
events = event_stages(package)
for subtype in package["subtype_names"]:
    start = time.time()
    path = trajectory_png(package, events, subtype, config["app"]["trajectory_stages"], ROOT / config["app"]["brain_cache"])
    print(f"{subtype}: {path.relative_to(ROOT)} ({time.time() - start:.0f} s)")
