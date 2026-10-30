"""TRIAL: one learned subtype's trajectory on 3D brains, drawn with yabplot (aparc cortex atlas).
Each row = one stage; each lobe is coloured by the highest cut-point it has reached by then in the model
(mean event stage over the MCMC samples, as the 'Events of each subtype' figure): none (grey), z 2, z 5, z 10.
Only the cortex is drawn: hippocampus and amygdala (part of MTL) are inside the brain, not on its surface.
Run from the project folder:   python code/09a_try_brain_trajectory.py [subtype number, default 1]
Output: results/trial/brain_trajectory_subtype<n>.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yabplot as yab
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figures_apply import score_cmap
from src.trained_model import event_stages, load_model_package

STAGES = [5, 10, 15, 20, 25, 30]
VIEWS = ["left_lateral", "left_medial", "right_medial", "right_lateral"]

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["model"]
package = load_model_package(cfg, ROOT)
subtype = package["subtype_names"][int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0]
events = event_stages(package)[subtype]                 # lobes x cut-points: mean stage of each event
cortex = set(yab.get_atlas_regions(atlas="aparc", category="cortical"))


def level_at(stage):
    """{aparc region: highest cut-point its lobe has reached by this stage}; regions not reached are left out
    (yabplot draws them grey)."""
    values = {}
    for lobe, row in events.iterrows():
        reached = [z for z, s in row.dropna().items() if s <= stage]
        if not reached:
            continue
        for region in package["regions"][lobe]:          # ctx-lh-entorhinal -> L_entorhinal
            name = region.replace("ctx-lh-", "L_").replace("ctx-rh-", "R_")
            if name in cortex:
                values[name] = max(reached)
    return values


start = time.time()
fig, axes = plt.subplots(len(STAGES), len(VIEWS), figsize=(3 * len(VIEWS), 2.3 * len(STAGES)))
for row, stage in enumerate(STAGES):
    data = level_at(stage)
    for col, view in enumerate(VIEWS):
        ax = axes[row, col]
        before = set(fig.axes)
        yab.plot_cortical(data=data, atlas="aparc", ax=ax, views=[view], cmap=score_cmap((0, 10)),
                          vminmax=[0, 10], nan_color="#E4E7EB", style="default")
        for extra in set(fig.axes) - before:             # yabplot adds a colour bar to every panel: one is enough
            extra.remove()
        if col == 0:
            ax.text(-0.05, 0.5, f"stage {stage}", transform=ax.transAxes, ha="right", va="center", fontsize=11)
        if row == 0:
            ax.set_title(view.replace("_", " "), fontsize=10)
bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 10), cmap=score_cmap((0, 10))), ax=axes,
                   fraction=0.02, pad=0.02, ticks=[0, 2, 5, 10])
bar.set_label("highest cut-point reached (grey = none)")
fig.suptitle(f"{subtype}: trajectory on the brain (trial)", fontsize=13, fontweight="bold")
out = ROOT / "results" / "trial" / f"brain_trajectory_subtype{package['subtype_names'].index(subtype) + 1}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"{len(STAGES) * len(VIEWS)} brains drawn in {time.time() - start:.0f} s -> {out.relative_to(ROOT)}")
