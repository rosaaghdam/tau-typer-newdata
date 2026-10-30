"""A learned subtype's trajectory on 3D brains, drawn with yabplot (Desikan-Killiany 'aparc' cortex atlas).

Each row = one stage; each lobe's cortical regions are coloured by the highest cut-point the lobe has reached by
then in the model (mean event stage over the MCMC samples, src.trained_model.event_stages). Only the cortex is
drawn: hippocampus and amygdala (part of MTL) lie inside the brain. Drawing takes ~2 s per stage, so every
figure is saved once in a cache folder and read from there afterwards (it depends on the model only).
"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

VIEWS = ["left_lateral", "left_medial", "right_medial", "right_lateral"]
NOT_REACHED = "#A9B1BB"                                   # a lobe that has not reached z 2 yet
OUTSIDE = "#FFFFFF"                                       # regions in no lobe of the model (and the medial wall)
LEVEL_COLOURS = ["#9ECAE1", "#4292C6", "#08306B"]         # z 2, z 5, z 10: the app's blue scale, no black
VERSION = 3                                               # raise when the drawing changes: old cached files are not used


def atlas_name(region):
    """The model's FreeSurfer name -> yabplot's aparc name: ctx-lh-entorhinal -> L_entorhinal."""
    return region.replace("ctx-lh-", "L_").replace("ctx-rh-", "R_")


def region_levels(events, regions, stage):
    """{aparc region: level} at one stage: 1 = its lobe has not reached the first cut-point, 2 = first cut-point
    (z 2), 3 = second, ... (yabplot draws 0 as "no data", so the levels start at 1). Regions in no lobe are left
    out (drawn white). events: lobes x cut-points table of mean event stages; regions: {lobe: [FreeSurfer regions]}."""
    levels = {}
    for lobe, row in events.iterrows():
        reached = int((row.dropna() <= stage).sum())       # cut-points are in increasing order
        for region in regions[lobe]:
            levels[atlas_name(region)] = reached + 1
    return levels


def draw_trajectory(events, regions, title, stages, views=VIEWS):
    """The figure: one row per stage, the views side by side, one legend. Needs yabplot (imported here: it
    loads VTK, which is slow, and only this figure uses it)."""
    import yabplot as yab
    cortex = set(yab.get_atlas_regions(atlas="aparc", category="cortical"))
    n_levels = events.shape[1]
    cmap = ListedColormap([NOT_REACHED] + LEVEL_COLOURS[:n_levels])
    fig, axes = plt.subplots(len(stages), 1, figsize=(11, 1.6 * len(stages) + 1.0), squeeze=False)
    for ax, stage in zip(axes[:, 0], stages):
        data = {r: v for r, v in region_levels(events, regions, stage).items() if r in cortex}
        before = set(fig.axes)
        yab.plot_cortical(data=data, atlas="aparc", ax=ax, views=list(views), layout=(1, len(views)), cmap=cmap,
                          vminmax=[0.5, n_levels + 1.5], nan_color=OUTSIDE)
        for extra in set(fig.axes) - before:             # yabplot adds a colour bar to every row: the legend is enough
            extra.remove()
        ax.text(-0.01, 0.5, f"stage {stage}", transform=ax.transAxes, ha="right", va="center", fontsize=11)
    for i, view in enumerate(views):
        axes[0, 0].text((i + 0.5) / len(views), 1.04, view.replace("_", " "), transform=axes[0, 0].transAxes,
                        ha="center", va="bottom", fontsize=10, color="#52606D")
    labels = ["not reached yet"] + [f"reached z {z:g}" for z in events.columns] + ["not in a lobe of the model"]
    shades = [NOT_REACHED] + LEVEL_COLOURS[:n_levels] + [OUTSIDE]
    fig.legend([Patch(facecolor=s, edgecolor="#9AA5B1") for s in shades], labels, loc="lower center",
               ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.1, right=0.99, top=1 - 0.75 / fig.get_figheight(), bottom=0.45 / fig.get_figheight(),
                        hspace=0.08)
    return fig


def cache_key(package, stages, views=VIEWS):
    """Changes whenever the model, its regions, the stages or the drawing change."""
    h = hashlib.sha1(np.ascontiguousarray(package["samples_sequence"]).tobytes())
    h.update(np.ascontiguousarray(package["samples_f"]).tobytes())
    h.update(json.dumps([package["regions"], package["subtype_names"], list(stages), list(views), VERSION]).encode())
    return h.hexdigest()[:12]


def trajectory_png(package, events_by_subtype, subtype, stages, folder):
    """The PNG of one subtype's trajectory: drawn the first time, read from the cache folder afterwards."""
    path = Path(folder) / cache_key(package, stages) / f"subtype{package['subtype_names'].index(subtype) + 1}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = draw_trajectory(events_by_subtype[subtype], package["regions"], f"{subtype}: trajectory on the brain",
                              stages)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def picture_figure(path, dpi=150):
    """A saved PNG as a matplotlib figure of the same size (for the app's Save PNG / PDF and the report)."""
    image = plt.imread(path)
    fig = plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    return fig
