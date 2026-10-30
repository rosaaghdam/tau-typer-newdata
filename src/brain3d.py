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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

VIEWS = ["left_lateral", "left_medial", "right_medial", "right_lateral"]
NOT_REACHED = "#A9B1BB"                                   # a lobe that has not reached z 2 yet
OUTSIDE = "#FFFFFF"                                       # regions in no lobe of the model (and the medial wall)
LEVEL_COLOURS = ["#9ECAE1", "#4292C6", "#08306B"]         # z 2, z 5, z 10: the app's blue scale, no black
VERSION = 4                                               # raise when the drawing changes: old cached files are not used


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


def draw_brain_rows(rows, cmap, vminmax, title, views=VIEWS, legend=None, colour_label=None):
    """Rows of 3D brains: rows = [(row label, {aparc region: value})]; the views side by side in every row. Below:
    legend = (colours, labels) for categories, or a colour bar labelled colour_label. Needs yabplot (imported
    here: it loads VTK, which is slow, and only these figures use it)."""
    import yabplot as yab
    cortex = set(yab.get_atlas_regions(atlas="aparc", category="cortical"))
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.6 * len(rows) + 1.0), squeeze=False)
    for ax, (label, values) in zip(axes[:, 0], rows):
        before = set(fig.axes)
        yab.plot_cortical(data={r: v for r, v in values.items() if r in cortex}, atlas="aparc", ax=ax,
                          views=list(views), layout=(1, len(views)), cmap=cmap, vminmax=list(vminmax),
                          nan_color=OUTSIDE)
        for extra in set(fig.axes) - before:             # yabplot adds a colour bar to every row: one is enough
            extra.remove()
        ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=11)
    for i, view in enumerate(views):
        axes[0, 0].text((i + 0.5) / len(views), 1.04, view.replace("_", " "), transform=axes[0, 0].transAxes,
                        ha="center", va="bottom", fontsize=10, color="#52606D")
    if legend is not None:
        colours, labels = legend
        fig.legend([Patch(facecolor=c, edgecolor="#9AA5B1") for c in colours], labels, loc="lower center",
                   ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    bottom = (0.45 if legend is not None else 1.1) / fig.get_figheight()
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 0.75 / fig.get_figheight(), bottom=bottom, hspace=0.08)
    if colour_label is not None:
        bar_ax = fig.add_axes([0.3, 0.6 / fig.get_figheight(), 0.4, 0.12 / fig.get_figheight()])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*vminmax), cmap=cmap), cax=bar_ax,
                           orientation="horizontal", ticks=list(range(2, int(vminmax[1]) + 1, 2)))
        bar.set_label(colour_label, fontsize=9)
        bar.outline.set_visible(False)
    return fig


def draw_trajectory(events, regions, title, stages, views=VIEWS):
    """The model's trajectory: one row per stage, each lobe coloured by the highest cut-point it has reached."""
    n_levels = events.shape[1]
    cmap = ListedColormap([NOT_REACHED] + LEVEL_COLOURS[:n_levels])
    rows = [(f"stage {stage}", region_levels(events, regions, stage)) for stage in stages]
    labels = ["not reached yet"] + [f"reached z {z:g}" for z in events.columns] + ["not in a lobe of the model"]
    shades = [NOT_REACHED] + LEVEL_COLOURS[:n_levels] + [OUTSIDE]
    return draw_brain_rows(rows, cmap, (0.5, n_levels + 1.5), title, views, legend=(shades, labels))


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


# ---------- Step 4: your people's average z-scores per subtype and stage group ----------
LOW_Z = 2                                                  # below the first cut-point = normal: drawn grey
Z_CMAP = LinearSegmentedColormap.from_list("z_on_brain", [(0, NOT_REACHED), (0.999 / 9, NOT_REACHED),
                                                          (1 / 9, "#C6DBEF"), (4 / 9, "#2171B5"), (1, "#08306B")])
Z_RANGE = (1, 10)                                          # 1 = grey (below 2); 2 -> light blue ... 10 -> dark blue


def lobe_z_values(means, regions):
    """{aparc region: value to draw} from one row of lobe means: below 2 -> 1 (grey), else the mean up to 10."""
    values = {}
    for lobe, mean in means.items():
        if np.isnan(mean):
            continue
        value = 1.0 if mean < LOW_Z else min(float(mean), Z_RANGE[1])
        for region in regions[lobe]:
            values[atlas_name(region)] = value
    return values


def subtype_z_rows(z, table, lobes, regions, subtype_number, stage_edges):
    """The rows of one subtype: all its tau+ people, then each stage group (e.g. 1-6, 7-12, ...); empty groups are
    left out. Each row: the people's average z-score per lobe on their regions."""
    people = (table["subtype_number"] == subtype_number) & (table["ml_stage"] > 0)
    rows = [(f"all stages\nn = {int(people.sum())}", lobe_z_values(z.loc[people, lobes].mean(), regions))]
    for low, high in zip(stage_edges[:-1], stage_edges[1:]):
        group = people & table["ml_stage"].between(low, high - 1)
        if group.any():
            rows.append((f"stages {low}-{high - 1}\nn = {int(group.sum())}",
                         lobe_z_values(z.loc[group, lobes].mean(), regions)))
    return rows


def subtype_brains_png(z, table, lobes, regions, subtype_number, name, stage_edges, folder):
    """The PNG of one subtype in your data: drawn the first time, read from the cache folder afterwards (the key
    changes with the people, their z-scores, subtypes and stages)."""
    rows = subtype_z_rows(z, table, lobes, regions, subtype_number, stage_edges)
    h = hashlib.sha1(json.dumps([[label, sorted(values.items())] for label, values in rows] + [name, VERSION]).encode())
    path = Path(folder) / "your_data" / f"{h.hexdigest()[:16]}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = draw_brain_rows(rows, Z_CMAP, Z_RANGE, f"{name}: trajectory in your data",
                              colour_label=f"average z-score of the people (grey = below {LOW_Z}, normal)")
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path
