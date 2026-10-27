"""Brain figures: tau per subtype (and per stage group) drawn on the cortex, with the outlines of the
Desikan-Killiany atlas from the ggseg package (python-ggseg, the Python port of the ggseg brain plots used
in many papers). Lateral and medial views of both hemispheres.

Our regions are matched by name. A region named after a lobe (e.g. L_frontal, R_MTL) colours every atlas
region of that lobe; a region named after an atlas region (e.g. L_entorhinal) colours just that one.
Left or right from the start of the name (L_, R_, lh_, rh_, left_, right_).
"""
import functools
import os.path as op
import re

import ggseg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ggseg import _svg_parse_
from scipy import ndimage
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

ATLAS = op.join(op.dirname(ggseg.__file__), "data", "dk")
LOBES = {    # Desikan-Killiany regions of each lobe (cingulate and insula belong to none)
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "mtl": ["entorhinal", "parahippocampal"],
}
LOBE_WORDS = {"mtl": ["mtl", "medial_temporal", "medialtemporal", "hippocamp"], "temporal": ["temporal"],
              "parietal": ["parietal"], "occipital": ["occipital"], "frontal": ["frontal"]}
ATLAS_REGIONS = sorted({r for regions in LOBES.values() for r in regions} |
                       {"caudalanteriorcingulate", "rostralanteriorcingulate", "posteriorcingulate",
                        "isthmuscingulate", "insula"})
NO_DATA = "#E4E7EB"      # no region of the data there (e.g. cingulate, insula)
BELOW = "#BFC5CE"        # a value below the first cut-point: normal, as in Vogel et al. 2021
# Vogel et al. 2021 colours: from red at the first cut-point (z 2) through orange to yellow at the top
VOGEL_COLOURS = "#C8211B, #F26B1D, #FBB735, #F9E833"


def side_of(region):
    """"L", "R" or None (both) from the start of a region name."""
    match = re.match(r"^(l|lh|left|r|rh|right)[_\-.]", region.lower())
    return None if not match else ("L" if match.group(1).startswith("l") else "R")


def atlas_regions_of(region):
    """The atlas regions one of our regions stands for: itself if it is named after one (entorhinal,
    precuneus, ...), else every region of its lobe; [] if the name matches nothing."""
    name = region.lower().replace("-", "").replace(" ", "")
    for atlas_region in sorted(ATLAS_REGIONS, key=len, reverse=True):     # longest name first
        if atlas_region in name:
            return [atlas_region]
    for lobe, words in LOBE_WORDS.items():                                # MTL before temporal
        if any(word in name for word in words):
            return LOBES[lobe]
    return []


def lobe_of(region):
    """True-ish if the brain figure can place this region (kept for the app's check)."""
    return bool(atlas_regions_of(region))


def atlas_values(means):
    """{(side, atlas region): value} from {our region: value}; a region without a side counts for both."""
    found = {}
    for region, value in means.items():
        side = side_of(region)
        for atlas_region in atlas_regions_of(region):
            for s in ([side] if side else ["L", "R"]):
                found.setdefault((s, atlas_region), []).append(value)
    return {key: float(np.mean(v)) for key, v in found.items()}


@functools.lru_cache(maxsize=None)
def outline(name):
    """The matplotlib Path of one atlas file (a region, or a whole view such as lateral_left)."""
    codes, verts = _svg_parse_(open(op.join(ATLAS, name)).read())
    return Path(verts, codes)


@functools.lru_cache(maxsize=None)
def view_box(view):
    """(x0, y0, x1, y1) of one view (lateral_left, medial_left, lateral_right, medial_right) in atlas units."""
    path = outline(view)
    points = path.vertices[path.codes != Path.CLOSEPOLY]
    return (*points.min(axis=0), *points.max(axis=0))


def region_points(atlas_regions, hemisphere, view):
    """The outline points of some atlas regions that lie in one view (to place a label on them)."""
    x0, y0, x1, y1 = view_box(view)
    points = [outline(f"{r}_{hemisphere}").vertices[outline(f"{r}_{hemisphere}").codes != Path.CLOSEPOLY]
              for r in atlas_regions]
    points = np.concatenate(points) if points else np.zeros((0, 2))
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return points[inside]


@functools.lru_cache(maxsize=None)
def lobe_map(view, step=2.0):
    """(x, y, lobe number of every grid point) of one view: 0 = no lobe, 1, 2, ... = the lobes of LOBES.
    Made once per view (the atlas does not change); used to draw the borders between the lobes."""
    hemisphere = "left" if view.endswith("left") else "right"
    x0, y0, x1, y1 = view_box(view)
    xs, ys = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
    grid = np.column_stack([g.ravel() for g in np.meshgrid(xs, ys)])
    lobes = np.zeros((len(ys), len(xs)), dtype=int)
    for number, atlas_regions in enumerate(LOBES.values(), start=1):
        inside = np.zeros(len(grid), dtype=bool)
        for region in atlas_regions:
            inside |= outline(f"{region}_{hemisphere}").contains_points(grid)
        # the atlas has thin gaps between its regions: close them, so a lobe is one area
        inside = ndimage.binary_fill_holes(ndimage.binary_closing(inside.reshape(lobes.shape), iterations=4))
        lobes[inside & (lobes == 0)] = number
    return xs, ys, lobes


def ink_on(face):
    """White or dark text, whichever reads better on this colour."""
    r, g, b = matplotlib.colors.to_rgb(face)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "#1F2933"


def draw_view(ax, view, values, x, y, cmap, vmax, labels=(), low=0):
    """One view of one hemisphere with its top-left corner at (x, y) in the axes (y grows down). Each atlas
    region is filled with values[(side, region)] on the colour scale low ... vmax; below low (normal) it is
    grey, as in Vogel et al. 2021. A region's edge has its own colour, so the atlas regions of one of our
    regions (a lobe) merge into one area; white lines separate the lobes. labels: (text, atlas regions,
    value) written on that area - the small MTL gets its value under the view, with an arrow to it.
    Returns the view's width and height."""
    side = "L" if view.endswith("left") else "R"
    hemisphere = "left" if side == "L" else "right"
    x0, y0, x1, y1 = view_box(view)
    move = Affine2D().translate(x - x0, y - y0) + ax.transData
    clip = Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, transform=move, visible=False)
    ax.add_patch(clip)
    colours = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    shape = PathPatch(outline(view), facecolor="#C9CED6", edgecolor="#9AA5B1", lw=2.5, transform=move)
    ax.add_patch(shape)                                      # the whole view: its outline and the medial wall
    shape.set_clip_path(clip)
    for region in ATLAS_REGIONS:
        value = values.get((side, region))
        if value is None or value < low:
            face = NO_DATA if value is None else BELOW
        else:
            face = colours(float(np.clip((value - low) / (vmax - low), 0, 1)))
        patch = PathPatch(outline(f"{region}_{hemisphere}"), facecolor=face, edgecolor=face, lw=1.5,
                          transform=move)
        ax.add_patch(patch)
        patch.set_clip_path(clip)
    xs, ys, lobes = lobe_map(view)                           # white lines between the lobes
    for number in range(1, len(LOBES) + 1):
        if (lobes == number).any():
            ax.contour(xs - x0 + x, ys - y0 + y, (lobes == number).astype(float), levels=[0.5], colors="white",
                       linewidths=1.8, zorder=3)
    for text, atlas_regions, value in labels:
        points = region_points(atlas_regions, hemisphere, view)
        if len(points) <= 3:
            continue
        cx, cy = points.mean(axis=0)
        if set(atlas_regions) <= set(LOBES["mtl"]):          # small area: its value under the view, with an arrow
            ax.annotate(f"MTL {text}", xy=(cx - x0 + x, cy - y0 + y), xytext=(cx - x0 + x, y1 - y0 + y + 95),
                        ha="center", va="top", fontsize=7.5, fontweight="bold", color="#1F2933", zorder=5,
                        arrowprops={"arrowstyle": "-|>", "color": "#1F2933", "lw": 0.8, "shrinkA": 0,
                                    "shrinkB": 1, "mutation_scale": 7})
        else:
            face = BELOW if value < low else colours(float(np.clip((value - low) / (vmax - low), 0, 1)))
            ax.text(cx - x0 + x, cy - y0 + y, text, ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=ink_on(face), zorder=4)
    return x1 - x0, y1 - y0


def lobe_labels(means, regions, side, view):
    """Labels (value, atlas regions, value) for our regions of one side that are best seen in this view."""
    hemisphere = "left" if side == "L" else "right"
    other = view.replace("lateral", "medial") if view.startswith("lateral") else view.replace("medial", "lateral")
    labels = []
    for region in regions:
        atlas_regions = atlas_regions_of(region)
        if side_of(region) not in (side, None) or not atlas_regions or np.isnan(means[region]):
            continue
        here = len(region_points(atlas_regions, hemisphere, view))
        if here >= len(region_points(atlas_regions, hemisphere, other)):    # most of it shows in this view
            labels.append((f"{means[region]:.1f}", atlas_regions, means[region]))
    return labels


def draw_four_views(ax, values, means, regions, cmap, vmax, low, sides=True):
    """One brain as in Vogel et al. 2021 Fig. 1c: lateral views of the left and right hemisphere on top,
    their medial views below. means given: each region's value written on it. sides: "left" and "right"
    written above the two hemispheres."""
    rows = [["lateral_left", "lateral_right"], ["medial_left", "medial_right"]]
    room = 110 if means is not None else 30                  # under the medial views: the MTL value
    y, width = 0, 0
    for views in rows:
        x, height = 0, 0
        for view in views:
            side = "L" if view.endswith("left") else "R"
            labels = lobe_labels(means, regions, side, view) if means is not None else []
            w, h = draw_view(ax, view, values, x, y, cmap, vmax, labels, low)
            if sides and y == 0:
                ax.text(x + w / 2, -25, "left" if side == "L" else "right", ha="center", va="bottom",
                        fontsize=8, color="#52606D")
            x, height = x + w + 30, max(height, h)
        y, width = y + height + room, max(width, x - 30)
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(y, -150 if sides else -10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_brain_figure(z, table, regions, subtype_titles, n_events, vmax=12, cmap=None, low=2, colors=None):
    """Tau of the subtypes on the brain in one figure, following Vogel et al. 2021 (Fig. 1c and 1e).
    Top: each subtype's average z-score (its tau+ people), lateral and medial views of both hemispheres,
    with each region's value. Bottom: each subtype through the SuStaIn stages - columns at stages 6, 11,
    16, ...; each brain is the mean of the people at that stage and up to four stages lower; the same four
    views as on top. Grey = below the first cut-point (normal).
    None if no region name matches the atlas."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyArrowPatch as Arrow
    from src.figures import colour_list, subtype_colors
    if not any(atlas_regions_of(r) for r in regions):
        return None
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("vogel", colour_list(VOGEL_COLOURS))
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean()   # subtypes x regions
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    stages = list(range(6, n_events + 1, 5)) or [n_events]
    k, g = len(subtype_titles), len(stages)
    shades = subtype_colors(k, colors)
    fig = plt.figure(figsize=(max(2.9 * k, 3.0 * g + 1.2), 4.4 + 2.2 * k))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 2.2 * k], hspace=0.18)
    top = grid[0].subgridspec(1, k, wspace=0.18)
    bottom = grid[1].subgridspec(k, g, hspace=0.3, wspace=0.05)
    for n, title in enumerate(subtype_titles, start=1):               # Fig. 1c: one block per subtype
        ax = fig.add_subplot(top[n - 1])
        row_means = means.loc[n] if n in means.index else None
        draw_four_views(ax, atlas_values(row_means) if row_means is not None else {}, row_means, regions,
                        cmap, vmax, low)
        share = 100 * counts.get(n, 0) / max(int(tau_pos.sum()), 1)
        ax.text(-0.03, 0.5, f"{title}\n{share:.0f}% (n = {counts.get(n, 0)})", rotation=90, ha="right",
                va="center", transform=ax.transAxes, fontsize=9, fontweight="bold")
    stage_axes = np.empty((k, g), dtype=object)
    for n, title in enumerate(subtype_titles, start=1):               # Fig. 1e: one row per subtype
        people = (table["ml_subtype"] == n) & tau_pos
        for col, stage in enumerate(stages):
            ax = fig.add_subplot(bottom[n - 1, col])
            ax.set_facecolor("none")                                  # the subtype arrow shows behind
            stage_axes[n - 1, col] = ax
            members = people & table["ml_stage"].between(stage - 4, stage)
            ax.axis("off")
            if members.sum() == 0:
                ax.text(0.5, 0.5, "nobody", ha="center", va="center", fontsize=8, color="#9AA5B1",
                        transform=ax.transAxes)
            else:
                draw_four_views(ax, atlas_values(z[members][regions].mean()), None, regions, cmap, vmax, low,
                                sides=n == 1)
                ax.text(0.5, -0.02, f"n = {int(members.sum())}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=7, color="#52606D")
        ax0 = stage_axes[n - 1, 0]
        ax0.text(-0.12, 0.5, title, rotation=90, ha="right", va="center", transform=ax0.transAxes,
                 fontsize=9, fontweight="bold")
    fig.canvas.draw()                                                  # positions are known from here on
    for n in range(k):                                                 # the subtype's arrow behind its row
        left, right = stage_axes[n, 0].get_position(), stage_axes[n, -1].get_position()
        y = (left.y0 + left.y1) / 2
        fig.patches.append(Arrow((left.x0, y), (right.x1 + 0.03, y), transform=fig.transFigure, zorder=-1,
                                 arrowstyle="simple,head_width=1.4,head_length=1.1,tail_width=0.55",
                                 mutation_scale=22, color=shades[n], alpha=0.3, lw=0))
    first, last = stage_axes[0, 0].get_position(), stage_axes[0, -1].get_position()
    y = first.y1 + 0.045
    fig.patches.append(Arrow((first.x0, y), (last.x1 + 0.03, y), transform=fig.transFigure, arrowstyle="-|>",
                             mutation_scale=14, lw=1.2, color="#52606D"))
    fig.text(first.x0, y + 0.008, "SuStaIn stage (early to late)", ha="left", va="bottom", fontsize=9, color="#1F2933")
    blocks = [ax for ax in fig.axes[:k]]
    top_y = max(ax.get_position().y1 for ax in blocks)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(low, vmax), cmap=cmap),
                       cax=fig.add_axes([0.35, top_y + 0.03, 0.3, 0.01]), orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    bar.ax.xaxis.set_ticks_position("top")
    bar.ax.xaxis.set_label_position("top")
    bar.set_label(f"average z-score; grey = below {low:g} (normal)", fontsize=9)
    return fig
