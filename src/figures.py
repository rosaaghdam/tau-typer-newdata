"""Figures of SuStaIn results. Drawing is done by pySuStaIn's own plot functions where possible."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from scipy import stats

from src.subtype_analysis import region_pairs

plt.style.use(Path(__file__).with_name("tau_typer.mplstyle"))   # one look for every figure


def plot_pvd(result, Z_vals, regions, subtype_titles, save_path, stage_interval=2, subtype_order=None,
             legend=True, people=True):
    """Clearer positional variance diagram (PVD), drawn by pySuStaIn's plot_positional_var.

    Rows = regions, columns = SuStaIn stages. Red = region reaches z 2, magenta = z 5,
    blue = z 10. Dark square = SuStaIn is sure the event happens at that stage; pale = unsure.
    save_path is given without ".png" (None = do not save).
    subtype_order: which of pySuStaIn's subtypes to draw first, second, ... (None = pySuStaIn's choice).
    stage_interval: a stage number every ... stages (2: 1, 3, 5, ...). legend: the colour legend on top.
    Each title gets pySuStaIn's f and n, as pySuStaIn writes them: f = the subtype's share of ALL people
    (mean over the MCMC samples; S0 people are shared out over the subtypes too), n = f x number of people.
    tau+ = people with stage > 0 assigned to the subtype - the number in the app's tables and other figures.
    people=False: the model alone (no people yet), titles show only f.
    """
    k = len(subtype_titles)
    shares = np.mean(result["samples_f"], axis=1)                     # pySuStaIn's own numbers
    by_f = np.argsort(shares)[::-1]                                   # pySuStaIn's numbering of ml_subtype
    order = by_f if subtype_order is None else subtype_order
    if people:
        n_people = len(result["ml_subtype"])
        tau = result["ml_stage"].ravel() > 0
        tau_counts = np.bincount(result["ml_subtype"].ravel()[tau].astype(int), minlength=k)
        tau_of = {by_f[j]: tau_counts[j] for j in range(k)}           # by the subtype's number in samples_f
        subtype_titles = [f"{title} (f={shares[i]:.2f}, n={round(shares[i] * n_people)}, tau+ {tau_of[i]})"
                          for title, i in zip(subtype_titles, order)]
    else:
        subtype_titles = [f"{title} (f={shares[i]:.2f})" for title, i in zip(subtype_titles, order)]
    n_rows, n_cols = (1, k) if k < 3 else (2, int(np.ceil(k / 2)))     # pySuStaIn's own grid (up to 6)
    n_events = int((Z_vals > 0).sum())
    panel_height = 7 / n_cols * len(regions) / n_events + 1.2          # cells are square in the PVD
    figs, _ = ZscoreSustain.plot_positional_var(
        result["samples_sequence"], result["samples_f"], np.inf, Z_vals,
        biomarker_labels=regions, subtype_titles=subtype_titles,
        figsize=(14, n_rows * panel_height), title_font_size=10, label_font_size=8, stage_font_size=8,
        stage_interval=stage_interval, subtype_order=subtype_order)
    fig = figs[0]
    for ax in fig.axes:
        ax.set_xlabel("")                                       # columns = stages (said in the app's help text)
    top = 1 - (0.95 if legend else 0.5) / fig.get_figheight()          # room for the legend (and titles)
    fig.subplots_adjust(hspace=0.35, wspace=0.08, top=top)
    if legend:     # one legend for the colours (pySuStaIn's: 1st cut-point red, 2nd magenta, 3rd blue, ...)
        colours = ["red", "magenta", "blue", "#8000ff", "cyan", "#00ff80"]
        levels = np.unique(Z_vals[Z_vals > 0])
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours[:len(levels)]]
        fig.legend(handles, [f"z-score {z:g}" for z in levels], loc="upper center", ncol=len(levels),
                   bbox_to_anchor=(0.5, 1), frameon=False, fontsize=9)             # on top
    if save_path is not None:
        fig.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_stage_progression(z, table, regions, subtype_titles, stage_edges, save_path, vmax=12, cmap="Reds"):
    """Average z-score per region for real people, grouped by subtype and stage (paper Fig. 1e idea).

    One panel per subtype: rows = regions, columns = stage groups (e.g. 1-6, 7-12, ...), n = people.
    Darker = more tau on average. Reading left to right shows tau spreading with stage.
    """
    labels = [f"{a}-{b - 1}" for a, b in zip(stage_edges[:-1], stage_edges[1:])]
    stage_group = pd.cut(table["ml_stage"], stage_edges, right=False, labels=labels)
    n = len(subtype_titles)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 1.6, 0.36 * len(regions) + 2.2), squeeze=False)
    for k, (ax, title) in enumerate(zip(axes[0], subtype_titles), start=1):
        people = (table["ml_subtype"] == k) & (table["ml_stage"] > 0)      # this subtype, tau+ only
        means = z[people].groupby(stage_group[people], observed=False)[regions].mean().T  # regions x groups
        counts = stage_group[people].value_counts().reindex(labels)
        image = heat_cells(ax, means.to_numpy(), cmap, vmax, numbers=False)
        ax.set_xticks(range(len(labels)), [f"{lab}\nn={c}" for lab, c in zip(labels, counts)])
        ax.set_yticks(range(len(regions)), [nice_region(r) for r in regions] if k == 1 else [])
        ax.set_title(f"{title}\n{int(people.sum())} tau+ people", fontweight="bold")
    axes[0, 0].set_xlabel("stage group")
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.02, pad=0.02, shrink=0.7)
    style_colorbar(bar, "average z-score")
    if save_path:                                         # the app shows the figure without saving
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def nice_region(name):
    """L_MTL -> "L MTL" (easier to read in a figure)."""
    return name.replace("_", " ")


def heat_cells(ax, values, cmap, vmax, numbers=True, aspect="auto"):
    """A heat map with a thin white gap around every cell, no frame and no tick marks; numbers=True writes
    each value in its cell (white on dark cells, whatever the colour map). Returns the image (for a colour bar)."""
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect=aspect)
    rows, cols = values.shape
    ax.set_xticks(np.arange(-0.5, cols), minor=True)
    ax.set_yticks(np.arange(-0.5, rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if numbers:
        for (row, col), value in np.ndenumerate(values):
            if not np.isnan(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.6      # white numbers on dark cells
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "#1F2933")
    return image


def style_colorbar(bar, label):
    """A quiet colour bar: no outline, small ticks, label in ink."""
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors="#52606D")
    bar.set_label(label, color="#1F2933")


def left_right_layout(regions, max_cols=6):
    """Panel order for region figures: a row of left regions with their right partners in the row
    right below (same column), at most max_cols pairs per row; regions without a partner come last.
    Returns (list of region names, None = empty panel; number of columns)."""
    pairs = list(region_pairs(regions).values())
    paired = {region for pair in pairs for region in pair}
    others = [region for region in regions if region not in paired]
    n_cols = min(len(pairs), max_cols) if pairs else min(len(regions), 5)
    cells = []
    for start in range(0, len(pairs), n_cols):
        chunk = pairs[start:start + n_cols]
        empty = [None] * (n_cols - len(chunk))
        cells += [left for left, _ in chunk] + empty + [right for _, right in chunk] + empty
    return cells + others, n_cols


def plot_region_histograms(data, regions, xlabel="value (e.g. SUVR)", color="#4C72B0"):
    """One small histogram per region column - raw values in tab 1, z-scores in tab 2.

    Left regions on top, each right region right below its left partner (same x-axis), so
    left/right differences are easy to see. Shows the "normal" bump and the tau+ tail of each
    region, and makes outliers or odd extra bumps easy to spot. xlabel is the label under each panel.
    """
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")   # text becomes missing
    cells, n_cols = left_right_layout(list(regions))
    n_rows = max(1, int(np.ceil(len(cells) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows), squeeze=False)
    pair_rows = 2 * int(np.ceil(len(region_pairs(list(regions))) / n_cols))
    for col in range(n_cols):                             # left and right partner: same x-axis
        for row in range(1, pair_rows, 2):
            axes[row, col].sharex(axes[row - 1, col])
    for ax, region in zip(axes.flat, cells + [None] * (axes.size - len(cells))):
        if region is None:
            ax.set_visible(False)                         # empty panel
            continue
        ax.hist(values[region].dropna(), bins=40, color=color)
        ax.set_title(region, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        if ax.get_subplotspec().is_first_col():                   # y label only on the left
            ax.set_ylabel("people", fontsize=8)
    fig.tight_layout()
    return fig


def plot_missing(data):
    """Missing values of the whole table in one figure. Left: every column that has missing values (most
    missing on top) against every person (people with the most missing values on the left); dark = missing,
    so patterns show (e.g. the same people missing all cognitive tests). Right: how many values each of these
    columns misses (count and %). Columns with no missing value are named in the title."""
    missing = data.isna()
    per_column = missing.sum().sort_values(ascending=False)
    with_missing = per_column[per_column > 0]
    complete_people = int((~missing.any(axis=1)).sum())
    summary = (f"{int(missing.values.sum()):,} of {missing.size:,} values missing ({100 * missing.values.mean():.1f}%); "
               f"{complete_people} of {len(data)} rows complete")
    if with_missing.empty:
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No missing values in any column.", ha="center", va="center", fontsize=10)
        return fig
    people = missing[with_missing.index].sum(axis=1).sort_values(ascending=False, kind="stable").index
    matrix = missing.loc[people, with_missing.index].T.to_numpy()          # columns x people
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.3 * len(with_missing) + 1.6), sharey=True,
                                   gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04})
    shades = matplotlib.colors.ListedColormap(["#E4E7EB", "#0F5C6E"])       # present, missing
    ax1.imshow(matrix, cmap=shades, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax1.set_yticks(range(len(with_missing)), with_missing.index)
    ax1.set_xticks([])
    ax1.set_xlabel(f"{len(data)} rows (most missing values on the left); dark = missing")
    share = 100 * with_missing / len(data)
    bars = ax2.barh(range(len(with_missing)), share, color="#0F5C6E", height=0.7)
    bars[0].set_color("#9A5B00")                                           # the most missing column
    for row, (count, pct) in enumerate(zip(with_missing, share)):
        ax2.text(pct + share.max() * 0.03, row, f"{count} ({pct:.0f}%)", va="center", fontsize=8)
    ax2.set_xlim(0, share.max() * 1.45)
    ax2.set_xlabel("% missing")
    ax2.tick_params(axis="y", length=0)
    for ax in (ax1, ax2):
        for side in ax.spines.values():
            side.set_visible(False)
    complete_columns = [c for c in per_column.index if per_column[c] == 0]
    fig.suptitle(summary, x=0.02, ha="left", fontsize=10, fontweight="bold")
    if complete_columns:
        fig.text(0.02, -0.02, f"No missing values in the other {len(complete_columns)} columns.",
                 fontsize=8, color="#52606D", va="top")
    return fig


def plot_region_eda(data, regions, x=None, groups=None, colors=None, value_label="value (e.g. SUVR)",
                    single_color="#52606D"):
    """Explore the regional values (tab 1). One panel per region (left on top, right below):
    - x = None: the distribution of the values; with groups, each bar is split into the groups (stacked,
      so the bar's height is still all people in that bin);
    - x = a number column (e.g. age): each person as a dot, with a straight trend line per group - if the
      lines of two groups are not parallel, the trend differs between the groups (an interaction);
    - x = a category column (e.g. sex): a box per category, with the people as dots coloured by group.
    groups: a Series of group labels (e.g. sex, or age split into low/middle/high), same rows as data;
    colors: {group: colour}; single_color: the colour without groups. value_label: what the regional values are
    (axis label)."""
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")
    names = list(colors) if groups is not None else [None]
    member = {g: (groups == g) if g is not None else pd.Series(True, index=data.index) for g in names}
    x_values = None if x is None else data[x]
    x_is_number = x is not None and pd.to_numeric(x_values, errors="coerce").notna().mean() > 0.9
    cells, n_cols = left_right_layout(list(regions))
    n_rows = max(1, int(np.ceil(len(cells) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.7 * n_rows), squeeze=False)
    for ax, region in zip(axes.flat, cells + [None] * (axes.size - len(cells))):
        if region is None:
            ax.set_visible(False)
            continue
        if x is None:                                             # histogram; split: stacked groups per bin
            edges = np.histogram_bin_edges(values[region].dropna(), bins=30)
            ax.hist([values.loc[member[g], region].dropna() for g in names], bins=edges, stacked=True,
                    color=[colors[g] if g is not None else single_color for g in names],
                    edgecolor="white", linewidth=0.3)
        for g in names:
            color = colors[g] if g is not None else single_color
            y = values.loc[member[g], region]
            if x is None:
                continue
            elif x_is_number:
                xs = pd.to_numeric(x_values[member[g]], errors="coerce")
                ok = xs.notna() & y.notna()
                ax.scatter(xs[ok], y[ok], s=7, color=color, alpha=0.6, label=g)
                if ok.sum() >= 3:                                  # straight trend line of this group
                    slope, intercept = np.polyfit(xs[ok], y[ok], 1)
                    line_x = np.array([xs[ok].min(), xs[ok].max()])
                    ax.plot(line_x, intercept + slope * line_x, color=color, lw=2)
        if x is not None and not x_is_number:                     # boxes per category, people as dots
            categories = sorted(x_values.dropna().astype(str).unique())
            ax.boxplot([values.loc[x_values.astype(str) == c, region].dropna() for c in categories],
                       widths=0.5, showfliers=False, medianprops={"color": "#1F2933"})
            rng = np.random.default_rng(0)
            for g in names:
                color = colors[g] if g is not None else single_color
                for i, c in enumerate(categories, start=1):
                    y = values.loc[member[g] & (x_values.astype(str) == c), region].dropna()
                    ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=6, color=color, alpha=0.6,
                               label=g if i == 1 else None)
            ax.set_xticks(range(1, len(categories) + 1), categories, fontsize=8)
        ax.set_title(region, fontsize=10)
        ax.set_xlabel(x or value_label, fontsize=8)
        if ax.get_subplotspec().is_first_col():                   # y label only on the left
            ax.set_ylabel("people" if x is None else value_label, fontsize=8)
    if groups is not None:                                        # one legend for all panels
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in names]
        fig.legend(handles, [str(g) for g in names], loc="upper center", ncol=len(names), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    return fig


def plot_gmm_fit(values, gmm, region, normal_mean, normal_sd, z_events):
    """One region: histogram of the values + every GMM bell curve (normal one in green), and the
    cut-points drawn on the value axis (SUVR = normal mean + z * normal SD) - the tab-2 check."""
    values = np.asarray(values)
    cut_values = [normal_mean + z * normal_sd for z in z_events]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values, bins=60, density=True, color="lightgray", label="people")
    x = np.linspace(values.min(), max(values.max(), max(cut_values)), 500)
    means, weights = gmm.means_.ravel(), gmm.weights_
    sds = np.sqrt(gmm.covariances_.ravel())
    normal = np.argmin(means)
    for k in np.argsort(means):                          # draw from lowest to highest mean
        is_normal = k == normal
        ax.plot(x, weights[k] * stats.norm.pdf(x, means[k], sds[k]), lw=2.5 if is_normal else 1.5,
                color="#2E7D32" if is_normal else "#C62828", label="normal" if is_normal else "abnormal")
    for z, cut in zip(z_events, cut_values):
        ax.axvline(cut, ls="--", color="black", lw=1)
        ax.text(cut, ax.get_ylim()[1] * 0.95, f" z={z:g}", fontsize=9, va="top")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))                    # one legend entry per label
    ax.legend(unique.values(), unique.keys())
    ax.set_title(f"{region}: normal mean {normal_mean:.3f}, normal SD {normal_sd:.3f}")
    ax.set_xlabel("value (e.g. SUVR)")
    ax.set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_subtype_patterns(z, table, regions, subtype_titles, vmax=12, cmap="Reds"):
    """Average z-score per region in each subtype, tau+ people only (paper Fig. 1a/c, as a heat map).

    Columns = subtypes (n = number of tau+ people), rows = regions; square cells with the average z in them.
    """
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean().T   # regions x subtypes
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    cell = 0.72                                            # inches: square cells
    fig, ax = plt.subplots(figsize=(cell * len(means.columns) + 2.6, cell * len(regions) + 1.2))
    image = heat_cells(ax, means.to_numpy(), cmap, vmax, aspect="equal")
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(means.columns)), [f"{subtype_titles[k - 1]}\nn = {counts[k]}" for k in means.columns],
                  rotation=0)
    ax.set_yticks(range(len(regions)), [nice_region(r) for r in regions])
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, shrink=0.7)
    style_colorbar(bar, "average z-score (tau+ people)")
    return fig


# Subtype colours: teal, pink, brown, olive (+ raspberry, green, grey-brown, grey for 5-8 subtypes).
# Checked with a colour-blind simulator (dataviz validate_palette.js, all pairs): the first 4 stay apart
# for red-green colour blindness (worst Delta E 7.1 - fine because every figure also labels the subtypes)
# and use none of the diagnosis hues (blue, violet, yellow, red).
SUBTYPE_COLORS = ["#00897B", "#E87BA4", "#8A5A00", "#9E9D24", "#AD1457", "#008300", "#6D4C41", "#9E9E9E"]


def subtype_colors(k, colors=None):
    """One fixed colour per subtype (Subtype 1 = first colour), used by all tab-4 and tab-5 figures.
    colors: the user's own colours (Subtype 1 first); missing ones are the default colours."""
    colors = list(colors or []) + SUBTYPE_COLORS[len(colors or []):]
    return matplotlib.colors.to_rgba_array([colors[i % len(colors)] for i in range(k)])


def colour_list(text):
    """Colours typed by the user, e.g. "#00897B, #E87BA4" (or names: red, teal) -> a list.
    A code without "#" is accepted too; anything that is not a colour is left out."""
    colours = []
    for part in (text or "").split(","):
        part = part.strip()
        if len(part) == 6 and all(c in "0123456789abcdefABCDEF" for c in part):
            part = "#" + part
        if part and matplotlib.colors.is_color_like(part):
            colours.append(part)
    return colours


def colour_map(text, default):
    """A colour map from what the user typed: a matplotlib name (Reds, Blues, viridis, ...), one colour
    (white to that colour) or two or more colours from low to high (e.g. "#FFFFFF, #B71C1C").
    Anything else gives the default map."""
    text = (text or "").strip()
    if text in matplotlib.colormaps:
        return text
    colours = colour_list(text)
    if len(colours) == 1:
        colours = ["#FFFFFF"] + colours
    if len(colours) >= 2:
        return matplotlib.colors.LinearSegmentedColormap.from_list("typed", colours)
    return default


def override_colors(colors, typed):
    """{group: colour} with the user's typed colours (a list) used for the first groups, in order."""
    return {g: typed[i] if i < len(typed) else c for i, (g, c) in enumerate(colors.items())}


def plot_subtype_probabilities(prob_subtype, table, subtype_titles, seed=0, colors=None):
    """Every tau+ person inside a polygon with one subtype at each corner (paper Fig. 1b).
    Close to a corner = surely that subtype; towards the middle = a mix of subtypes.
    Corners: number (and %) of tau+ people in each subtype. Dashed box (as in the paper): the middle
    of the polygon, half its size; the legend gives the % of people inside and outside it.
    A little random jitter spreads the many people who sit exactly on a corner.

    prob_subtype: people x subtypes, columns already in the app's order (Subtype 1 first).
    """
    k = prob_subtype.shape[1]
    tau_pos = (table["ml_stage"] > 0).to_numpy()
    n_tau = tau_pos.sum()
    subtype = table["ml_subtype"].to_numpy()[tau_pos]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    angles = np.pi / 2 + 2 * np.pi * np.arange(k) / k
    corners = np.column_stack([np.cos(angles), np.sin(angles)])
    xy = prob_subtype[tau_pos] @ corners                  # weighted average of the corners
    inside = (np.abs(xy) <= 0.5).all(axis=1)             # inside the dashed box (before the jitter)
    xy = xy + np.random.default_rng(seed).normal(0, 0.02, xy.shape)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color="gray", lw=2, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=subtype_colors(k, colors)[subtype - 1], s=30, edgecolors="black", linewidths=0.5,
               zorder=2)
    box = ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ls="--", lw=2, color="black", zorder=3))
    for n, ((x, y), title) in enumerate(zip(corners, subtype_titles), start=1):
        count = (subtype == n).sum()
        side = np.sign(round(x, 6)) * 0.3                   # side corners: label away from the dots
        ax.text(1.12 * x + side, 1.17 * y, f"{title}\nn = {count} ({100 * count / n_tau:.0f}%)",
                ha="center", va="center", fontsize=9, fontweight="bold")     # text in ink, the dots carry colour
    ax.legend([box], [f"Inside: {100 * inside.mean():.0f}%\nOutside: {100 * (1 - inside.mean()):.0f}%"],
              loc="upper right", bbox_to_anchor=(1, 0.92), frameon=False, handlelength=1.5, handleheight=2.5)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_stages(table, subtype_titles, sd_limit=None, person=None, subtype=None, colors=None):
    """Top: % of the tau+ people at each SuStaIn stage 1, 2, ..., per subtype (paper Ext. Fig. 2). The S0
    people (stage 0, no abnormal tau - no subtype) are not drawn; their number is written under the axis.
    Middle: how sure SuStaIn is about each person's stage (table column stage_sd) against the most likely
    stage (high where events overlap - the pale squares of the PVD).
    Right: the same uncertainty per subtype. Dashed red line = sd_limit (above = uncertain stage).
    person: this person's dot gets a black ring. subtype (1, 2, ...): only this subtype in colour in the
    middle and right panels, the others grey."""
    k = len(subtype_titles)
    colors = subtype_colors(k, colors)
    stages = np.arange(table["ml_stage"].max() + 1)
    fig = plt.figure(figsize=(13, 9))                    # top: S0 + stages; bottom: uncertainty + per subtype
    grid = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(grid[0])
    bottom_row = grid[1].subgridspec(1, 2, width_ratios=[3, 1])
    ax2, ax3 = fig.add_subplot(bottom_row[0]), fig.add_subplot(bottom_row[1])
    bottom = np.zeros(len(stages))
    n_tau = max(int((table["ml_stage"] > 0).sum()), 1)
    n_s0 = int((table["ml_stage"] == 0).sum())
    for n, title in enumerate(subtype_titles, start=1):
        counts = table.loc[(table["ml_subtype"] == n) & (table["ml_stage"] > 0), "ml_stage"]
        heights = 100 * counts.value_counts().reindex(stages, fill_value=0).to_numpy() / n_tau
        ax1.bar(stages, heights, bottom=bottom, color=colors[n - 1],
                label=f"{title}: {100 * len(counts) / n_tau:.0f}% of tau+ (n={len(counts)})")
        bottom += heights
    ax1.set_xlabel(f"most likely SuStaIn stage (tau+ people, n = {n_tau}; not shown: S0, no abnormal tau, "
                   f"n = {n_s0})")
    ax1.set_ylabel("% of tau+ people")
    ax1.legend()

    tau = table[table["ml_stage"] > 0]
    if subtype is None:
        ax2.scatter(tau["ml_stage"], tau["stage_sd"], s=14, c=colors[tau["ml_subtype"].to_numpy() - 1])
    else:                                                 # chosen subtype in colour, on top of the others in grey
        chosen = tau["ml_subtype"] == subtype
        ax2.scatter(tau.loc[~chosen, "ml_stage"], tau.loc[~chosen, "stage_sd"], s=12, color="lightgray")
        ax2.scatter(tau.loc[chosen, "ml_stage"], tau.loc[chosen, "stage_sd"], s=20, color=colors[subtype - 1],
                    label=subtype_titles[subtype - 1])
    if person is not None and person in tau.index:
        ax2.scatter(tau.loc[person, "ml_stage"], tau.loc[person, "stage_sd"], s=150, facecolors="none",
                    edgecolors="black", linewidths=2, label=str(person))
    ax2.set_xlabel("most likely SuStaIn stage")
    ax2.set_ylabel("stage uncertainty: SD, in stages")
    ax1.set_xticks(stages[1:], stages[1:], fontsize=7)                 # every stage number
    ax1.set_xlim(0.4, stages[-1] + 0.6)
    ax2.set_xticks(stages[1:], stages[1:], fontsize=7)
    ax2.set_xlim(0.4, stages[-1] + 0.6)

    ax3.boxplot([tau.loc[tau["ml_subtype"] == n, "stage_sd"] for n in range(1, k + 1)], widths=0.6,
                patch_artist=True, medianprops={"color": "white"})
    for n, (patch, color) in enumerate(zip(ax3.patches, colors), start=1):
        patch.set_facecolor(color if subtype in (None, n) else "lightgray")
    ax3.set_xticks(range(1, k + 1), subtype_titles, rotation=30, ha="right")
    ax3.set_ylabel("stage uncertainty: SD, in stages")
    ax3.set_ylim(ax2.get_ylim())
    if sd_limit is not None:
        for ax in (ax2, ax3):
            ax.axhline(sd_limit, ls="--", color="#C62828", lw=1.2)
        n_above = (tau["stage_sd"] > sd_limit).sum()
        ax2.plot([], [], ls="--", color="#C62828", label=f"limit {sd_limit:g}: {n_above} above")
        top = max(ax2.get_ylim()[1], sd_limit * 1.1)
        ax2.set_ylim(top=top)
        ax3.set_ylim(ax2.get_ylim())
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_model_choice(likelihoods, cross_validation=None, traces=None):
    """How many subtypes? Log-likelihood (rises with every extra subtype) and, after cross-validation,
    CVIC (lower = better; the paper chose the number of subtypes by CVIC) - paper Ext. Fig. 1c-e.
    traces {k: MCMC log-likelihoods}: extra panel, one line per model; flat = the MCMC is stable."""
    n_panels = 1 + (cross_validation is not None) + (traces is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.2), squeeze=False)
    ax = axes[0, 0]
    ax.plot(likelihoods.index, likelihoods["mean log-likelihood"], "o-")
    ax.set_xlabel("number of subtypes")
    ax.set_ylabel("mean log-likelihood (MCMC)")
    ax.set_xticks(likelihoods.index)
    if cross_validation is not None:
        ax = axes[0, 1]
        cvic = cross_validation["CVIC"]
        ax.plot(cvic.index, cvic, "o-", color="#C62828")
        ax.plot(cvic.idxmin(), cvic.min(), "*", ms=16, color="#C62828", label=f"lowest: {cvic.idxmin()}")
        ax.set_xlabel("number of subtypes")
        ax.set_ylabel("CVIC (lower = better)")
        ax.set_xticks(cvic.index)
        ax.legend()
    if traces is not None:
        ax = axes[0, -1]
        for k, trace in traces.items():
            ax.plot(trace, lw=1, label=f"{k} subtype{'s' if k > 1 else ''}")
        ax.set_xlabel("MCMC sample")
        ax.set_ylabel("log-likelihood")
        ax.set_title("MCMC per model (flat = stable)", fontsize=10)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)    # beside, not on the lines
    fig.tight_layout()
    return fig


def pairs_together(flows):
    """flows: people from each subtype of model k (rows) to each subtype of model k+1 (columns).
    Of all pairs of people in one subtype of model k, the % still in one subtype of model k+1:
    per subtype of model k, and for everyone. 100% = the subtype stays together (or only grows)."""
    pairs = flows * (flows - 1) / 2                       # pairs inside each ribbon
    before = flows.sum(axis=1) * (flows.sum(axis=1) - 1) / 2
    per_subtype = 100 * pairs.sum(axis=1) / before.where(before > 0)
    return per_subtype, 100 * pairs.values.sum() / max(before.sum(), 1)


def plot_subtype_tree(subtypes_by_k, colors=None):
    """How people move between subtypes when one more subtype is allowed (paper Ext. Fig. 1f), drawn to
    show how stable the subtypes are.

    subtypes_by_k: {k: Series of subtype numbers per person (0 = S0)}, same people for every k.
    Each column = one model; boxes = subtypes (height = tau+ people); ribbons = people going from a subtype
    of model k to one of model k+1. Each subtype is placed level with the subtype it mostly comes from, so a
    subtype that stays together runs straight across. The main ribbon of each subtype (where most of its
    people go) is drawn strong and labelled with n and %; the others are light. Above each step: the % of
    people who follow their subtype's main ribbon. S0 (no tau) is left out.
    Subtype numbers are given again in each model (by size), so Subtype 1 of one model is not the same
    group as Subtype 1 of the next. Bottom: for each subtype, the % of its pairs of people still together
    in one subtype in the next model (pairs_together) - how well the subtype holds together.
    """
    ks = sorted(subtypes_by_k)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(2.8 * len(ks) + 1, 8.4), sharex=True,
                                  gridspec_kw={"height_ratios": [5.8, 2.2], "hspace": 0.15})
    n_people = max((s > 0).sum() for s in subtypes_by_k.values())
    gap, width = 0.04 * n_people, 0.22
    flows_of = {}
    order = {ks[0]: sorted(subtypes_by_k[ks[0]][subtypes_by_k[ks[0]] > 0].unique())}
    for k1, k2 in zip(ks[:-1], ks[1:]):                   # each new subtype next to where its people come from
        both = (subtypes_by_k[k1] > 0) & (subtypes_by_k[k2] > 0)
        flows = pd.crosstab(subtypes_by_k[k1][both], subtypes_by_k[k2][both])
        flows_of[(k1, k2)] = flows
        rank = {a: i for i, a in enumerate(order[k1])}
        weight = {b: (flows[b] * flows.index.map(rank)).sum() / max(flows[b].sum(), 1) for b in flows.columns}
        others = [b for b in subtypes_by_k[k2][subtypes_by_k[k2] > 0].unique() if b not in weight]
        order[k2] = sorted(weight, key=weight.get) + sorted(others)
    bottoms = {}                                           # (k, subtype) -> y of the box bottom
    for x, k in enumerate(ks):
        counts = subtypes_by_k[k][subtypes_by_k[k] > 0].value_counts()
        y = 0
        for subtype in order[k]:
            count = counts[subtype]
            ax.add_patch(plt.Rectangle((x - width / 2, y), width, count, color=subtype_colors(k, colors)[subtype - 1],
                                       zorder=3))
            ax.text(x, y + count / 2, f"{subtype}\n{count}", ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=4)
            bottoms[(k, subtype)] = y
            y += count + gap
    t = np.linspace(0, 1, 60)
    ease = (1 - np.cos(np.pi * t)) / 2                     # smooth S-shaped ribbons
    for x, (k1, k2) in enumerate(zip(ks[:-1], ks[1:])):
        flows = flows_of[(k1, k2)]
        out = {a: bottoms[(k1, a)] for a in flows.index}   # where the next ribbon leaves / arrives
        into = {b: bottoms[(k2, b)] for b in flows.columns}
        main = flows.idxmax(axis=1)                        # where most people of each subtype go
        followed = sum(flows.loc[a, main[a]] for a in flows.index) / max(flows.values.sum(), 1)
        ax.text(x + 0.5, -0.06 * n_people, f"{100 * followed:.0f}% follow\ntheir main ribbon", ha="center",
                va="top", fontsize=8, color="#52606D")
        rank_into = {b: i for i, b in enumerate(order[k2])}
        rank_out = {a: i for i, a in enumerate(order[k1])}
        pairs = [(a, b) for a in order[k1] if a in flows.index
                 for b in sorted(flows.columns, key=rank_into.get) if flows.loc[a, b] > 0]
        arrivals = {}                                      # ribbons into a box ordered by where they come from
        for a, b in sorted(pairs, key=lambda p: (rank_into[p[1]], rank_out[p[0]])):
            arrivals[(a, b)] = into[b]
            into[b] += flows.loc[a, b]
        xs = x + width / 2 + (1 - width) * t
        for a, b in pairs:
            n = flows.loc[a, b]
            low = out[a] + (arrivals[(a, b)] - out[a]) * ease
            strong = b == main[a]
            ax.fill_between(xs, low, low + n, color=subtype_colors(k1, colors)[a - 1], lw=0,
                            alpha=0.55 if strong else 0.15, zorder=2 if strong else 1)
            if strong:                                     # n and % of the subtype on its main ribbon
                ax.text(x + 0.5, (out[a] + arrivals[(a, b)]) / 2 + n / 2,
                        f"{n} ({100 * n / flows.loc[a].sum():.0f}%)", ha="center", va="center", fontsize=8,
                        zorder=5, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
            out[a] += n
        per_subtype, everyone = pairs_together(flows)      # bottom: pairs still together, per subtype
        subtypes = [a for a in order[k1] if a in flows.index]
        step = 0.6 / len(subtypes)
        for i, a in enumerate(subtypes):
            x_bar = x + 0.5 - 0.3 + step * (i + 0.5)
            value = per_subtype[a] if pd.notna(per_subtype[a]) else 0
            ax2.bar(x_bar, value, width=step * 0.85, color=subtype_colors(k1, colors)[a - 1])
            ax2.text(x_bar, value + 2, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
            ax2.text(x_bar, -4, str(a), ha="center", va="top", fontsize=7, color="#52606D")
        ax2.plot([x + 0.2, x + 0.8], [everyone, everyone], color="#1F2933", lw=1, ls="--")
        ax2.text(x + 0.82, everyone, f"all: {everyone:.0f}%", va="center", fontsize=7, color="#1F2933")
    ax2.set_ylim(0, 115)
    ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("% of pairs\nstill together")
    ax2.tick_params(axis="x", length=0)
    for name in ("top", "right"):
        ax2.spines[name].set_visible(False)
    ax2.text(0.5, 1.0, "How well each subtype (number under its bar) holds together: % of its pairs of people "
             "still in one subtype after the next split", transform=ax2.transAxes, ha="center", va="bottom", fontsize=8, color="#52606D")
    ax2.set_xticks(range(len(ks)), [f"{k} subtype{'s' if k > 1 else ''}" for k in ks])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.16 * n_people, None)
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel("tau+ people")
    return fig


def plot_person(person, z_person, regions, expected, prob_subtype, prob_stage, subtype_titles, colors=None):
    """One person's "fingerprint": their z-scores per region (bars) next to the average of the people
    in the same subtype and stage group (dots), plus their subtype and stage probabilities."""
    k = len(subtype_titles)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1, 1.6]})
    ax1.bar(range(len(regions)), z_person[regions], color="#4C72B0", label=person)
    if expected is not None:
        ax1.plot(range(len(regions)), expected[regions], "o", color="black", label="average of subtype + stage group")
    ax1.set_xticks(range(len(regions)), regions, rotation=45, ha="right")
    ax1.set_ylabel("z-score")
    ax1.legend(fontsize=8)
    ax2.bar(range(k), prob_subtype, color=subtype_colors(k, colors))
    ax2.set_xticks(range(k), subtype_titles, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("subtype probability")
    ax3.bar(range(len(prob_stage)), prob_stage / prob_stage.sum(), color="gray")
    ax3.set_xlabel("SuStaIn stage")
    ax3.set_ylabel("stage probability")
    fig.tight_layout()
    return fig


def plot_regional_contrasts(t_values, p_fdr, subtype_titles, cmap="RdBu_r"):
    """Paper Fig. 1d: t-value per region and subtype (this subtype vs all other tau+ people, same
    stage). Red = more tau than the others, blue = less. * = FDR-corrected p < 0.05."""
    limit = max(np.nanmax(np.abs(t_values.to_numpy())), 1)
    fig, ax = plt.subplots(figsize=(1.8 * t_values.shape[1] + 3, 0.45 * t_values.shape[0] + 1.8))
    image = ax.imshow(t_values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(t_values.shape[1]), [subtype_titles[k - 1] for k in t_values.columns])
    ax.set_yticks(range(t_values.shape[0]), t_values.index)
    for (row, col), t in np.ndenumerate(t_values.to_numpy()):
        star = "*" if p_fdr.iloc[row, col] < 0.05 else ""
        ax.text(col, row, f"{t:.1f}{star}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="t (this subtype vs other tau+, adjusted for stage)")
    ax.set_title("Where does each subtype differ? (* = FDR p < 0.05)")
    fig.tight_layout()
    return fig


def plot_stage_vs(table, values, label, subtype_titles, colors=None):
    """Paper Fig. 3a/b: SuStaIn stage against a number (age, MMSE, ...) for tau+ people,
    coloured by subtype, with one straight line per subtype."""
    numeric = pd.to_numeric(values, errors="coerce")
    colors = subtype_colors(len(subtype_titles), colors)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, title in enumerate(subtype_titles, start=1):
        people = (table["ml_subtype"] == k) & (table["ml_stage"] > 0) & numeric.notna()
        x, y = table.loc[people, "ml_stage"], numeric[people]
        ax.scatter(x, y, s=14, color=colors[k - 1], label=f"{title} (n={people.sum()})")
        if people.sum() > 2 and x.nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, intercept + slope * xs, color=colors[k - 1], lw=2)
    ax.set_xlabel("SuStaIn stage")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_asymmetry(table, asymmetry, subtype_titles, colors=None):
    """Paper Ext. Fig. 9: left-right asymmetry of tau per subtype (box plot) and against stage.
    Above 0 = more tau on the left, below 0 = more on the right."""
    colors = subtype_colors(len(subtype_titles), colors)
    tau_pos = table["ml_stage"] > 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    data = [asymmetry[tau_pos & (table["ml_subtype"] == k)] for k in range(1, len(subtype_titles) + 1)]
    boxes = ax1.boxplot(data, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
    ax1.set_xticks(range(1, len(subtype_titles) + 1), subtype_titles, rotation=30, ha="right")
    ax1.axhline(0, color="gray", lw=1)
    ax1.set_ylabel("asymmetry (left - right)")
    ax2.scatter(table.loc[tau_pos, "ml_stage"], asymmetry[tau_pos], s=12,
                c=colors[table.loc[tau_pos, "ml_subtype"].to_numpy() - 1])
    ax2.axhline(0, color="gray", lw=1)
    ax2.set_xlabel("SuStaIn stage")
    ax2.set_ylabel("asymmetry (left - right)")
    fig.tight_layout()
    return fig


def plot_cohort_subtypes(percent, subtype_titles, colors=None):
    """How often each subtype (and S0) is found in each cohort: one bar per cohort, split into the % of its
    people in S0 (grey) and in each subtype. percent: rows "S0", "Subtype 1", ...; columns = cohorts."""
    k = len(subtype_titles)
    shades = ["#D9E2EC"] + list(subtype_colors(k, colors))
    labels = ["S0 (no abnormal tau)"] + list(subtype_titles)
    fig, ax = plt.subplots(figsize=(1.3 * percent.shape[1] + 4, 4.2))
    bottom = np.zeros(percent.shape[1])
    for row, shade, label in zip(["S0"] + [f"Subtype {n}" for n in range(1, k + 1)], shades, labels):
        heights = percent.loc[row].to_numpy() if row in percent.index else np.zeros(percent.shape[1])
        ax.bar(range(percent.shape[1]), heights, bottom=bottom, color=shade, label=label, width=0.6)
        bottom += heights
    ax.set_xticks(range(percent.shape[1]), percent.columns)
    ax.set_ylabel("% of the cohort's people")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_followup(scans, subtype_titles, time_label, colors=None):
    """SuStaIn stage over time for the people with follow-up scans: one line per person from the first scan
    (time 0) through the later ones, coloured by the subtype at the first scan (grey = S0 at first).
    scans: one row per scan with columns person, time, stage, first_subtype (0 = S0)."""
    k = len(subtype_titles)
    shades = subtype_colors(k, colors)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, person in scans.sort_values("time").groupby("person"):
        first = int(person["first_subtype"].iloc[0])
        ax.plot(person["time"], person["stage"], "-o", ms=3, lw=1, alpha=0.6,
                color=shades[first - 1] if first > 0 else "#9AA5B1")
    handles = [plt.Line2D([], [], color=shades[n], lw=2) for n in range(k)] + [plt.Line2D([], [], color="#9AA5B1", lw=2)]
    ax.legend(handles, list(subtype_titles) + ["S0 at the first scan"], title="Subtype at the first scan",
              loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_xlabel(time_label)
    ax.set_ylabel("SuStaIn stage")
    fig.tight_layout()
    return fig


def plot_truth_confusion(counts, pairing, cmap="Blues"):
    """Simulation check: true subtype (rows) against the subtype SuStaIn found (columns), with the number
    of people and the % of the true subtype in each cell. A dark box marks each found subtype's partner
    (the true subtype it shares most people with); a perfect result has all people in those boxes."""
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(1.5 * counts.shape[1] + 3, 0.7 * counts.shape[0] + 2))
    image = ax.imshow(share, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (row, col), n in np.ndenumerate(counts.to_numpy()):
        ax.text(col, row, f"{n}\n{share.iloc[row, col]:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if share.iloc[row, col] > 60 else "#1F2933")
    partners = {"S0": "S0", **pairing}
    for col, found in enumerate(counts.columns):
        if partners.get(found) in counts.index:
            row = list(counts.index).index(partners[found])
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, lw=2.5, color="#1F2933"))
    labels = [f"{found}\n= {pairing[found]}" if found in pairing else found for found in counts.columns]
    ax.set_xticks(range(counts.shape[1]), labels)
    ax.set_yticks(range(counts.shape[0]), counts.index)
    ax.set_xlabel("found by SuStaIn (= its true partner)")
    ax.set_ylabel("true subtype")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(image, ax=ax, label="% of the true subtype", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_stage_truth(rows, pairing, n_subtypes, colors=None, seed=0):
    """Simulation check: found SuStaIn stage against the true stage (true tau+ people), one dot per person
    coloured like the found subtype that is the partner of their true subtype (grey = no partner).
    Dashed line = perfect agreement. Right: how far off the stage is, per true subtype."""
    shades = subtype_colors(n_subtypes, colors)
    colour_of = {true: shades[int(found.split()[-1]) - 1] for found, true in pairing.items()}
    tau = rows[(rows["true subtype"] != "S0") & rows["true stage"].notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    jitter = np.random.default_rng(seed).uniform(-0.25, 0.25, (2, len(tau)))     # many people share a stage
    names = sorted(tau["true subtype"].unique())
    for true in names:
        people = (tau["true subtype"] == true).to_numpy()
        ax1.scatter(tau["true stage"][people] + jitter[0, people], tau["found stage"][people] + jitter[1, people],
                    s=16, alpha=0.75, color=colour_of.get(true, "#9AA5B1"), label=f"{true} (n={people.sum()})")
    top = max(tau["true stage"].max(), tau["found stage"].max()) + 1 if len(tau) else 1
    ax1.plot([0, top], [0, top], "--", color="#52606D", lw=1)
    ax1.set_xlabel("true stage")
    ax1.set_ylabel("stage found by SuStaIn")
    ax1.legend(title="true subtype", fontsize=8)
    ax2.boxplot([tau.loc[tau["true subtype"] == t, "stage error"] for t in names], patch_artist=True,
                medianprops={"color": "white"}, boxprops={"facecolor": "#9AA5B1"})
    for patch, true in zip(ax2.patches, names):
        patch.set_facecolor(colour_of.get(true, "#9AA5B1"))
    ax2.axhline(0, ls="--", color="#52606D", lw=1)
    ax2.set_xticks(range(1, len(names) + 1), names, rotation=30, ha="right")
    ax2.set_ylabel("found - true stage (+ = found too late)")
    fig.tight_layout()
    return fig


# Diagnosis colours (Rosa's palette; SCD a deeper violet and MCI a deeper yellow so that all four stay
# apart for colour-blind readers - checked with validate_palette.js, all pairs pass)
DIAGNOSIS_COLORS = {"CN": "#11A0D9", "CU": "#11A0D9", "HC": "#11A0D9", "control": "#11A0D9",
                    "SCD": "#6250D6", "MCI": "#EDA100", "AD": "#F24405", "dementia": "#F24405"}
OTHER_COLORS = ["#3F8C61", "#8C2E62", "#332288", "#44AA99", "#DDCC77"]


def group_colors(groups):
    """One fixed color per diagnosis: blue for healthy, red for AD; other labels get the next free color."""
    colors = {}
    extra = 0
    for g in groups:
        if g in DIAGNOSIS_COLORS:
            colors[g] = DIAGNOSIS_COLORS[g]
        else:
            colors[g] = OTHER_COLORS[extra % len(OTHER_COLORS)]
            extra += 1
    return colors


def frame(ax):
    """A light border on all four sides, so the small panels of a figure are clearly apart."""
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color("#BCCCDC")


def plot_covariate_boxplots(data, numbers, group_col=None, groups=None, n_cols=4, colors=None):
    """One box plot per number column (MMSE, age, ...), one colored box per group, with every person as a dot.
    colors: {group: colour} (None = the diagnosis colours)."""
    n_rows = int(np.ceil(len(numbers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    labels = groups if group_col else ["All"]
    colors = colors or (group_colors(labels) if group_col else {"All": "#11A0D9"})
    rng = np.random.default_rng(0)                                 # same dot positions every time
    for ax, col in zip(axes.flat, numbers):
        values = pd.to_numeric(data[col], errors="coerce")
        for i, g in enumerate(labels, start=1):
            in_group = data[group_col].astype(str) == g if group_col else values.notna()
            group_values = values[in_group].dropna()
            if len(group_values) == 0:
                continue
            box = ax.boxplot([group_values], positions=[i], widths=0.6, patch_artist=True, showfliers=False)
            box["boxes"][0].set(facecolor=colors[g], alpha=0.35)
            box["medians"][0].set(color="black")
            x = i + rng.uniform(-0.2, 0.2, len(group_values))
            ax.scatter(x, group_values, s=6, color=colors[g], alpha=0.7)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_title(col, fontsize=10)
        frame(ax)
    for ax in axes.flat[len(numbers):]:
        ax.set_visible(False)
    if group_col:                                                  # one legend for all panels
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in labels]
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def plot_category_bars(data, categories, group_col=None, groups=None, n_cols=4, colors=None):
    """One bar chart per category column (sex, APOE4, ...): % of each value, one bar per group
    (same colors as the box plots, so no legend here)."""
    n_rows = int(np.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, categories):
        values = data[col].dropna()
        values = values.map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))
        if group_col:
            group = data.loc[values.index, group_col].astype(str)
            table = pd.crosstab(values, group, normalize="columns")[groups] * 100
            bar_colors = [(colors or group_colors(groups))[g] for g in groups]
        else:
            table = values.value_counts(normalize=True).to_frame("All") * 100
            bar_colors = [(colors or {}).get("All", "#11A0D9")]
        table.plot.bar(ax=ax, legend=False, rot=0, color=bar_colors)
        ax.set_title(col, fontsize=10)
        frame(ax)
        ax.set_xlabel("")
        ax.set_ylabel("%")
    for ax in axes.flat[len(categories):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_correlations(data, numbers, cmap="RdBu"):
    """Spearman correlation between all number columns, as a triangle of circles (like R's corrplot):
    bigger circle = stronger correlation; blue = rise together, red = one falls when the other rises."""
    values = data[numbers].apply(pd.to_numeric, errors="coerce")
    corr = values.corr(method="spearman").to_numpy()
    n = len(numbers)
    size = 0.42 * n + 2.5
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    cmap = plt.get_cmap(cmap)
    for row in range(n):
        for col in range(row, n):                         # upper triangle (and the diagonal)
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#D9E2EC", lw=0.8))
            r = corr[row, col]
            if np.isnan(r):
                continue
            ax.add_patch(plt.Circle((col, row), 0.45 * np.sqrt(abs(r)), color=cmap((r + 1) / 2)))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.set_xticks(range(n), numbers, rotation=45, ha="left")
    ax.set_yticks(range(n), numbers)
    ax.tick_params(length=0, colors="#1F2933")             # names in ink, not grey
    for side in ax.spines.values():
        side.set_visible(False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=cmap), ax=ax, shrink=0.8)
    colorbar.set_label("Spearman r")
    fig.tight_layout()
    return fig