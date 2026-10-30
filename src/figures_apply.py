"""Figures of the new app (applying a trained model). Same look as src/figures.py (tau_typer.mplstyle)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from src.figures import left_right_layout, nice_region, style_colorbar, subtype_colors
from src.roi_names import region_key

S0_COLOR = "#BFC5CE"                      # S0 (no abnormal tau) - grey in every figure
NEW_COLOR, TRAINING_COLOR = "#0F5C6E", "#E8A33D"


def _panels(lobes, width=2.9, height=2.3):
    """Left lobes in one row, their right partners right below (shared x axis)."""
    cells, n_cols = left_right_layout(list(lobes))
    n_rows = max(1, int(np.ceil(len(cells) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False)
    return fig, axes, cells + [None] * (axes.size - len(cells))


def _test_vs_training_hist(ax, new, old, title, xlabel):
    """One panel: the test data (filled) and the training cohort (outline), as % of each group."""
    bins = np.histogram_bin_edges(pd.concat([new, old]), bins=40)
    ax.hist(old, bins=bins, weights=np.full(len(old), 100 / max(len(old), 1)), histtype="step", lw=1.4,
            color=TRAINING_COLOR, label=f"training (n={len(old)})")
    ax.hist(new, bins=bins, weights=np.full(len(new), 100 / max(len(new), 1)), color=NEW_COLOR, alpha=0.55,
            label=f"test data (n={len(new)})")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)


def region_pairs(regions):
    """The regions as (right, left) pairs in the given order, e.g. (ctx-rh-entorhinal, ctx-lh-entorhinal);
    a region without a partner is paired with None."""
    by_key = {region_key(r): r for r in regions}
    pairs, seen = [], set()
    for region in regions:
        side, name = region_key(region)
        if region in seen:
            continue
        partner = by_key.get(("R" if side == "L" else "L", name)) if side else None
        right, left = (partner, region) if side == "L" else (region, partner)
        pairs.append((right, left))
        seen.update({right, left})
    return pairs


def plot_region_pairs(new, training, chosen, all_regions, per_row=6, xlabel="regional SUVR"):
    """Histograms of the chosen regions, test data vs the training cohort: per_row region pairs per row, the RIGHT
    region above its LEFT partner (the same column); a cell stays empty when that side is not chosen."""
    pairs = [pair for pair in region_pairs(all_regions) if set(pair) & set(chosen)]
    n_cols = min(per_row, len(pairs))
    n_rows = 2 * int(np.ceil(len(pairs) / per_row))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.9 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, pair in enumerate(pairs):
        for row_shift, region in enumerate(pair):             # 0 = right (above), 1 = left (below)
            if region is None or region not in chosen:
                continue
            ax = axes[2 * (i // per_row) + row_shift, i % per_row]
            ax.set_visible(True)
            _test_vs_training_hist(ax, new[region].dropna(), training[region].dropna(), region, xlabel)
            if i % per_row == 0:
                ax.set_ylabel("% of people", fontsize=7)
            ax.tick_params(labelsize=6)
    first = next(ax for ax in axes.flat if ax.get_visible())
    first.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    return fig


def plot_new_vs_training(new_lobes, training_lobes, lobes, xlabel="lobe value (mean SUVR of its regions)"):
    """For every lobe: the distribution of your people (filled) and of the model's
    training cohort (outline), as % of each group - do your values sit on the same scale?"""
    fig, axes, cells = _panels(lobes)
    for ax, lobe in zip(axes.flat, cells):
        if lobe is None:
            ax.set_visible(False)
            continue
        new, old = new_lobes[lobe].dropna(), training_lobes[lobe].dropna()
        bins = np.histogram_bin_edges(pd.concat([new, old]), bins=40)
        ax.hist(old, bins=bins, weights=np.full(len(old), 100 / max(len(old), 1)), histtype="step", lw=1.6,
                color=TRAINING_COLOR, label=f"training (n={len(old)})")
        ax.hist(new, bins=bins, weights=np.full(len(new), 100 / max(len(new), 1)), color=NEW_COLOR, alpha=0.55,
                label=f"test data (n={len(new)})")
        ax.set_title(nice_region(lobe))
        ax.set_xlabel(xlabel, fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("% of people")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


HEAT_LABEL = "Standardized regional score"
BLUES = ["#FFFFFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#08306B", "#000000"]   # white -> blue -> black
VALUE_CMAP = LinearSegmentedColormap.from_list("values", BLUES)        # e.g. SUVR: evenly from the low to the high limit


def score_cmap(limits):
    """Standardized scores: white up to 0 (normal), light blue at the model's first cut-point (z 2), blue at 5,
    dark blue to black at the top - so normal people stay white and blue means abnormal tau."""
    low, high = limits
    stops = [(low, "#FFFFFF"), (0, "#F7FBFF"), (2, "#9ECAE1"), (5, "#2171B5"), ((5 + high) / 2, "#08306B"),
             (high, "#000000")]
    return LinearSegmentedColormap.from_list("score", [((v - low) / (high - low), c) for v, c in stops
                                                       if low <= v <= high])


def heatmap_scale(limits):
    """(colour map, norm) of a heatmap with fixed limits, white -> blue -> black: standardized scores (a scale that
    reaches below 0) are anchored at 0, 2 and 5; other values (e.g. SUVR) change evenly from low to high."""
    if limits[0] < 0 < limits[1]:
        return score_cmap(limits), plt.Normalize(*limits)
    return VALUE_CMAP, plt.Normalize(*limits)


def heatmap_order(z, lobes):
    """The heatmap's row order: people sorted by their mean score over the lobes, lowest first (top)."""
    return z[lobes].mean(axis=1).sort_values().index


def heatmap_colors(values, limits):
    """Hex colour of every value on the fixed colour scale: values below the lower limit get the lowest
    colour, above the upper limit the highest (the same limits for every method, so colours compare fairly)."""
    cmap, norm = heatmap_scale(limits)
    values = np.clip(np.nan_to_num(np.asarray(values, float), nan=limits[0]), *limits)
    rgba = cmap(norm(values))
    return np.apply_along_axis(lambda c: "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c[:3]), -1, rgba)


def plot_z_heatmap(z, lobes, limits=(-3, 10), order=None):
    """Every person's scores (rows, sorted by mean score, lowest at the top) per lobe (columns), on a FIXED
    colour scale (limits): white (lowest) -> blue -> dark blue -> black (highest); beyond the limits = the end colours."""
    rows = heatmap_order(z, lobes) if order is None else order
    cmap, norm = heatmap_scale(limits)
    fig, ax = plt.subplots(figsize=(0.55 * len(lobes) + 2.5, 6.5))
    image = ax.imshow(z.loc[rows, lobes].to_numpy(float), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(lobes)), [nice_region(l) for l in lobes], rotation=90)
    ax.set_yticks([])
    ax.set_ylabel(f"people (n = {len(rows)}), sorted by mean score (lowest at the top)")
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.04, extend="both")
    bar.set_ticks([t for t in (limits[0], 0, 2, 5, limits[1]) if limits[0] <= t <= limits[1]])
    style_colorbar(bar, HEAT_LABEL)
    fig.tight_layout()
    return fig


def plot_subtype_events(events, lobes, n_events, stage, colors=None, per_row=2):
    """The learned subtypes, one panel each: every lobe's events (circle = z 2, diamond = z 5, square = z 10) at
    the stage where the model puts them (mean over its MCMC samples). Events reached by the chosen stage are
    solid, later ones faint; the vertical line marks the chosen stage."""
    k = len(events)
    shades = subtype_colors(k, colors)
    markers = ["o", "D", "s", "^", "v", "P"]
    n_rows = int(np.ceil(k / per_row))
    fig, axes = plt.subplots(n_rows, per_row, figsize=(6.2 * per_row, 0.38 * len(lobes) * n_rows + 1.4 * n_rows),
                             squeeze=False)
    y = np.arange(len(lobes))
    for ax in axes.flat[k:]:
        ax.set_visible(False)
    for ax, (name, table), shade in zip(axes.flat, events.items(), shades):
        for row, lobe in enumerate(lobes):
            stages = table.loc[lobe].dropna()
            ax.plot(stages, [row] * len(stages), color=shade, lw=1, alpha=0.25)
            for (z, x), marker in zip(stages.items(), markers):
                ax.plot(x, row, marker, ms=8, color=shade, alpha=1 if x <= stage else 0.18)
        ax.axvline(stage, color="#1F2933", lw=1.2)
        ax.set_yticks(y, [nice_region(lobe) for lobe in lobes])
        ax.set_ylim(len(lobes) - 0.5, -0.5)
        ax.set_xlim(0, n_events + 0.5)
        ax.set_xlabel("stage")
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.set_title(name, loc="left", fontsize=12, fontweight="bold")
    levels = list(next(iter(events.values())).columns)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=8, mfc="none", mec="#1F2933") for m in markers[:len(levels)]]
    fig.legend(handles, [f"z = {z:g}" for z in levels], loc="upper left", ncol=len(levels), frameon=False,
               title=f"Stage {stage}: solid = event reached, faint = later event", title_fontsize=10,
               alignment="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    return fig


def plot_z_histograms(z, lobes, cut_points=(2, 5, 10), training_z=None):
    """z-scores of each lobe; dashed lines = the model's cut-points (z 2, 5, 10). With training_z: the training
    cohort's z-scores as an outline, both groups as % of their people (the groups differ in size)."""
    fig, axes, cells = _panels(lobes)
    for ax, lobe in zip(axes.flat, cells):
        if lobe is None:
            ax.set_visible(False)
            continue
        if training_z is None:
            ax.hist(z[lobe].dropna(), bins=40, color=NEW_COLOR)
            ax.set_title(nice_region(lobe))
        else:
            _test_vs_training_hist(ax, z[lobe].dropna(), training_z[lobe].dropna(), nice_region(lobe), "")
            ax.title.set_fontsize(plt.rcParams["axes.titlesize"])
        for cut in cut_points:
            ax.axvline(cut, ls="--", lw=0.9, color="#C62828")
        ax.set_xlabel("z-score", fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("people" if training_z is None else "% of people")
    if training_z is not None:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig
    fig.tight_layout()
    return fig


def boxplot_limits(z, lobes, setting="auto"):
    """The y range of the box plots, the same for every lobe: setting "auto" = the lowest and highest z-score of all
    lobes (plus a small margin), or a fixed [low, high] from config.yaml."""
    if setting != "auto":
        return tuple(setting)
    values = z[lobes].to_numpy(float)
    low, high = np.nanmin(values), np.nanmax(values)
    margin = 0.03 * (high - low)
    return low - margin, high + margin


def plot_lobe_boxplots(z, table, lobes, names, colors=None, limits=None):
    """z-score of every lobe per group (S0 grey, then each subtype in its colour). limits: the same y range for
    every lobe (e.g. (-5, 20)), so the lobes can be compared; None = each lobe its own range."""
    k = len(names)
    shades = [S0_COLOR] + list(subtype_colors(k, colors))
    groups = [0] + list(range(1, k + 1))
    labels = ["S0"] + list(names)
    fig, axes, cells = _panels(lobes, width=3.2, height=2.7)
    for ax, lobe in zip(axes.flat, cells):
        if lobe is None:
            ax.set_visible(False)
            continue
        data = [z.loc[table["subtype_number"] == g, lobe].dropna() for g in groups]
        parts = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops={"color": "#1F2933"})
        for patch, shade in zip(parts["boxes"], shades):
            patch.set_facecolor(shade)
            patch.set_edgecolor("#52606D")
        ax.axhline(2, ls="--", lw=0.8, color="#C62828")
        if limits is not None:
            ax.set_ylim(*limits)
        ax.set_title(nice_region(lobe))
        ax.set_xticks(range(1, len(groups) + 1), [label.replace(" ", "\n") for label in labels], fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("z-score")
    fig.tight_layout()
    return fig


def plot_stage_histograms(table, names, n_events, colors=None):
    """The stages of each subtype's people (one panel per subtype, same axes)."""
    k = len(names)
    shades = subtype_colors(k, colors)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 2.8), sharey=True, squeeze=False)
    for n, ax in enumerate(axes[0], start=1):
        stages = table.loc[table["subtype_number"] == n, "ml_stage"]
        ax.hist(stages, bins=np.arange(0.5, n_events + 1.5, 1), color=shades[n - 1])
        ax.set_title(f"{names[n - 1]}\nn = {len(stages)}")
        ax.set_xlabel("SuStaIn stage")
        ax.set_xlim(0.5, n_events + 0.5)
    axes[0, 0].set_ylabel("people")
    fig.tight_layout()
    return fig


def plot_group_counts(counts, colors=None):
    """People per group (S0 + subtypes, rows) for one or more results (columns), as % of each result."""
    share = 100 * counts / counts.sum()
    names = list(counts.index)
    shades = [S0_COLOR if name == "S0" else c for name, c in
              zip(names, [S0_COLOR] + list(subtype_colors(len(names) - 1, colors)))]
    fig, ax = plt.subplots(figsize=(1.6 * counts.shape[1] + 4, 4.2))
    bottom = np.zeros(counts.shape[1])
    for name, shade in zip(names, shades):
        heights = share.loc[name].to_numpy(float)
        ax.bar(range(counts.shape[1]), heights, bottom=bottom, color=shade, width=0.6, label=name)
        for x, (h, b, n) in enumerate(zip(heights, bottom, counts.loc[name])):
            if h >= 4:
                ax.text(x, b + h / 2, f"{n}", ha="center", va="center", fontsize=8,
                        color="#1F2933" if name == "S0" else "white")
        bottom += heights
    ax.set_xticks(range(counts.shape[1]), counts.columns)
    ax.set_ylabel("% of people (numbers = people)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_agreement(crosstab, row_label, col_label):
    """Two results on the same people: rows = groups of the first, columns = groups of the second.
    Numbers = people; colour = % of the row (dark diagonal = they agree)."""
    share = crosstab.div(crosstab.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(1.1 * crosstab.shape[1] + 3.2, 0.7 * crosstab.shape[0] + 2))
    image = ax.imshow(share.fillna(0).to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (row, col), n in np.ndenumerate(crosstab.to_numpy()):
        value = share.iloc[row, col]
        ax.text(col, row, f"{n}\n{0 if np.isnan(value) else value:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if value > 60 else "#1F2933")
    ax.set_xticks(range(crosstab.shape[1]), crosstab.columns, rotation=30, ha="right")
    ax.set_yticks(range(crosstab.shape[0]), crosstab.index)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    style_colorbar(bar, "% of the row")
    fig.tight_layout()
    return fig


def plot_stage_agreement(stage_a, stage_b, label_a, label_b, seed=0):
    """Stage of each person in two results (small jitter: many people share a stage). Dashed = equal."""
    jitter = np.random.default_rng(seed).uniform(-0.25, 0.25, (2, len(stage_a)))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(stage_a + jitter[0], stage_b + jitter[1], s=12, alpha=0.6, color=NEW_COLOR)
    top = max(stage_a.max(), stage_b.max()) + 1
    ax.plot([0, top], [0, top], "--", color="#52606D", lw=1)
    ax.set_xlabel(f"stage: {label_a}")
    ax.set_ylabel(f"stage: {label_b}")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_normal_values(tables, lobes):
    """The normal mean (dot) +- SD (line) of every lobe for each method - how far apart are the 'normals'?"""
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(len(tables), 1)
    for i, (name, table) in enumerate(tables.items()):
        x = np.arange(len(lobes)) + (i - (len(tables) - 1) / 2) * width
        means = table.loc[lobes, "normal mean"].astype(float)
        sds = table.loc[lobes, "normal SD"].astype(float)
        ax.errorbar(x, means, yerr=sds, fmt="o", ms=5, capsize=3, label=name)
    ax.set_xticks(range(len(lobes)), [nice_region(l) for l in lobes], rotation=40, ha="right")
    ax.set_ylabel("normal mean +- SD (SUVR)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


PROB_LABEL = "ROI high-signal probability"   # the probability of the higher bell curve (2-component GMM)
RETAINED, RESET = NEW_COLOR, TRAINING_COLOR


def short_region(name):
    """ctx-lh-entorhinal -> "L entorhinal" (other names unchanged)."""
    for code, side in (("-lh-", "L "), ("-rh-", "R ")):
        if code in name:
            return side + name.split(code, 1)[1]
    return name


def plot_cleanup_bars(summary):
    """False-positive check, one horizontal bar per initial subtype: retained vs reset to S0 (people and %).
    summary: from trained_model.cleanup_summary."""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(summary) + 1.4))
    y = np.arange(len(summary))
    ax.barh(y, summary["retained"], color=RETAINED, label="retained")
    ax.barh(y, summary["reset to S0"], left=summary["retained"], color=RESET, label="reset to S0")
    for row, (_, r) in enumerate(summary.iterrows()):
        if r["retained"]:
            ax.text(r["retained"] / 2, row, f"{int(r['retained'])} ({100 - r['% reset']:.0f}%)", ha="center",
                    va="center", color="white", fontsize=9)
        if r["reset to S0"]:
            ax.text(r["retained"] + r["reset to S0"] + 0.01 * summary["total"].max(), row,
                    f"{int(r['reset to S0'])} reset ({r['% reset']:.0f}%)", va="center", fontsize=9, color="#8A5A00")
    ax.set_yticks(y, summary.index)
    ax.invert_yaxis()
    ax.set_xlim(0, summary["total"].max() * 1.25)
    ax.set_xlabel("people the model gave this subtype")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_scan_check(person, p_row, threshold, before, after):
    """One scan: its four ROI high-signal probabilities (bars), the threshold (dashed), and its assignment before
    and after the check (title)."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(p_row) + 1.5))
    y = np.arange(len(p_row))
    ax.barh(y, p_row.to_numpy(float), color=[RETAINED if v > threshold else "#BFC5CE" for v in p_row], height=0.6)
    for row, value in enumerate(p_row):
        ax.text(min(value, 1) + 0.01, row, f"{value:.3f}", va="center", fontsize=9)
    ax.axvline(threshold, ls="--", color="#C62828", lw=1.2, label=f"threshold {threshold:g}")
    ax.set_yticks(y, [short_region(c) for c in p_row.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(PROB_LABEL)
    ax.set_title(f"{person}: {before}  ->  {after}", loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def region_order(training):
    """Region (column) order of the regional heatmaps, as ESM_utils.Plot_Probabilites: the training cohort's
    mean value per region, highest first. The same order for the test data and the training data."""
    return training.mean(axis=0).sort_values(ascending=False).index.tolist()


def region_limits(training, low=1, high=99):
    """Colour limits of the regional heatmaps: the training cohort's 1st and 99th percentile of all regional values,
    the same for both heatmaps so the colours compare fairly."""
    values = training.to_numpy(float).ravel()
    return tuple(float(np.round(np.nanpercentile(values, q), 2)) for q in (low, high))


def region_comparison(new, training):
    """Per region: the test data's and the training cohort's mean (SD), the difference, the standardized
    difference (Cohen's d = difference / pooled SD) with its size (|d| < 0.2 negligible, 0.2-0.5 small,
    0.5-0.8 medium, >= 0.8 large; Cohen 1988), and the p-value of a Welch t-test (unequal variances), also
    FDR-corrected over all regions (Benjamini-Hochberg). With many people even a tiny difference has a small p:
    p says whether a difference is real, d how big it is."""
    rows = []
    for region in region_order(training):
        a, b = new[region].dropna(), training[region].dropna()
        pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / pooled
        rows.append({"region": region, "test mean": a.mean(), "test SD": a.std(), "training mean": b.mean(),
                     "training SD": b.std(), "difference (test - training)": a.mean() - b.mean(), "Cohen's d": d,
                     "size of difference": "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else
                                           "small" if abs(d) >= 0.2 else "negligible",
                     "p (Welch t-test)": ttest_ind(a, b, equal_var=False).pvalue})
    table = pd.DataFrame(rows).set_index("region")
    table["p (FDR-corrected)"] = multipletests(table["p (Welch t-test)"], method="fdr_bh")[1]
    p_columns = ["p (Welch t-test)", "p (FDR-corrected)"]
    table[p_columns] = table[p_columns].map(lambda p: float(f"{p:.3g}"))
    return table.round({c: 3 for c in table.columns if c not in p_columns and c != "size of difference"})


def plot_region_comparison(table):
    """Every region: the test data (teal) and the training cohort (orange). Dot = the mean SUVR of all people,
    line = from mean - 1 SD to mean + 1 SD (where about two thirds of the people lie)."""
    regions = table.index.tolist()
    y = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(10, 0.2 * len(regions) + 1.4))
    for shift, prefix, color, label in ((-0.18, "test", NEW_COLOR, "test data"),
                                        (0.18, "training", TRAINING_COLOR, "training cohort")):
        ax.errorbar(table[f"{prefix} mean"], y + shift, xerr=table[f"{prefix} SD"], fmt="o", ms=4, lw=1,
                    color=color, label=label)
    ax.set_yticks(y, regions, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("regional SUVR")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_region_boxplots(new, training):
    """Every region (training mean highest on top) as two box plots: the test data (teal) and the training cohort
    (orange). Line in the box = median, box = middle 50% of the people (25th-75th percentile), whiskers = the
    rest up to 1.5 x the box length. Single extreme people are not drawn (the training values stay private)."""
    regions = region_order(training)
    y = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(regions) + 1.4))
    for shift, table, color, label in ((-0.2, new, NEW_COLOR, "test data"),
                                       (0.2, training, TRAINING_COLOR, "training cohort")):
        parts = ax.boxplot([table[r].dropna() for r in regions], positions=y + shift, widths=0.34, vert=False,
                           showfliers=False, patch_artist=True, manage_ticks=False)
        for box in parts["boxes"]:
            box.set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=0.8)
        for key in ("whiskers", "caps"):
            for line in parts[key]:
                line.set(color=color, linewidth=0.8)
        for median in parts["medians"]:
            median.set(color=color, linewidth=1.6)
        ax.plot([], [], color=color, linewidth=6, alpha=0.5, label=label)       # legend entry
    ax.set_yticks(y, regions, fontsize=8)
    ax.set_ylim(len(regions) - 0.5, -0.5)
    ax.set_xlabel("regional SUVR")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_region_heatmaps(new, training, limits):
    """Your people (left) and the training cohort (right): regions (columns, training mean highest first) x people
    (rows, sorted by their mean, lowest at the top), on the same colour scale - as ESM_utils.Plot_Probabilites."""
    order = region_order(training)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    for ax, table, title in ((axes[0], new, f"test data (n = {len(new)})"),
                             (axes[1], training, f"training cohort (n = {len(training)})")):
        rows = table[order].mean(axis=1).sort_values().index
        cmap, norm = heatmap_scale(limits)
        image = ax.imshow(table.loc[rows, order].to_numpy(float), aspect="auto", cmap=cmap, norm=norm,
                          interpolation="nearest")
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5)
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_ylabel("people, sorted by mean (lowest at the top)")
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, extend="both")
    style_colorbar(bar, "regional SUVR")
    return fig
