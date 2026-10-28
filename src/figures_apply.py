"""Figures of the new app (applying a trained model). Same look as src/figures.py (tau_typer.mplstyle)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import left_right_layout, nice_region, style_colorbar, subtype_colors

S0_COLOR = "#BFC5CE"                      # S0 (no abnormal tau) - grey in every figure
NEW_COLOR, TRAINING_COLOR = "#0F5C6E", "#E8A33D"


def _panels(lobes, width=2.9, height=2.3):
    """Left lobes in one row, their right partners right below (shared x axis)."""
    cells, n_cols = left_right_layout(list(lobes))
    n_rows = max(1, int(np.ceil(len(cells) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False)
    return fig, axes, cells + [None] * (axes.size - len(cells))


def plot_new_vs_training(new_lobes, training_lobes, lobes, xlabel="lobe value (mean SUVR of its regions)"):
    """Jake's notebook cell 14, for every lobe: the distribution of your people (filled) and of the model's
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
                label=f"your data (n={len(new)})")
        ax.set_title(nice_region(lobe))
        ax.set_xlabel(xlabel, fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("% of people")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_z_heatmap(z, lobes, vmin=0, vmax=10, order=None):
    """Every person's z-scores (rows, sorted from least to most tau) per lobe (columns), as Jake's
    Plot_Probabilites view with vmin 2 ... vmax 10: dark = far above normal."""
    values = z[lobes]
    rows = values.mean(axis=1).sort_values().index if order is None else order
    fig, ax = plt.subplots(figsize=(0.55 * len(lobes) + 2.5, 6.5))
    image = ax.imshow(values.loc[rows].to_numpy(float), aspect="auto", cmap="magma_r", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    ax.set_xticks(range(len(lobes)), [nice_region(l) for l in lobes], rotation=60, ha="right")
    ax.set_yticks([])
    ax.set_ylabel(f"people (n = {len(rows)}), least tau at the top")
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.04)
    style_colorbar(bar, "z-score")
    fig.tight_layout()
    return fig


def plot_z_histograms(z, lobes, cut_points=(2, 5, 10)):
    """z-scores of each lobe; dashed lines = the model's cut-points (z 2, 5, 10)."""
    fig, axes, cells = _panels(lobes)
    for ax, lobe in zip(axes.flat, cells):
        if lobe is None:
            ax.set_visible(False)
            continue
        ax.hist(z[lobe].dropna(), bins=40, color=NEW_COLOR)
        for cut in cut_points:
            ax.axvline(cut, ls="--", lw=0.9, color="#C62828")
        ax.set_title(nice_region(lobe))
        ax.set_xlabel("z-score", fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("people")
    fig.tight_layout()
    return fig


def plot_lobe_boxplots(z, table, lobes, names, colors=None):
    """Jake's cell 69: z-score of every lobe per group (S0 grey, then each subtype in its colour)."""
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
        ax.set_title(nice_region(lobe))
        ax.set_xticks(range(1, len(groups) + 1), labels, rotation=40, ha="right", fontsize=7)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("z-score")
    fig.tight_layout()
    return fig


def plot_stage_histograms(table, names, n_events, colors=None):
    """Jake's cell 70: the stages of each subtype's people (one panel per subtype, same axes)."""
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
    ax.set_xticks(range(counts.shape[1]), counts.columns, rotation=15, ha="right")
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


def plot_cleanup(p, threshold):
    """P(abnormal) of each cleanup region for every person (dots), threshold dashed: a person is tau+ if
    at least one region is above it."""
    fig, ax = plt.subplots(figsize=(1.6 * p.shape[1] + 2.5, 3.8))
    rng = np.random.default_rng(0)
    for i, roi in enumerate(p.columns):
        ax.scatter(i + rng.uniform(-0.25, 0.25, len(p)), p[roi], s=8, alpha=0.5, color=NEW_COLOR)
    ax.axhline(threshold, ls="--", color="#C62828", lw=1.2, label=f"threshold {threshold:g}")
    ax.set_xticks(range(p.shape[1]), [c.replace("ctx-", "") for c in p.columns], rotation=20, ha="right")
    ax.set_ylabel("P(abnormal tau)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig
