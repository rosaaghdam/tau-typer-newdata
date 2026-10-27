"""EDA report (app tab "EDA report"): an automatic, statistical look at the uploaded data before any
modelling - what is in it, what is missing, how each variable is distributed, how the groups and sites
differ and what goes together. Each part has a table and/or figure and a few short conclusions.

eda_sections() returns the parts as plain data, so the app page and the PDF show exactly the same.
"""
import io
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.figures import (plot_category_bars, plot_correlations, plot_covariate_boxplots, plot_missing,
                         plot_region_eda, plot_region_histograms)
from src.qc import number_and_category_columns
from src.subtype_analysis import region_pairs


def p_text(p):
    """p-value as text: "p < 0.001" or "p = 0.03"."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def size_word(value, small, medium, large):
    """Effect size in words."""
    return "large" if value >= large else "medium" if value >= medium else "small" if value >= small else "negligible"


# ---------- the parts of the report
def overview(data, roles, regions):
    numbers, categories = number_and_category_columns(data, [c for c in data.columns if c not in regions
                                                               and c != roles.get("id")])
    missing = data.isna()
    duplicate_ids = int(data[roles["id"]].duplicated().sum()) if roles.get("id") else 0
    table = pd.DataFrame({"value": [len(data), data.shape[1], len(regions), len(numbers), len(categories),
                                    f"{100 * missing.values.mean():.1f}%", int((~missing.any(axis=1)).sum()),
                                    int(data.duplicated().sum()), duplicate_ids]},
                         index=["rows", "columns", "region columns", "other number columns", "category columns",
                                "missing values", "complete rows", "duplicate rows", "duplicate IDs"])
    findings = [f"{len(data)} rows and {data.shape[1]} columns: {len(regions)} tau regions, {len(numbers)} other "
                f"number and {len(categories)} category columns.",
                f"{100 * missing.values.mean():.1f}% of all values are missing; {int((~missing.any(axis=1)).sum())} "
                f"rows ({100 * (~missing.any(axis=1)).mean():.0f}%) are complete."]
    if int(data.duplicated().sum()) or duplicate_ids:
        findings.append(f"Check: {int(data.duplicated().sum())} duplicate rows, {duplicate_ids} duplicate IDs.")
    else:
        findings.append("No duplicate rows or IDs.")
    return {"title": "Overview", "findings": findings, "tables": [("The data in numbers", table)], "figures": []}


def missing_values(data):
    missing = data.isna()
    per_column = 100 * missing.mean()
    findings = []
    if per_column.max() == 0:
        findings.append("No value is missing.")
    else:
        worst = per_column.idxmax()
        findings.append(f"Most missing: {worst} ({per_column.max():.0f}%); {int((per_column > 0).sum())} of "
                        f"{data.shape[1]} columns have missing values, {int((per_column > 20).sum())} more than 20%.")
        cols = per_column.index[per_column > 0]
        best, pair = 0, None                              # columns that are missing in the same rows
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                both, either = (missing[a] & missing[b]).sum(), (missing[a] | missing[b]).sum()
                if either and both / either > best:
                    best, pair = both / either, (a, b)
        if pair and best >= 0.3:
            findings.append(f"{pair[0]} and {pair[1]} are often missing together ({100 * best:.0f}% overlap): "
                            "probably not by chance - check how these values were collected.")
        findings.append("Complete-case analyses use only the complete rows of the columns they need; with much "
                        "missing data, compare with an analysis that keeps everyone.")
    return {"title": "Missing values", "findings": findings, "tables": [],
            "figures": [("Missing values per column and row", lambda: plot_missing(data))]}


def number_variables(data, columns):
    if not columns:
        return None
    rows = {}
    for col in columns:
        values = pd.to_numeric(data[col], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        fence = 1.5 * (q3 - q1)
        normal_p = stats.shapiro(values.sample(min(len(values), 5000), random_state=0))[1] if len(values) > 2 else np.nan
        rows[col] = {"n": len(values), "missing %": round(100 * data[col].isna().mean(), 1),
                     "mean": values.mean(), "SD": values.std(), "median": values.median(), "Q1": q1, "Q3": q3,
                     "min": values.min(), "max": values.max(), "skew": stats.skew(values),
                     "outliers (1.5 IQR)": int(((values < q1 - fence) | (values > q3 + fence)).sum()),
                     "normal (Shapiro p)": normal_p}
    table = pd.DataFrame(rows).T
    skewed = [c for c in columns if abs(table.loc[c, "skew"]) > 1]
    not_normal = [c for c in columns if table.loc[c, "normal (Shapiro p)"] < 0.05]
    outliers = table["outliers (1.5 IQR)"].astype(int)
    findings = [f"{len(not_normal)} of {len(columns)} number columns are not normally distributed (Shapiro p < 0.05)"
                + (": report median (IQR) and use rank-based tests (as this app does)." if not_normal else "."),
                ("Strongly skewed (|skew| > 1): " + ", ".join(skewed) + "." if skewed else
                 "No number column is strongly skewed.")]
    if outliers.max() > 0:
        findings.append(f"Most outlying values (beyond 1.5 IQR): {outliers.idxmax()} ({outliers.max()}); check them "
                        "for entry errors before excluding anyone.")
    return {"title": "Number variables", "findings": findings,
            "tables": [("Summary of each number column", table.round(3))],
            "figures": [("Each number column (box = middle half, dots = people)",
                         lambda: plot_covariate_boxplots(data, columns))]}


def category_variables(data, columns):
    if not columns:
        return None
    rows, findings = {}, []
    for col in columns:
        shares = data[col].dropna().astype(str).value_counts(normalize=True)
        rows[col] = {"n": int(data[col].notna().sum()), "missing %": round(100 * data[col].isna().mean(), 1),
                     "values": len(shares), "most common": f"{shares.index[0]} ({100 * shares.iloc[0]:.0f}%)",
                     "all values (%)": ", ".join(f"{v} {100 * s:.0f}" for v, s in shares.head(6).items())}
        if shares.iloc[0] > 0.8:
            findings.append(f"{col} is unbalanced: {100 * shares.iloc[0]:.0f}% are {shares.index[0]}.")
        rare = shares[shares < 0.05]
        if len(rare):
            findings.append(f"{col} has rare values (< 5%): {', '.join(rare.index[:4])} - small groups give unstable "
                            "estimates.")
    if not findings:
        findings.append("Every category column is reasonably balanced.")
    return {"title": "Category variables", "findings": findings,
            "tables": [("Summary of each category column", pd.DataFrame(rows).T)],
            "figures": [("% of each value", lambda: plot_category_bars(data, columns))]}


def regional_values(data, regions):
    values = data[regions].apply(pd.to_numeric, errors="coerce")
    table = pd.DataFrame({"mean": values.mean(), "SD": values.std(), "median": values.median(), "min": values.min(),
                          "max": values.max(), "skew": values.apply(lambda v: stats.skew(v.dropna()))}).round(3)
    findings = [f"Highest average value: {table['mean'].idxmax()} ({table['mean'].max():.2f}); lowest: "
                f"{table['mean'].idxmin()} ({table['mean'].min():.2f}).",
                f"Most right-skewed: {table['skew'].idxmax()} (skew {table['skew'].max():.1f}) - a long tail of high "
                "values is the abnormal-tau group SuStaIn models."]
    pairs = region_pairs(regions)
    if pairs:
        rhos = {f"{left} / {right}": stats.spearmanr(values[left], values[right], nan_policy="omit")[0]
                for left, right in pairs.values()}
        weakest = min(rhos, key=rhos.get)
        findings.append(f"Left and right agree well (Spearman rho {min(rhos.values()):.2f} to {max(rhos.values()):.2f}); "
                        f"least in {weakest}.")
    return {"title": "Regional tau values", "findings": findings, "tables": [("Summary of each region", table)],
            "figures": [("Distribution of each region", lambda: plot_region_histograms(data, regions))]}


def group_differences(data, columns, group_col, groups):
    if not group_col or not columns:
        return None
    rows = {}
    for col in columns:
        numbers, _ = number_and_category_columns(data, [col])
        keep = data[[col, group_col]].dropna()
        if numbers:
            samples = [pd.to_numeric(keep.loc[keep[group_col].astype(str) == g, col], errors="coerce").dropna()
                       for g in groups]
            samples = [s for s in samples if len(s)]
            if len(samples) < 2:
                continue
            h, p = stats.kruskal(*samples)
            n, k = sum(len(s) for s in samples), len(samples)
            effect = max((h - k + 1) / (n - k), 0)
            rows[col] = {"test": "Kruskal-Wallis", "statistic": h, "p": p, "effect size": effect,
                         "effect": size_word(effect, 0.01, 0.06, 0.14), "effect measure": "eta squared (H)"}
        else:
            counts = pd.crosstab(keep[col].astype(str), keep[group_col].astype(str))
            if counts.shape[0] < 2 or counts.shape[1] < 2:
                continue
            chi2, p, _, _ = stats.chi2_contingency(counts)
            v = np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1)))
            rows[col] = {"test": "chi-square", "statistic": chi2, "p": p, "effect size": v,
                         "effect": size_word(v, 0.1, 0.3, 0.5), "effect measure": "Cramer's V"}
    if not rows:
        return None
    table = pd.DataFrame(rows).T
    table["p (FDR)"] = multipletests(table["p"].astype(float), method="fdr_bh")[1]
    table = table[["test", "statistic", "p", "p (FDR)", "effect size", "effect measure", "effect"]]
    significant = table[table["p (FDR)"] < 0.05].sort_values("effect size", ascending=False)
    findings = [f"{len(significant)} of {len(table)} variables differ between the {group_col} groups "
                f"({', '.join(groups)}) after FDR correction (q < 0.05)."]
    if len(significant):
        findings.append("Largest differences: " + "; ".join(
            f"{v} ({significant.loc[v, 'effect']}, {p_text(significant.loc[v, 'p'])})" for v in significant.index[:5]) + ".")
    calm = table[table["p (FDR)"] >= 0.05].index
    if len(calm):
        findings.append("No clear difference in: " + ", ".join(calm[:8]) + ".")
    numbers, _ = number_and_category_columns(data, list(significant.index[:8]))
    figures = [("The most different number columns by group",
                lambda: plot_covariate_boxplots(data, numbers, group_col, groups))] if numbers else []
    shown = table.copy()
    for col in ["statistic", "effect size"]:
        shown[col] = shown[col].astype(float).round(3)
    for col in ["p", "p (FDR)"]:
        shown[col] = shown[col].astype(float).map(lambda v: f"{v:.2g}")
    return {"title": f"Differences between {group_col} groups", "findings": findings,
            "tables": [(f"Test per variable ({group_col})", shown)], "figures": figures}


def correlations(data, numbers, regions):
    data = data.copy()
    if regions:
        data["mean of regions"] = data[regions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        numbers = numbers + ["mean of regions"]
    if len(numbers) < 2:
        return None
    values = data[numbers].apply(pd.to_numeric, errors="coerce")
    rho = values.corr(method="spearman")
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(numbers) for b in numbers[i + 1:]]
    top = sorted(pairs, key=lambda p: -abs(p[2]))[:10]
    table = pd.DataFrame([{"variable 1": a, "variable 2": b, "Spearman rho": round(r, 2),
                           "n": int((values[a].notna() & values[b].notna()).sum())} for a, b, r in top])
    strong = [p for p in pairs if abs(p[2]) >= 0.7]
    findings = [f"{len(strong)} pairs are strongly related (|rho| >= 0.7)" + (
        ": " + "; ".join(f"{a} - {b} ({r:+.2f})" for a, b, r in strong[:5]) + "." if strong else ".")]
    if regions:
        with_tau = rho["mean of regions"].drop("mean of regions").sort_values(key=abs, ascending=False)
        findings.append(f"Most related to the mean tau of the regions: {with_tau.index[0]} ({with_tau.iloc[0]:+.2f})"
                        + (f", {with_tau.index[1]} ({with_tau.iloc[1]:+.2f})" if len(with_tau) > 1 else "") + ".")
    findings.append("Strongly related variables carry almost the same information: adjusting for both at once "
                    "makes estimates unstable.")
    return {"title": "Correlations", "findings": findings,
            "tables": [("The 10 strongest pairs", table)],
            "figures": [("Spearman correlation between the number columns",
                         lambda: plot_correlations(data, numbers))]}


def site_differences(data, regions, cohort_col):
    if not cohort_col or data[cohort_col].nunique() < 2 or not regions:
        return None
    rows = {}
    for region in regions:
        samples = [pd.to_numeric(g[region], errors="coerce").dropna() for _, g in data.groupby(cohort_col)]
        h, p = stats.kruskal(*samples)
        rows[region] = {"statistic": h, "p": p,
                        **{f"median {site}": pd.to_numeric(g[region], errors="coerce").median()
                           for site, g in data.groupby(cohort_col)}}
    table = pd.DataFrame(rows).T
    table["p (FDR)"] = multipletests(table["p"], method="fdr_bh")[1]
    different = table.index[table["p (FDR)"] < 0.05]
    findings = [f"{len(different)} of {len(regions)} regions differ between the {data[cohort_col].nunique()} "
                f"{cohort_col} groups (Kruskal-Wallis, FDR q < 0.05)" + (": " + ", ".join(different) + "." if len(different) else "."),
                "Differences between sites can come from scanners and processing, not only from the people: the "
                "app z-scores each site against its own normal (tab 2) for this reason."
                if len(different) else "The sites look alike in their regional values."]
    shown = table.round(3)
    shown["p"] = table["p"].map(lambda v: f"{v:.2g}")
    shown["p (FDR)"] = table["p (FDR)"].map(lambda v: f"{v:.2g}")
    return {"title": f"Differences between {cohort_col} groups", "findings": findings,
            "tables": [(f"Regional values per {cohort_col}", shown)],
            "figures": [(f"Regional values per {cohort_col}", lambda: plot_region_eda(data, regions, x=cohort_col))]}


def eda_sections(data, roles, regions, group_order=None):
    """Every part of the report, in order: [{"title", "findings": [text], "tables": [(caption, table)],
    "figures": [(caption, function that draws the figure)]}]. roles: the tab-1 column choices."""
    skip = set(regions) | {roles.get("id"), roles.get("visit")}
    others = [c for c in data.columns if c not in skip]
    numbers, categories = number_and_category_columns(data, others)
    group_col = roles.get("diagnosis")
    groups = group_order or (sorted(data[group_col].dropna().astype(str).unique()) if group_col else None)
    parts = [overview(data, roles, regions), missing_values(data), number_variables(data, numbers),
             category_variables(data, categories), regional_values(data, regions) if regions else None,
             group_differences(data, [c for c in numbers + categories if c != group_col], group_col, groups),
             correlations(data, numbers, regions), site_differences(data, regions, roles.get("cohort"))]
    return [part for part in parts if part is not None]


# ---------- the PDF
def build_eda_pdf(sections, title):
    """The report as a PDF (bytes): per part a page with its conclusions and tables, then its figures."""
    from matplotlib.backends.backend_pdf import PdfPages
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for number, part in enumerate(sections, start=1):
            page = plt.figure(figsize=(8.27, 11.69))
            page.text(0.07, 0.955, title if number == 1 else "", fontsize=13, fontweight="bold", va="top")
            page.text(0.07, 0.92, f"{number}. {part['title']}", fontsize=12, fontweight="bold", va="top")
            lines = []
            for finding in part["findings"]:
                lines += textwrap.wrap("- " + finding, 100, subsequent_indent="  ")
            page.text(0.07, 0.89, "\n".join(lines), fontsize=8.5, va="top", linespacing=1.5)
            top = 0.89 - 0.016 * (len(lines) + 2)
            for caption, table in part["tables"]:
                shown = table.reset_index(names=table.index.name or "").astype(str).iloc[:30]
                height = min(0.025 * (len(shown) + 1), top - 0.05)
                ax = page.add_axes([0.07, top - height - 0.03, 0.86, height])
                ax.axis("off")
                ax.set_title(caption, fontsize=9, loc="left")
                cells = ax.table(cellText=shown.values, colLabels=list(shown.columns), loc="upper left",
                                 cellLoc="left")
                cells.auto_set_font_size(False)
                cells.set_fontsize(6)
                top -= height + 0.07
            pdf.savefig(page)
            plt.close(page)
            for caption, make in part["figures"]:
                fig = make()
                fig.suptitle(f"{number}. {part['title']}: {caption}", fontsize=10, y=1.03)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return buffer.getvalue()
