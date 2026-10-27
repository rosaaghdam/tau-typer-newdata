"""App tab 5 - compare the subtypes (Vogel et al. 2021: Table 1, Fig. 1d, Fig. 3a/b, Ext. Fig. 9).

All functions take the per-person table from src/results.py subject_table() (subtypes numbered
by size, "S0" = stage 0) plus z-scores or other columns of the same people.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests

from src.qc import number_and_category_columns


def covariate_terms(covariates, names):
    """The chosen adjustment covariates (age, sex, education, ...) as a table + formula terms.
    Columns are renamed cov0, cov1, ... so any column name works in a formula; numbers are used
    as they are, categories as C(cov)."""
    numbers, _ = number_and_category_columns(covariates, names)
    data = pd.DataFrame(index=covariates.index)
    terms = []
    for i, name in enumerate(names):
        col = f"cov{i}"
        if name in numbers:
            data[col] = pd.to_numeric(covariates[name], errors="coerce")
            terms.append(col)
        else:
            data[col] = covariates[name]
            terms.append(f"C({col})")
    return data, terms


def as_categories(values):
    """Category labels as text, empty cells left out; whole numbers without ".0" (APOE4 1.0 -> "1")."""
    known = values.dropna()
    return known.map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))


def one_vs_others_p(values, groups, tau_pos, subtype, numeric):
    """p for one subtype vs all other tau+ people: Mann-Whitney (numbers) or chi-square (categories)."""
    this, others = tau_pos & (groups == subtype), tau_pos & (groups != subtype)
    if numeric:
        a, b = values[this].dropna(), values[others].dropna()
        return stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan
    labels = as_categories(values[tau_pos])
    counts = pd.crosstab(labels, this[labels.index])
    return stats.chi2_contingency(counts).pvalue if counts.shape[0] > 1 and counts.shape[1] == 2 else np.nan


def group_table(table, covariates, variables, adjust_for=()):
    """Table 1: each variable per group (S0, Subtype 1, ...) and tests between the subtypes (tau+ only).

    Numbers: mean (SD); p (subtypes) = Kruskal-Wallis; adjusted p = linear model
    "variable ~ subtype + stage + adjust_for" (e.g. age, sex, education - as in the paper).
    Categories: % for every value; p (subtypes) = chi-square.
    Like the paper, each subtype is also compared with all other tau+ people (Mann-Whitney or
    chi-square); * after a value = different from the other subtypes (FDR-corrected p < 0.05).
    """
    groups = table["subtype"]
    adjusted_name = "p (subtypes, adjusted for " + ", ".join(["stage"] + list(adjust_for)) + ")"
    subtypes = sorted(g for g in groups.unique() if g != "S0")
    order = (["S0"] if (groups == "S0").any() else []) + subtypes
    tau_pos = table["ml_stage"] > 0
    rows = [{"variable": "N", **{g: str(int((groups == g).sum())) for g in order}}]
    marks = []                                               # (row number, subtype, p one vs others)
    for var in variables:
        values = covariates[var]
        numeric_values = pd.to_numeric(values, errors="coerce")
        is_number = numeric_values.notna().mean() > 0.9 and values.nunique() > 2
        if is_number:
            row = {"variable": var, **{g: f"{numeric_values[groups == g].mean():.1f} ({numeric_values[groups == g].std():.1f})"
                                       for g in order}}
            samples = [numeric_values[(groups == g) & tau_pos].dropna() for g in subtypes]
            if len(samples) > 1:
                row["p (subtypes)"] = stats.kruskal(*samples).pvalue
                extra, terms = covariate_terms(covariates, [a for a in adjust_for if a != var])
                data = pd.DataFrame({"y": numeric_values, "subtype": groups, "stage": table["ml_stage"]})
                data = pd.concat([data, extra], axis=1)[tau_pos].dropna()
                fit = smf.ols(" + ".join(["y ~ C(subtype)", "stage"] + terms), data=data).fit()
                row[adjusted_name] = sm.stats.anova_lm(fit, typ=2).loc["C(subtype)", "PR(>F)"]   # p for subtype
            new_rows = [row]
        else:
            labels = as_categories(values)                        # empty cells are not counted
            counts = pd.crosstab(labels, groups[labels.index]).reindex(columns=order, fill_value=0)
            percent = 100 * counts / counts.sum().replace(0, np.nan)
            tau_labels = labels[tau_pos[labels.index]]
            subtype_counts = pd.crosstab(tau_labels, groups[tau_labels.index])
            p = (stats.chi2_contingency(subtype_counts).pvalue
                 if subtype_counts.shape[0] > 1 and subtype_counts.shape[1] > 1 else np.nan)
            new_rows = [{"variable": f"{var} = {level}, %", **{g: f"{percent.loc[level, g]:.1f}" for g in order}}
                        for level in counts.index]
            new_rows[0]["p (subtypes)"] = p
        if len(subtypes) > 1:
            for g in subtypes:
                values_used = numeric_values if is_number else values
                marks.append((len(rows), g, one_vs_others_p(values_used, groups, tau_pos, g, is_number)))
        rows += new_rows
    result = pd.DataFrame(rows)
    if marks:
        p_values = np.array([p for _, _, p in marks], dtype=float)
        ok = ~np.isnan(p_values)
        corrected = np.full(len(p_values), np.nan)
        if ok.any():
            corrected[ok] = multipletests(p_values[ok], method="fdr_bh")[1]
        for (row, g, _), q in zip(marks, corrected):
            if q < 0.05:
                result.loc[row, g] += " *"
    for col in ["p (subtypes)", adjusted_name]:
        if col in result:
            result[col] = result[col].map(lambda p: "" if pd.isna(p) else f"{p:.3g}")
    return result


def regional_contrasts(z, table, regions, covariates=None, adjust_for=()):
    """Paper Fig. 1d: where does each subtype have more (or less) tau than all other tau+ people?

    Per subtype and region a linear model "z ~ this_subtype + stage + adjust_for" (tau+ people only),
    so the comparison is at the same disease stage (and age, sex, ... if chosen). Returns t-values and FDR-corrected p-values
    (Benjamini-Hochberg over all subtypes x regions), each as a regions x subtypes table.
    """
    tau_pos = table["ml_stage"] > 0
    subtypes = sorted(table.loc[tau_pos, "ml_subtype"].unique())
    extra, terms = covariate_terms(covariates, list(adjust_for)) if adjust_for else (pd.DataFrame(index=z.index), [])
    formula = " + ".join(["z ~ this", "stage"] + terms)
    t_values = pd.DataFrame(index=regions, columns=subtypes, dtype=float)
    p_values = t_values.copy()
    for k in subtypes:
        for region in regions:
            data = pd.DataFrame({"z": z.loc[tau_pos, region], "this": (table.loc[tau_pos, "ml_subtype"] == k).astype(int),
                                 "stage": table.loc[tau_pos, "ml_stage"]})
            data = pd.concat([data, extra.loc[data.index]], axis=1).dropna()
            fit = smf.ols(formula, data=data).fit()
            t_values.loc[region, k], p_values.loc[region, k] = fit.tvalues["this"], fit.pvalues["this"]
    corrected = multipletests(p_values.to_numpy().ravel(), method="fdr_bh")[1].reshape(p_values.shape)
    return t_values, pd.DataFrame(corrected, index=regions, columns=subtypes)


def stage_association(table, values):
    """Paper Fig. 3a/b: Spearman correlation between SuStaIn stage and a number (e.g. age, MMSE),
    tau+ people only - overall and per subtype. Returns a table: group, N, rho, p."""
    numeric = pd.to_numeric(values, errors="coerce")
    tau_pos = (table["ml_stage"] > 0) & numeric.notna()
    rows = []
    for name, people in [("All tau+", tau_pos)] + [(f"Subtype {k}", tau_pos & (table["ml_subtype"] == k))
                                                    for k in sorted(table.loc[tau_pos, "ml_subtype"].unique())]:
        if people.sum() > 2:
            rho, p = stats.spearmanr(table.loc[people, "ml_stage"], numeric[people])
            rows.append({"group": name, "N": int(people.sum()), "rho": round(rho, 3), "p": f"{p:.3g}"})
    return pd.DataFrame(rows)


SIDES = [("L_", "R_", "start"), ("Left-", "Right-", "start"), ("Left_", "Right_", "start"),
         ("lh_", "rh_", "start"), ("_L", "_R", "end"), ("_lh", "_rh", "end")]


def region_pairs(regions):
    """Left/right pairs from names like L_MTL / R_MTL, Left-X / Right-X, lh_X / rh_X, X_L / X_R, X_lh / X_rh.
    Returns {name: (left region, right region)}."""
    pairs = {}
    for region in regions:
        for left, right, where in SIDES:
            if where == "start" and region.startswith(left):
                name = region[len(left):]
                partner = right + name
            elif where == "end" and region.endswith(left):
                name = region[:-len(left)]
                partner = name + right
            else:
                continue
            if partner in regions:
                pairs[name] = (region, partner)
    return pairs


def asymmetry_index(z, regions):
    """Paper Ext. Fig. 9: left-right asymmetry per person = mean over region pairs of
    (left - right) / (|left| + |right|). Positive = more tau on the left. None if no pairs found."""
    pairs = region_pairs(regions)
    if not pairs:
        return None
    parts = [(z[left] - z[right]) / (z[left].abs() + z[right].abs() + 1e-6) for left, right in pairs.values()]
    return pd.concat(parts, axis=1).mean(axis=1).rename("asymmetry")


def compare_runs(table_a, table_b):
    """Agreement between two runs on the same people (paper Fig. 2b idea / sensitivity check):
    cross-table of the groups, adjusted Rand index of the subtypes (tau+ in both runs) and the
    correlation of the stages."""
    common = table_a.index.intersection(table_b.index)
    a, b = table_a.loc[common], table_b.loc[common]
    crosstab = pd.crosstab(a["subtype"], b["subtype"], rownames=["run A"], colnames=["run B"])
    both = (a["ml_stage"] > 0) & (b["ml_stage"] > 0)
    ari = adjusted_rand_score(a.loc[both, "ml_subtype"], b.loc[both, "ml_subtype"]) if both.sum() > 1 else np.nan
    rho = stats.spearmanr(a["ml_stage"], b["ml_stage"])[0] if len(common) > 2 else np.nan
    return crosstab, {"people in both runs": len(common), "tau+ in both": int(both.sum()),
                      "adjusted Rand index (subtypes)": round(ari, 3), "stage correlation (Spearman)": round(rho, 3)}


def subtypes_per_cohort(table, cohorts):
    """(% of each cohort's people in S0 and in each subtype, counts, p). p: chi-square test of whether the
    subtypes (tau+ people only) are equally common in every cohort (NaN if it cannot be tested)."""
    label = table["subtype"]
    counts = pd.crosstab(label, cohorts.reindex(table.index))
    percent = (100 * counts / counts.sum()).round(1)
    tau = counts.drop(index="S0", errors="ignore")
    tau = tau.loc[tau.sum(axis=1) > 0, tau.sum() > 0]
    p = stats.chi2_contingency(tau)[1] if tau.shape[0] > 1 and tau.shape[1] > 1 else np.nan
    return percent, counts, p


def followup_changes(first, later, times):
    """Each follow-up scan next to the person's first scan. first: subject table of the first scans (the
    model's people); later: subject table of the follow-up scans (index = person, may repeat); times: years
    or visits since the first scan, same rows as later. Returns one row per follow-up scan."""
    rows = pd.DataFrame({"person": later.index, "time": np.asarray(times, dtype=float),
                         "first subtype": first.loc[later.index, "subtype"].to_numpy(),
                         "first stage": first.loc[later.index, "ml_stage"].to_numpy(),
                         "subtype": later["subtype"].to_numpy(), "stage": later["ml_stage"].to_numpy(),
                         "subtype probability": later["prob_subtype"].round(2).to_numpy()})
    rows["stage change"] = rows["stage"] - rows["first stage"]
    return rows


def followup_summary(changes):
    """Plain sentences: does the stage rise, and do people keep their subtype? (last scan of each person)"""
    last = changes.sort_values("time").groupby("person").tail(1)
    lines = [f"{changes['person'].nunique()} people have {len(changes)} follow-up scans."]
    both = last[(last["first subtype"] != "S0") & (last["subtype"] != "S0")]
    if len(both):
        same = (both["first subtype"] == both["subtype"]).mean()
        lines.append(f"Of the {len(both)} people tau+ at both their first and last scan, {100 * same:.0f}% keep "
                     "their subtype.")
    up, down = (last["stage change"] > 0).mean(), (last["stage change"] < 0).mean()
    lines.append(f"Stage at the last scan: higher for {100 * up:.0f}%, the same for {100 * (1 - up - down):.0f}%, "
                 f"lower for {100 * down:.0f}% (a few lower stages are expected from noise).")
    became = ((last["first subtype"] == "S0") & (last["subtype"] != "S0")).sum()
    if (last["first subtype"] == "S0").any():
        lines.append(f"{became} of {(last['first subtype'] == 'S0').sum()} people at S0 first have abnormal tau later.")
    return lines
