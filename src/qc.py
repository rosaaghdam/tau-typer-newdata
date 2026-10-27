"""Tab 1 - checks (QC) for an uploaded tau-PET table."""
import re

import numpy as np
import pandas as pd

# Words that often appear in each column's name (lower case), most specific first.
COLUMN_HINTS = {
    "id": ["subject_id", "subject", "rid", "id"],
    "age": ["age"],
    "sex": ["sex", "gender"],
    "diagnosis": ["diagnosis", "diag", "dx", "group"],
    "amyloid": ["amyloid", "abeta", "ab_status"],
    "cohort": ["cohort", "site", "study", "centre", "center"],
    "visit": ["visit", "scan_date", "date", "timepoint", "session", "viscode"],
}
# Region columns start with a hemisphere: L_, R_, lh_, rh_, left_, right_ (or - / . instead of _)
REGION_PATTERN = re.compile(r"^(l|r|lh|rh|left|right)[_\-.]", re.IGNORECASE)


def guess_columns(columns):
    """First guess of which column plays which role. The user checks it in the app.

    For each role: an exact name match first, otherwise the first column containing a hint.
    Returns a dict: role -> column name (None if nothing fits), and "regions" -> list of names.
    """
    by_lower_name = {col.lower(): col for col in columns}
    guesses = {}
    for role, hints in COLUMN_HINTS.items():
        exact = [by_lower_name[h] for h in hints if h in by_lower_name]
        partial = [col for h in hints for col in columns if h in col.lower()]
        guesses[role] = (exact + partial + [None])[0]
    guesses["regions"] = [col for col in columns if REGION_PATTERN.match(col)]
    return guesses


def three_groups(values):
    """A number column in 3 groups (low / middle / high), named by the values in each group, e.g. "18 to 26",
    "27 to 29", "30". The groups are about equal thirds of the people; when many people share a value (e.g.
    CDR = 0), equal thirds are impossible, so the different values are split into thirds instead.
    Returns (group of each person, missing stays missing; group names low to high)."""
    known = pd.to_numeric(values, errors="coerce").dropna()
    bins = pd.qcut(known, 3, duplicates="drop").cat.codes
    if bins.nunique() < 3 and known.nunique() >= 3:
        cuts = np.quantile(np.unique(known), [1 / 3, 2 / 3])      # thirds of the different values
        bins = pd.Series(np.searchsorted(cuts, known, side="left"), index=known.index)
    names, groups = [], pd.Series(np.nan, index=values.index, dtype=object)
    for code in sorted(bins.unique()):
        members = known[bins == code]
        low, high = members.min(), members.max()
        name = f"{low:.3g}" if low == high else f"{low:.3g} to {high:.3g}"
        names.append(name)
        groups[members.index] = name
    return groups, names


def top_categories(values, keep=7):
    """A category column as text; with many different values only the most common `keep` stay and the rest
    become "other", so a figure legend stays readable. Returns (values, names in order)."""
    text = values.astype(str).where(values.notna())
    counts = text.value_counts()
    if len(counts) > keep + 1:
        text = text.where(text.isin(counts.index[:keep]) | text.isna(), "other")
        return text, sorted(counts.index[:keep]) + ["other"]
    return text, sorted(counts.index)


def guess_disease_columns(columns, words):
    """First guess of the disease-related columns (cognition, biomarkers, ...): names that contain one of the
    words (config.yaml disease_words). The user checks it in tab 1."""
    return [col for col in columns if any(word in col.lower() for word in words)]


def visit_times(values):
    """A visit column as numbers, and their unit: numbers stay as they are (visit 1, 2, ... or years),
    dates become years (2019-07-02 -> 2019.5). Returns (numbers, "years" or "visits")."""
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().mean() > 0.9:
        return numbers, "visits"
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365.25, "years"


def split_baseline(data, id_col, visit_col):
    """(baseline scans, follow-up scans). SuStaIn needs one scan per person: the first one (earliest visit).
    The later scans are kept, to be staged later with the fitted model. Without a visit column the order of
    the scans is unknown: then only the first row of each person is kept and there are no follow-ups."""
    if not id_col:
        return data, data.iloc[:0]
    ordered = data
    if visit_col:
        time = visit_times(data[visit_col])[0]
        ordered = data.assign(_time=time).sort_values([id_col, "_time"], kind="stable").drop(columns="_time")
    first = ~ordered[id_col].astype(str).duplicated()
    later = ordered[~first] if visit_col else data.iloc[:0]
    return ordered[first].sort_index(), later.sort_index()


def qc_summary(data, columns, regions, suvr_min, suvr_max):
    """Basic quality checks of the uploaded table. Returns a table: Check, Result, Status.

    columns: dict role -> chosen column ("" or None = not in file), from the tab-1 dropdowns.
    regions: chosen region columns. suvr_min / suvr_max: plausible SUVR range (config.yaml).
    """
    rows = []
    def add(check, result, ok):
        rows.append({"Check": check, "Result": result, "Status": "OK" if ok else "Warning"})

    add("People (rows)", len(data), len(data) > 0)
    if columns.get("id"):
        n_duplicates = int(data[columns["id"]].duplicated().sum())
        add("Duplicate subject IDs", n_duplicates, n_duplicates == 0)
    add("Region columns selected", len(regions), len(regions) >= 2)
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")   # text becomes missing
    n_missing = int(values.isna().sum().sum())
    add("Missing or non-numeric region values", n_missing, n_missing == 0)
    n_outside = int(((values < suvr_min) | (values > suvr_max)).sum().sum())
    add(f"Region values outside {suvr_min}-{suvr_max} (implausible SUVR)", n_outside, n_outside == 0)
    for role in ["age", "sex", "diagnosis", "amyloid"]:
        if columns.get(role):
            n_missing_role = int(data[columns[role]].isna().sum())
            add(f"Missing {role} values", n_missing_role, n_missing_role == 0)
        else:
            add(f"{role.capitalize()} column", "not in file", False)
    return pd.DataFrame(rows)


def demographics_table(data, columns):
    """Who is in the data: N, age, % female and % amyloid+, for everyone and per diagnosis.

    columns: dict role -> chosen column ("" = not in file); roles not in the file are left out.
    Assumes sex values starting with "F" mean female, and amyloid coded 1 = positive, 0 = negative.
    """
    parts = [("All", data)]
    if columns.get("diagnosis"):
        parts += list(data.groupby(data[columns["diagnosis"]].astype(str)))
    rows = []
    for group, part in parts:
        row = {"Group": group, "N": len(part)}
        if columns.get("age"):
            age = pd.to_numeric(part[columns["age"]], errors="coerce")
            row["Age, mean (SD)"] = f"{age.mean():.1f} ({age.std():.1f})"
        if columns.get("sex"):
            first_letter = part[columns["sex"]].astype(str).str.strip().str.upper().str[0]
            row["Female, %"] = round(100 * (first_letter == "F").mean(), 1)
        if columns.get("amyloid"):
            amyloid = pd.to_numeric(part[columns["amyloid"]], errors="coerce")
            row["Amyloid+, %"] = round(100 * amyloid.mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_inclusion_rule(data, columns, min_age_cn, cn_labels, impaired_labels):
    """The paper's inclusion rule (Vogel 2021): keep cognitively normal people older than
    min_age_cn, and MCI/AD people who are amyloid-positive (coded 1). Remove everyone else.

    Labels are compared ignoring upper/lower case. Returns (kept data, table of reasons with counts).
    If the diagnosis, age or amyloid column is missing, nothing is removed and the table says why.
    """
    if not all(columns.get(role) for role in ["diagnosis", "age", "amyloid"]):
        return data, pd.DataFrame([{"Reason": "Rule NOT applied: needs diagnosis, age and amyloid columns",
                                    "N": len(data)}])
    diagnosis = data[columns["diagnosis"]].astype(str).str.strip().str.upper()
    age = pd.to_numeric(data[columns["age"]], errors="coerce")
    amyloid = pd.to_numeric(data[columns["amyloid"]], errors="coerce")
    is_cn = diagnosis.isin([label.upper() for label in cn_labels])
    is_impaired = diagnosis.isin([label.upper() for label in impaired_labels])
    keep_cn = is_cn & (age > min_age_cn)
    keep_impaired = is_impaired & (amyloid == 1)
    reasons = pd.DataFrame([
        (f"Kept: cognitively normal, older than {min_age_cn}", keep_cn.sum()),
        ("Kept: MCI/AD, amyloid-positive", keep_impaired.sum()),
        (f"Removed: cognitively normal, aged {min_age_cn} or younger (or age unknown)", (is_cn & ~keep_cn).sum()),
        ("Removed: MCI/AD, not amyloid-positive (negative or unknown)", (is_impaired & ~keep_impaired).sum()),
        ("Removed: other or missing diagnosis", (~is_cn & ~is_impaired).sum()),
    ], columns=["Reason", "N"])
    return data[keep_cn | keep_impaired], reasons


def number_and_category_columns(data, columns):
    """Split columns into numbers (e.g. MMSE, age) and categories (e.g. sex, APOE4 0/1).
    A number column has almost only numbers in its filled cells (empty cells do not count) and more than
    2 different values."""
    numbers, categories = [], []
    for col in columns:
        filled = data[col].dropna()
        values = pd.to_numeric(filled, errors="coerce")
        if len(filled) and values.notna().mean() > 0.9 and filled.nunique() > 2:
            numbers.append(col)
        else:
            categories.append(col)
    return numbers, categories


def as_text(values):
    """Values as text without empty cells; 1.0 becomes "1"."""
    return values.dropna().map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))


def covariate_summary(data, columns, group_col=None, groups=None):
    """One row per column: type, missing values, and mean (SD) for numbers or % per value for
    categories - for everyone and (if group_col is given) for each group, e.g. each diagnosis."""
    numbers, _ = number_and_category_columns(data, columns)
    parts = [("All", data)]
    if group_col:
        parts += [(g, data[data[group_col].astype(str) == g]) for g in groups]
    rows = []
    for col in columns:
        missing = data[col].isna().sum()
        row = {"Column": col, "Type": "number" if col in numbers else "category",
               "Missing": f"{missing} ({100 * missing / len(data):.0f}%)"}
        for name, part in parts:
            if col in numbers:
                values = pd.to_numeric(part[col], errors="coerce")
                row[name] = f"{values.mean():.1f} ({values.std():.1f})"
            else:
                shares = as_text(part[col]).value_counts(normalize=True)
                row[name] = ", ".join(f"{value}: {100 * share:.0f}%" for value, share in shares.head(4).items())
        rows.append(row)
    return pd.DataFrame(rows)


def region_trend_tests(data, regions, x, groups=None, group_name="group"):
    """Per region: does the value change with x, does it differ between the groups, and does the effect of
    x differ between the groups (interaction)? Linear model: value ~ x (+ group + x:group), type-II tests.
    Returns a table of p-values (not corrected for testing many regions) and, for a number x, the slope.
    groups: a Series of group labels (same rows as data) or None."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    x_is_number = pd.to_numeric(data[x], errors="coerce").notna().mean() > 0.9
    rows = {}
    for region in regions:
        frame = pd.DataFrame({"y": pd.to_numeric(data[region], errors="coerce"),
                              "x": pd.to_numeric(data[x], errors="coerce") if x_is_number else data[x].astype(str)})
        x_term = "x" if x_is_number else "C(x)"
        if groups is not None:
            frame["g"] = groups.astype(str)
        frame = frame.dropna()
        formula = f"y ~ {x_term}" + (f" * C(g)" if groups is not None else "")
        fit = smf.ols(formula, data=frame).fit()
        tests = sm.stats.anova_lm(fit, typ=2)["PR(>F)"]
        row = {"people": len(frame)}
        if x_is_number:
            row[f"slope per unit of {x}"] = fit.params["x"]
        row[f"p: {x}"] = tests[x_term]
        if groups is not None:
            row[f"p: {group_name}"] = tests["C(g)"]
            row[f"p: {x} x {group_name} (interaction)"] = tests[f"{x_term}:C(g)"]
        rows[region] = row
    return pd.DataFrame(rows).T

