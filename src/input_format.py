"""The input format users should follow: one row per scan, one column per item below.
The app's template download, its format table and the README's table are all made from input_format(),
so they always list exactly the regions the model needs."""
import pandas as pd

# the non-region columns: (name, required, contents, example)
OTHER_COLUMNS = [
    ("subject_id", "yes", "a unique ID per scan", "sub-001"),
    ("study", "no", "study or cohort name", "study_A"),
    ("age", "no", "age at the scan, in years", "71"),
    ("sex", "no", "F or M", "F"),
    ("diagnosis", "no", "clinical diagnosis, e.g. CN, SCD, MCI, AD", "CN"),
    ("amyloid", "no", "amyloid status: positive / negative (or 1 / 0)", "negative"),
    ("Left-Choroid-Plexus", "no", "choroid-plexus SUVR (normalization method 4)", "1.30"),
    ("Right-Choroid-Plexus", "no", "choroid-plexus SUVR (normalization method 4)", "1.28"),
]


def input_format(regions):
    """One row per column of the input file. regions: the FreeSurfer regions the model needs, in order."""
    rows = list(OTHER_COLUMNS)
    rows += [(region, "yes", "regional tau-PET SUVR (FreeSurfer Desikan-Killiany)", "1.15") for region in regions]
    return pd.DataFrame(rows, columns=["column", "required", "contents", "example"])


def template_table(format_table):
    """The template file: every column of the input format, with one example row to show the layout."""
    return pd.DataFrame([format_table["example"].tolist()], columns=format_table["column"].tolist())


def check_format(data, match, id_column):
    """Compare an uploaded table with the input format. Returns a table: Level (error / warning / OK),
    Check, Details. Errors stop the model; warnings are fixed automatically or only reported.
    match: match_regions() with the user's choices (region, column, status). id_column: "" if none."""
    rows = []
    def add(level, check, details=""):
        rows.append({"Level": level, "Check": check, "Details": details})

    missing = match.loc[match["column"].isna(), "region"].tolist()
    add("error" if missing else "OK", f"Required regions found: {len(match) - len(missing)} of {len(match)}",
        ("missing (or found twice - choose it in the sidebar): " + ", ".join(missing)) if missing else "")
    renamed = match[match["status"].isin(["renamed", "chosen by you"])]
    if len(renamed):
        add("warning", f"{len(renamed)} region(s) under another name than the template",
            ", ".join(f"{c} -> {r}" for r, c in zip(renamed["region"], renamed["column"])))
    found = match["column"].dropna().tolist()
    text = [c for c in found if pd.to_numeric(data[c], errors="coerce").isna().sum() > data[c].isna().sum()]
    if text:
        add("error", f"{len(text)} region column(s) contain text instead of numbers", ", ".join(text))
    if not id_column:
        add("warning", "No ID column (template: subject_id)", "rows are numbered instead")
    elif data[id_column].duplicated().any():
        add("warning", "Duplicate IDs", f"{int(data[id_column].duplicated().sum())} rows; rows are numbered instead")
    extra = [c for c in data.columns if c not in found and c != id_column]
    if extra:
        add("OK", f"{len(extra)} other column(s), not regions", ", ".join(map(str, extra[:12])) +
            (" ..." if len(extra) > 12 else ""))
    return pd.DataFrame(rows)
