"""Compare results on the same people: normalization methods with each other, or with known labels."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score


def group_names(names):
    """S0 first, then the subtypes in the model's order."""
    return ["S0"] + list(names)


def group_counts(groups, names):
    """{result name: Series of group labels (S0 / subtype names)} -> people per group (rows) per result."""
    order = group_names(names)
    return pd.DataFrame({result: labels.value_counts().reindex(order, fill_value=0) for result, labels in groups.items()})


def crosstab(a, b, names_a, names_b):
    """People per group of a (rows) and of b (columns), on the people in both; every group shown."""
    common = a.index.intersection(b.index)
    table = pd.crosstab(a.loc[common], b.loc[common])
    return table.reindex(index=group_names(names_a), columns=group_names(names_b), fill_value=0)


def agreement(groups, stages):
    """Every pair of results: people in both, % same group, % same stage, adjusted Rand index of the groups
    (1 = the same grouping, ~0 = chance) and the Spearman correlation of the stages."""
    rows = []
    for a, b in itertools.combinations(groups, 2):
        common = groups[a].index.intersection(groups[b].index)
        ga, gb = groups[a].loc[common], groups[b].loc[common]
        sa, sb = stages[a].loc[common].astype(float), stages[b].loc[common].astype(float)
        rho = stats.spearmanr(sa, sb).statistic if len(common) > 2 and sa.nunique() > 1 and sb.nunique() > 1 else np.nan
        rows.append({"result 1": a, "result 2": b, "people": len(common),
                     "same group, %": round(100 * (ga == gb).mean(), 1) if len(common) else np.nan,
                     "same stage, %": round(100 * (sa == sb).mean(), 1) if len(common) else np.nan,
                     "adjusted Rand index": round(adjusted_rand_score(ga, gb), 3) if len(common) else np.nan,
                     "stage correlation (Spearman)": round(float(rho), 3)})
    return pd.DataFrame(rows)


def labels_from_numbers(numbers, names):
    """0 -> "S0", n -> the n-th subtype name (numbers outside 0..k become "other")."""
    return pd.Series([("S0" if int(n) == 0 else names[int(n) - 1]) if pd.notna(n) and 0 <= int(n) <= len(names)
                      else "other" for n in numbers], index=numbers.index)


# ---------- Metrics of one run (no true labels needed) and run files ----------
AD_LABELS = ["AD", "DEMENTIA"]


def run_metrics(table, names, diagnosis=None, amyloid=None, cn_labels=()):
    """One run's numbers: groups (%), certainty (poor fit, subtype probability, stage SD) and, if diagnosis and amyloid
    are known, biological sense: % S0 among amyloid-negative CN people, % with a subtype among amyloid-positive AD.
    table: one row per person with subtype, subtype_number, ml_stage, prob_subtype, stage_sd, poor_fit."""
    from src.normalization import amyloid_negative, amyloid_positive
    tau = table[table["ml_stage"] > 0]
    out = {"people": len(table), "S0, %": round(100 * (table["subtype"] == "S0").mean(), 1)}
    for name in names:
        out[f"{name}, %"] = round(100 * (table["subtype"] == name).mean(), 1)
    out["poor fit, % of tau+"] = round(100 * tau["poor_fit"].astype(bool).mean(), 1) if len(tau) else np.nan
    out["subtype probability, mean (tau+)"] = round(float(tau["prob_subtype"].mean()), 3) if len(tau) else np.nan
    out["stage SD, median (tau+)"] = round(float(tau["stage_sd"].median()), 2) if len(tau) else np.nan
    out["stage, median (tau+)"] = float(tau["ml_stage"].median()) if len(tau) else np.nan
    if diagnosis is not None and amyloid is not None:
        dx = diagnosis.reindex(table.index).astype(str).str.strip().str.upper()
        am = amyloid.reindex(table.index)
        cn_neg = dx.isin([c.upper() for c in cn_labels]) & amyloid_negative(am)
        ad_pos = dx.isin(AD_LABELS) & amyloid_positive(am)
        out["amyloid-neg. CN: people"] = int(cn_neg.sum())
        out["amyloid-neg. CN: S0, %"] = round(100 * (table.loc[cn_neg, "subtype"] == "S0").mean(), 1) if cn_neg.any() else np.nan
        out["amyloid-pos. AD: people"] = int(ad_pos.sum())
        out["amyloid-pos. AD: with a subtype, %"] = (round(100 * (table.loc[ad_pos, "subtype"] != "S0").mean(), 1)
                                                     if ad_pos.any() else np.nan)
    return pd.Series(out)


RUN_COLUMNS = {"subtype": "subtype", "subtype_number": "subtype number", "ml_stage": "stage",
               "prob_subtype": "subtype probability", "stage_sd": "stage SD", "poor_fit": "poor fit"}


def run_file(table, settings, diagnosis=None, amyloid=None):
    """A run as one CSV table: per person the results (RUN_COLUMNS), diagnosis and amyloid if known, and the run's
    settings repeated in every row (so the file alone says how it was made)."""
    out = table[list(RUN_COLUMNS)].rename(columns=RUN_COLUMNS)
    if diagnosis is not None:
        out["diagnosis"] = diagnosis.reindex(table.index)
    if amyloid is not None:
        out["amyloid"] = amyloid.reindex(table.index)
    for key, value in settings.items():
        out[f"setting: {key}"] = value
    return out.rename_axis("ID")


def read_run_file(frame):
    """A run file back as (results table with the app's column names, diagnosis or None, amyloid or None,
    settings dict). Raises ValueError when a results column is missing."""
    frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    missing = [c for c in RUN_COLUMNS.values() if c not in frame.columns]
    if missing:
        raise ValueError("Not a Tau-typer run file - missing: " + ", ".join(missing))
    table = frame[list(RUN_COLUMNS.values())].rename(columns={v: k for k, v in RUN_COLUMNS.items()})
    table["poor_fit"] = table["poor_fit"].astype(str).str.lower().isin(["true", "1"])
    settings = {c.replace("setting: ", ""): frame[c].iloc[0] for c in frame.columns if c.startswith("setting: ")}
    return (table, frame["diagnosis"] if "diagnosis" in frame else None,
            frame["amyloid"] if "amyloid" in frame else None, settings)
