"""Check SuStaIn's results against the truth of simulated data (app tab "Simulation check").

With simulated data we know each person's true subtype and stage. SuStaIn never sees them: they are
removed before the run and used only afterwards, here. SuStaIn numbers its subtypes by size (Subtype 1,
2, ...), not by the names of the simulation, so each found subtype is first paired with the true subtype
it shares most people with.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

NO_TAU_WORDS = {"s0", "0", "none", "no", "nan", "", "normal", "tau-", "tau_negative"}


def read_truth(truth, id_col, subtype_col, stage_col=None):
    """The truth labels as a table by person (IDs as text): true_subtype (text; "S0" = no abnormal tau,
    also for 0 / none / empty) and true_stage (a number, or missing without a stage column)."""
    subtype = truth[subtype_col].astype(str).str.strip()
    subtype = subtype.where(~subtype.str.lower().isin(NO_TAU_WORDS), "S0")
    table = pd.DataFrame({"true_subtype": subtype.to_numpy()}, index=truth[id_col].astype(str).to_numpy())
    table["true_stage"] = pd.to_numeric(truth[stage_col], errors="coerce").to_numpy() if stage_col else np.nan
    return table[~table.index.duplicated()]


def match_subtypes(true_subtype, found_subtype):
    """Pair each found subtype ("Subtype 1", ...) with one true subtype, so that as many people as possible
    agree (Hungarian algorithm; people tau+ in both only). Returns {found: true}; a found subtype without
    a partner (more subtypes found than simulated) is left out."""
    both = (true_subtype != "S0") & (found_subtype != "S0")
    counts = pd.crosstab(found_subtype[both], true_subtype[both])
    if counts.empty:
        return {}
    rows, cols = linear_sum_assignment(-counts.to_numpy())
    return {counts.index[r]: counts.columns[c] for r, c in zip(rows, cols)}


def compare_with_truth(table, truth):
    """One row per person who is in both the run and the truth file: true and found subtype and stage.
    table: the run's subject table (subtype "S0" / "Subtype 1", ..., ml_stage). Returns (rows, pairing)."""
    people = table.index.astype(str).intersection(truth.index)
    found = table.set_axis(table.index.astype(str)).loc[people]
    rows = pd.DataFrame({"true subtype": truth.loc[people, "true_subtype"],
                         "found subtype": found["subtype"], "true stage": truth.loc[people, "true_stage"],
                         "found stage": found["ml_stage"], "subtype probability": found["prob_subtype"].round(2)})
    pairing = match_subtypes(rows["true subtype"], rows["found subtype"])
    rows["found = true subtype"] = rows["found subtype"].map({"S0": "S0", **pairing}).fillna("(no partner)")
    rows["subtype right"] = rows["found = true subtype"] == rows["true subtype"]
    rows["stage error"] = rows["found stage"] - rows["true stage"]
    return rows.rename_axis("person"), pairing


def summary_numbers(rows):
    """The key numbers: people, tau+ found right (sensitivity / specificity), subtype accuracy and
    adjusted Rand index (people tau+ in both), and how close the stages are (true tau+ people)."""
    true_tau, found_tau = rows["true subtype"] != "S0", rows["found subtype"] != "S0"
    both = true_tau & found_tau
    numbers = {"people": len(rows),
               "tau+ found (sensitivity)": found_tau[true_tau].mean(),
               "S0 found (specificity)": (~found_tau[~true_tau]).mean(),
               "subtype right (tau+ in both)": rows.loc[both, "subtype right"].mean(),
               "adjusted Rand index": (adjusted_rand_score(rows.loc[both, "true subtype"], rows.loc[both, "found subtype"])
                                       if both.sum() > 1 else np.nan)}
    staged = true_tau & rows["true stage"].notna()
    if staged.sum() > 2:
        numbers["stage correlation (Spearman)"] = stats.spearmanr(rows.loc[staged, "true stage"],
                                                                 rows.loc[staged, "found stage"])[0]
        numbers["stage error (mean absolute)"] = rows.loc[staged, "stage error"].abs().mean()
        numbers["stage within 2"] = (rows.loc[staged, "stage error"].abs() <= 2).mean()
    return numbers


def per_true_subtype(rows):
    """One row per true subtype: people, found as tau+, found as its partner subtype (recall), how many
    of those found as it really are it (precision), and the stage agreement."""
    result = {}
    for true in sorted(rows["true subtype"].unique(), key=lambda s: (s != "S0", s)):
        people = rows[rows["true subtype"] == true]
        named = rows[rows["found = true subtype"] == true]
        row = {"people": len(people),
               "found tau+ (%)": 100 * (people["found subtype"] != "S0").mean(),
               "found as this subtype (%)": 100 * people["subtype right"].mean(),
               "of those found as it, really it (%)": 100 * named["subtype right"].mean() if len(named) else np.nan}
        staged = people[people["true stage"].notna()]
        if true != "S0" and len(staged) > 2:
            row["stage Spearman rho"] = stats.spearmanr(staged["true stage"], staged["found stage"])[0]
            row["stage error, mean absolute"] = staged["stage error"].abs().mean()
            row["stage error, mean (+ = found too late)"] = staged["stage error"].mean()
        result[true] = row
    return pd.DataFrame(result).T.rename_axis("true subtype").round(2)


def confusion(rows, pairing):
    """Counts: true subtype (rows) x found subtype (columns), both S0 first. The true subtypes follow their
    found partners (Subtype 1's partner first, ...), so a perfect result is a diagonal."""
    table = pd.crosstab(rows["true subtype"], rows["found subtype"])
    columns = sorted(table.columns, key=lambda s: (s != "S0", s))
    partners = [pairing[c] for c in columns if c in pairing]
    others = sorted(t for t in table.index if t != "S0" and t not in partners)   # true subtypes not found
    order = [t for t in ["S0"] + partners + others if t in table.index]
    return table.loc[order, columns]
