"""Paper steps 6-9 - prepare the inputs for pySuStaIn's ZscoreSustain and run it.

We only WRAP pySuStaIn here: all the modelling is done by the original package.
"""
import os
import tempfile

import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from sklearn.model_selection import KFold


def cuts_per_region(regions, z_events):
    """z_events as {region: [cut-points]}. z_events is either one list for all regions (paper:
    [2, 5, 10]) or already a {region: list} dictionary (different events per region)."""
    if isinstance(z_events, dict):
        return {region: list(z_events[region]) for region in regions}
    return {region: list(z_events) for region in regions}


def make_z_vals(regions, z_events):
    """The regions x cut-points table ZscoreSustain needs. A region with fewer cut-points gets 0
    in the empty places - pySuStaIn skips zeros, so regions can have different numbers of events."""
    cuts = cuts_per_region(regions, z_events)
    width = max(len(c) for c in cuts.values())
    Z_vals = np.zeros((len(regions), width))
    for row, region in enumerate(regions):
        Z_vals[row, :len(cuts[region])] = cuts[region]
    return Z_vals


def sustain_inputs(z, regions, z_events, z_max):
    """Turn our z-score table + settings into the 3 inputs ZscoreSustain needs.

    Returns:
        data  - people x regions array of z-scores (plain numbers, no names)
        Z_vals - regions x cut-points table (see make_z_vals)
        Z_max  - the z-score each region reaches at the very end (one per region)
    z_events: one list for all regions, or {region: list}. z_max: one number, or one per region.
    """
    data = z[regions].to_numpy()
    Z_vals = make_z_vals(regions, z_events)
    Z_max = np.broadcast_to(np.asarray(z_max, dtype=float), (len(regions),)).copy()
    return data, Z_vals, Z_max


def count_people_above(z, regions, z_events):
    """How many people reach each cut-point in each region (rarely reached events are hard to place).
    Same cut-points everywhere: columns "z>=2", "z>=5", ...; different per region: columns
    "event 1", "event 2", ... (empty where a region has fewer events)."""
    cuts = cuts_per_region(regions, z_events)
    if all(c == cuts[regions[0]] for c in cuts.values()):
        return pd.DataFrame({f"z>={cut:g}": (z[regions] >= cut).sum() for cut in cuts[regions[0]]})
    rows = {}
    for region in regions:
        rows[region] = {f"event {i + 1}": int((z[region] >= cut).sum()) for i, cut in enumerate(cuts[region])}
    return pd.DataFrame(rows).T


def parse_cut_points(text):
    """Turn typed cut-points like "2, 5, 10" into sorted numbers [2.0, 5.0, 10.0].

    Raises ValueError (with a message for the user) if the text is empty, not numbers, or <= 0.
    """
    try:
        values = sorted(float(part) for part in text.replace(";", ",").split(",") if part.strip())
    except ValueError:
        raise ValueError("Cut-points must be numbers separated by commas, e.g. 2, 5, 10")
    if not values or values[0] <= 0:
        raise ValueError("Cut-points must be positive numbers, e.g. 2, 5, 10")
    return values


def parse_z_max(text, regions):
    """Turn typed Z_max values like "15, 15, 12" into one number per region (in region order).

    One number = the same Z_max for every region. Raises ValueError (message for the user) if the
    text is not numbers or the count does not match the number of regions.
    """
    try:
        values = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
    except ValueError:
        raise ValueError("Z_max must be numbers separated by commas, e.g. 15, 15, 12")
    if len(values) == 1:
        values = values * len(regions)
    if len(values) != len(regions):
        raise ValueError(f"Z_max: give one number per region ({len(regions)} regions), or one number for all "
                         f"- you gave {len(values)}.")
    return dict(zip(regions, values))


def z_max_table(z, method, manual_values, percentile, min_z):
    """Z_max per region, and the numbers that help choose it.

    Per region, max / p99 / p95 of z among people with z >= min_z (clearly abnormal there - tau
    status is unknown in real data). method "manual": Z_max = manual_values[region] (from
    parse_z_max); method "percentile": Z_max = that percentile (e.g. 95) of the same people.
    """
    rows = {}
    for region in z.columns:
        region_min = min_z[region] if isinstance(min_z, dict) else min_z     # first cut-point of the region
        high = z[region][z[region] >= region_min]
        rows[region] = {"people with z >= min": len(high), "max": high.max(),
                        "p99": high.quantile(0.99), "p95": high.quantile(0.95),
                        "Z_max used": manual_values[region] if method == "manual" else high.quantile(percentile / 100)}
    return pd.DataFrame(rows).T.round(2)


def run_sustain(data, Z_vals, Z_max, regions, settings, output_folder, seed, plot=True):
    """Fit pySuStaIn's ZscoreSustain with 1, 2, ... n_subtypes_max subtypes (paper steps 7-9).

    pySuStaIn saves one result file per number of subtypes in <output_folder>/pickle_files/
    and (plot=True) the positional variance diagrams as PNG files in <output_folder>.
    NOTE: if those result files already exist, pySuStaIn LOADS them instead of fitting again,
    so use a new output_folder whenever you change a setting.
    Returns the model and its outputs for the largest number of subtypes.
    """
    os.makedirs(output_folder, exist_ok=True)
    model = ZscoreSustain(data, Z_vals, Z_max, regions,
                          settings["n_startpoints"], settings["n_subtypes_max"],
                          settings["n_iterations_mcmc"], output_folder,
                          settings["dataset_name"], settings["use_parallel_startpoints"], seed)
    outputs = model.run_sustain_algorithm(plot=plot)
    return model, outputs


def stage_new_scans(z, regions, settings, result, n_samples=1000):
    """Subtype and stage new scans (e.g. follow-up scans) with an already fitted model, using pySuStaIn's
    own subtype_and_stage_individuals_newData. The model is not changed or fitted again.
    z: the new scans' z-scores (made with the same normal reference). result: one saved model.
    Returns the same names as a saved result (ml_subtype, prob_ml_subtype, ml_stage, ...), so
    results.subject_table can read it."""
    data, Z_vals, Z_max = sustain_inputs(z, regions, settings["cut_points"], settings["z_max"])
    model = ZscoreSustain(data, Z_vals, Z_max, regions, 1, 1, 1, tempfile.gettempdir(), "new_scans", False,
                          settings["seed"])                    # only used to stage; nothing is fitted
    names = ["ml_subtype", "prob_ml_subtype", "ml_stage", "prob_ml_stage", "prob_subtype", "prob_stage",
             "prob_subtype_stage"]
    staged = model.subtype_and_stage_individuals_newData(data, result["samples_sequence"], result["samples_f"],
                                                         n_samples)
    return dict(zip(names, staged))


def cross_validate(model, n_people, n_folds, seed):
    """pySuStaIn's own cross-validation (paper: 10 folds), for every model 1, 2, ... subtypes.

    We only make the folds: scikit-learn's KFold, shuffled with the seed (so the same folds every time).
    pySuStaIn fits each model on 9/10 of the people and scores it on the other 1/10, for each fold.
    Returns CVIC per model (lower = better; the paper chose the number of subtypes this way) and the
    test log-likelihood per fold and model. plot=True: pySuStaIn saves Log_likelihoods_cv_folds.png.
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_idxs = [test for _, test in folds.split(np.zeros(n_people))]
    return model.cross_validate_sustain_model(test_idxs, plot=True)
