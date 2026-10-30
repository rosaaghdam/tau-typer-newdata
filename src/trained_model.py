"""Apply an ALREADY-TRAINED SuStaIn model to new people.

Nothing is trained here: pySuStaIn's own subtype_and_stage_individuals_newData gives every person a
subtype, a stage and their probabilities from the saved model (samples_sequence, samples_f).
All file names and settings come from config.yaml (section "model").
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from sklearn.mixture import GaussianMixture

from src.results import NumbersOnlyUnpickler, event_names, stage_sd


def load_model_package(settings, root):
    """Everything needed to apply the model, read from the files named in config.yaml "model".
    Returns a dict: lobes (pySuStaIn's biomarker order), Z_vals, Z_max, samples_sequence, samples_f,
    regions {lobe: [FreeSurfer regions]}, roi_names, reference (training cohort's regional values),
    subtype_names, n_events, n_subtypes, folder."""
    folder = Path(root) / settings["folder"]
    model_settings = json.loads((folder / settings["settings_file"]).read_text())
    with open(folder / settings["samples_file"], "rb") as f:
        saved = NumbersOnlyUnpickler(f).load()            # numbers only - a pickle could otherwise run code
    regions = json.loads((folder / settings["regions_file"]).read_text())
    lobes = list(model_settings["biomarker_order"])
    Z_vals, Z_max = np.array(model_settings["Z_vals"], float), np.array(model_settings["Z_max"], float)
    sequence, shares = np.asarray(saved["samples_sequence"]), np.asarray(saved["samples_f"])
    n_events = int((Z_vals > 0).sum())
    problems = []                                          # the package must fit together
    if set(lobes) != set(regions):
        problems.append("the lobes in the settings file and the regions file differ")
    if Z_vals.shape[0] != len(lobes) or Z_max.shape != (len(lobes),):
        problems.append("Z_vals / Z_max do not have one row per lobe")
    if sequence.ndim != 3 or sequence.shape[1] != n_events or shares.shape != (sequence.shape[0], sequence.shape[2]):
        problems.append(f"the saved model's arrays {sequence.shape} / {shares.shape} do not fit {n_events} events")
    names = list(settings.get("subtype_names") or [])
    names = (names + [f"Subtype {n}" for n in range(len(names) + 1, sequence.shape[0] + 1)])[:sequence.shape[0]]
    if problems:
        raise ValueError("The model files do not fit together: " + "; ".join(problems) + ".")
    likelihoods = mcmc_likelihoods(folder, settings["samples_file"], sequence.shape[0], saved)
    return {"lobes": lobes, "Z_vals": Z_vals, "Z_max": Z_max, "samples_sequence": sequence, "samples_f": shares,
            "regions": regions, "roi_names": [str(r) for r in np.load(folder / settings["roi_names_file"])],
            "reference": pd.read_csv(folder / settings["reference_file"], index_col=settings["reference_id_column"]),
            "subtype_names": names, "n_events": n_events, "n_subtypes": sequence.shape[0], "folder": folder,
            "n_mcmc_samples": sequence.shape[2], "likelihoods": likelihoods}


def mcmc_likelihoods(folder, samples_file, n_subtypes, saved):
    """{number of subtypes: log-likelihood of every saved MCMC sample} of the trained models, as pySuStaIn
    saved them (samples_likelihood). pySuStaIn saves one file per number of subtypes (..._subtype0.pickle = 1
    subtype, ..._subtype3.pickle = 4); the files next to the model's are read too. Empty if none has them."""
    likelihoods = {}
    if "samples_likelihood" in saved:
        likelihoods[n_subtypes] = np.asarray(saved["samples_likelihood"], float).ravel()
    if re.search(r"subtype\d+", samples_file):
        for k in range(1, n_subtypes):
            path = Path(folder) / re.sub(r"subtype\d+", f"subtype{k - 1}", samples_file)
            if path.exists():
                with open(path, "rb") as f:
                    other = NumbersOnlyUnpickler(f).load()
                if "samples_likelihood" in other:
                    likelihoods[k] = np.asarray(other["samples_likelihood"], float).ravel()
    return dict(sorted(likelihoods.items()))


def event_stages(package):
    """Every event of every learned subtype: the stage at which the lobe reaches each cut-point (mean over the
    saved MCMC samples). Returns {subtype name: table lobes x cut-points (z 2, 5, 10)}, pySuStaIn's order;
    NaN where a lobe has no such cut-point."""
    Z_vals, lobes = package["Z_vals"], package["lobes"]
    names = event_names(Z_vals, lobes)
    levels = sorted(set(Z_vals[Z_vals > 0]))
    by_f = np.argsort(package["samples_f"].mean(axis=1))[::-1]          # pySuStaIn's numbering of the subtypes
    out = {}
    for name, s in zip(package["subtype_names"], by_f):
        positions = np.argsort(package["samples_sequence"][s], axis=0)  # events x samples: position of each event
        mean_stage = positions.mean(axis=1) + 1                         # stage = position + 1
        table = pd.DataFrame(np.nan, index=lobes, columns=levels)
        for e, event in enumerate(names):
            lobe, z = event.rsplit(" z", 1)
            table.loc[lobe, float(z)] = mean_stage[e]
        out[name] = table
    return out


def required_regions(package, cleanup_regions):
    """The FreeSurfer regions the app needs from an upload: those of the lobes, then the cleanup ones."""
    needed = [roi for lobe in package["lobes"] for roi in package["regions"][lobe]]
    return list(dict.fromkeys(needed + list(cleanup_regions)))      # each once, in this order


def lobe_values(regional, regions):
    """Each lobe = the plain mean of its regions (pandas mean, missing values skipped).
    regional: people x FreeSurfer regions (the model's names). regions: {lobe: [regions]}."""
    return pd.DataFrame({lobe: regional[rois].mean(axis=1) for lobe, rois in regions.items()}, index=regional.index)


def gmm_normal(values, gmm_settings):
    """The "normal" group of one lobe or region (as ESM_utils.model_tfm):
    compare a 1- and a 2-component Gaussian mixture by BIC; if 2 fit better, the people in the
    LOWER-mean component are normal (chosen explicitly, not by the component's number).
    Returns (normal people's mean, their SD, how many, the fitted 2-component model, a note)."""
    x = np.asarray(values, float).reshape(-1, 1)
    one = GaussianMixture(n_components=1, random_state=gmm_settings["random_state"]).fit(x)
    two = GaussianMixture(n_components=2, random_state=gmm_settings["random_state"], tol=gmm_settings["tol"],
                          max_iter=gmm_settings["max_iter"]).fit(x)
    if one.bic(x) <= two.bic(x):
        return float(x.mean()), float(x.std(ddof=1)), len(x), two, "1 bell curve fits better: everyone used as normal"
    normal = two.predict(x) == np.argmin(two.means_.ravel())
    group = x[normal, 0]
    return float(group.mean()), float(group.std(ddof=1)), int(normal.sum()), two, ""


def check_z(z, package, margin):
    """People whose z-scores pySuStaIn cannot handle: any z more than margin above Z_max, or below -margin
    (the likelihood becomes 0 for them and pySuStaIn fails), or missing. Returns a boolean Series (True = left out)."""
    values = z[package["lobes"]].to_numpy(float)
    too_far = (values > package["Z_max"] + margin) | (values < -margin) | np.isnan(values)
    return pd.Series(too_far.any(axis=1), index=z.index)


def apply_model(z, package, n_samples):
    """Subtype and stage each person with the saved model. No training.
    z: people x lobes (z-scores). Returns one row per person: ml_subtype (1..k in pySuStaIn's order =
    largest average share first), prob_subtype, ml_stage, prob_stage, stage_sd, p_S1..p_Sk and
    subtype_number (0 = S0: stage 0). Also returns pySuStaIn's prob_subtype (people x subtypes) and
    prob_stage (people x stages 0..n_events)."""
    data = z[package["lobes"]].to_numpy(float)
    folder = tempfile.mkdtemp(prefix="tau_typer_apply_")  # pySuStaIn wants an output folder; nothing is saved
    try:
        sustain = ZscoreSustain(data, package["Z_vals"], package["Z_max"], package["lobes"], 1,
                                package["n_subtypes"], 1, folder, "apply", False)
        ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage = \
            sustain.subtype_and_stage_individuals_newData(data, package["samples_sequence"], package["samples_f"],
                                                          int(n_samples))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    table = pd.DataFrame({"ml_subtype": ml_subtype.ravel().astype(int) + 1, "prob_subtype": prob_ml_subtype.ravel(),
                          "ml_stage": ml_stage.ravel().astype(int), "prob_stage": prob_ml_stage.ravel()},
                         index=z.index)
    table["stage_sd"] = stage_sd(prob_stage).round(3)
    for n in range(prob_subtype.shape[1]):
        table[f"p_S{n + 1}"] = prob_subtype[:, n]
    table["subtype_number"] = np.where(table["ml_stage"] > 0, table["ml_subtype"], 0)
    # the 5 groups S0, Subtype 1..k: P(stage 0), and P(subtype k at a stage above 0) - together 1 per person
    table["group_p_S0"] = prob_subtype_stage[:, 0, :].sum(axis=1)
    for n in range(prob_subtype_stage.shape[2]):
        table[f"group_p_S{n + 1}"] = prob_subtype_stage[:, 1:, n].sum(axis=1)
    return table, prob_subtype, prob_stage


def tau_positive(regional, reference, cleanup_regions, threshold, gmm_settings):
    """False-positive check: for each cleanup region (entorhinal, precuneus, both sides) a
    Gaussian mixture is fitted on the TRAINING cohort's raw values; each new person gets P(abnormal) =
    the probability of the higher-mean component. Tau+ = P above threshold in at least one region.
    Returns (P table people x regions, tau+ Series, notes)."""
    probabilities, notes = {}, []
    for roi in cleanup_regions:
        _, _, _, two, note = gmm_normal(reference[roi].dropna(), gmm_settings)
        if note:                                            # such a region is not used
            notes.append(f"{roi}: {note} - not used")
            continue
        abnormal = np.argmax(two.means_.ravel())
        probabilities[roi] = two.predict_proba(regional[[roi]].to_numpy(float))[:, abnormal]
    p = pd.DataFrame(probabilities, index=regional.index)
    return p, (p > threshold).any(axis=1), notes


def clean_up(table, tau_pos):
    """People with a subtype but not tau+ become S0 (subtype 0, stage 0). The values before
    are kept (subtype_before_cleanup, stage_before_cleanup); 'cleaned' marks the people changed."""
    table = table.copy()
    table["subtype_before_cleanup"] = table["subtype_number"]
    table["stage_before_cleanup"] = table["ml_stage"]
    table["cleaned"] = (table["subtype_number"] > 0) & ~tau_pos.reindex(table.index).fillna(False).astype(bool)
    table.loc[table["cleaned"], ["subtype_number", "ml_stage"]] = 0
    return table


def cleanup_summary(table, names):
    """Per initial subtype (the model's assignment before the false-positive check): people retained, reset to S0,
    total and % reset. table: after clean_up."""
    rows = {}
    for number, name in enumerate(names, start=1):
        first = table["subtype_before_cleanup"] == number
        reset = int((first & table["cleaned"].astype(bool)).sum())
        rows[name] = {"retained": int(first.sum()) - reset, "reset to S0": reset, "total": int(first.sum()),
                      "% reset": round(100 * reset / max(int(first.sum()), 1), 1)}
    return pd.DataFrame(rows).T.astype({"retained": int, "reset to S0": int, "total": int})


def label_table(table, names, poor_fit_threshold):
    """Readable columns: subtype ("S0" or the subtype's name) and poor_fit (tau+ but subtype probability
    below the threshold - paper: 0.5)."""
    table = table.copy()
    table["subtype"] = ["S0" if n == 0 else names[n - 1] for n in table["subtype_number"]]
    table["poor_fit"] = (table["ml_stage"] > 0) & (table["prob_subtype"] < poor_fit_threshold)
    return table
