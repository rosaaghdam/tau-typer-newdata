"""Paper steps 9-11 - read SuStaIn's saved results and turn them into readable tables and figures."""
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd


class NumbersOnlyUnpickler(pickle.Unpickler):
    """Reads pySuStaIn's result files but refuses anything except numbers and numpy arrays.
    (A normal pickle file can run any code when it is opened - an uploaded run must not.)"""
    ALLOWED = {"_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"}

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"not allowed in a result file: {module}.{name}")


def load_sustain_result(output_folder, dataset_name, n_subtypes):
    """Load pySuStaIn's saved result for the model with n_subtypes subtypes.

    pySuStaIn counts from 0: the 4-subtype model is saved as <dataset_name>_subtype3.pickle.
    """
    path = os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n_subtypes - 1}.pickle")
    return read_result_file(path, os.path.getmtime(path))


@lru_cache(maxsize=16)
def read_result_file(path, modified):
    """Read one result file once and keep it in memory (the tabs ask for the same file many times).
    'modified' is the file's time stamp, so a changed file is read again."""
    with open(path, "rb") as f:
        return NumbersOnlyUnpickler(f).load()


def size_order(result):
    """pySuStaIn's own subtype numbers, the subtype with the most tau+ people first.

    (pySuStaIn's PVD sorts by its estimated subtype proportions, which can differ from the number of
    people assigned; the app passes this order to the PVD too, so all figures and tables agree.)
    """
    k = result["samples_f"].shape[0]
    tau_pos = result["ml_stage"].ravel() > 0
    counts = np.bincount(result["ml_subtype"].ravel()[tau_pos].astype(int), minlength=k)
    return np.argsort(-counts, kind="stable")


def pvd_order(result, order):
    """The subtype_order to give pySuStaIn's PVD so its panels follow `order` (from size_order).

    pySuStaIn's ml_subtype and prob_subtype number the subtypes AFTER sorting them by estimated
    proportion f (in subtype_and_stage_individuals), but samples_sequence - drawn by the PVD - keeps
    the unsorted numbers. So we translate back, sorting exactly as pySuStaIn does.
    """
    by_f = np.argsort(np.mean(result["samples_f"], axis=1))[::-1]
    return by_f[np.asarray(order)]


def event_names(Z_vals, regions):
    """Names of the events in pySuStaIn's own order ("L_MTL z2", ...): all regions at the first
    cut-point, then all at the second, ...; empty places (0) are skipped, as pySuStaIn does."""
    return [f"{regions[i]} z{Z_vals[i, j]:g}" for j in range(Z_vals.shape[1]) for i in range(len(regions))
            if Z_vals[i, j] > 0]


def event_orders(result, Z_vals, regions, order, min_share=0.05):
    """For each subtype (Subtype 1 first): (EM order, MCMC order, MCMC stage ranges).
    EM order = the single best order EM found (ml_sequence_EM; the MCMC starts there).
    MCMC order = events sorted by their average position over all MCMC samples: a summary of the samples
    (the PVD shows the full distribution of positions); this exact order may never occur in any sample.
    Stage range of each MCMC event, e.g. "9-11": the stages where it is in at least min_share of the
    samples; "" if it is (almost) always at one stage."""
    names = event_names(Z_vals, regions)
    orders = []
    for s in pvd_order(result, order):                      # same subtype numbers as samples_sequence
        em = result["ml_sequence_EM"][s].astype(int)
        sequences = result["samples_sequence"][s]                 # positions x samples: event at each position
        positions = np.argsort(sequences, axis=0)                 # events x samples: position of each event
        mcmc = np.argsort(positions.mean(axis=1), kind="stable")  # by average position; ties keep pySuStaIn's order
        ranges = []
        for e in mcmc:
            share = np.bincount(positions[e], minlength=len(names)) / positions.shape[1]
            stages = np.flatnonzero(share >= min_share) + 1       # stage = position + 1
            ranges.append(f"{stages.min()}-{stages.max()}" if stages.min() < stages.max() else "")
        orders.append(([names[e] for e in em], [names[e] for e in mcmc], ranges))
    return orders


def subtype_fractions(result, order):
    """Estimated fraction f of each subtype (Subtype 1 first): (EM, in the best MCMC sample, mean over all
    MCMC samples)."""
    subtypes = pvd_order(result, order)                    # same numbers as samples_sequence
    samples_f = np.asarray(result["samples_f"])
    best = int(np.argmax(np.asarray(result["samples_likelihood"]).ravel()))
    return (np.asarray(result["ml_f_EM"]).ravel()[subtypes], samples_f[subtypes, best],
            samples_f.mean(axis=1)[subtypes])


def best_sample_orders(result, Z_vals, regions, order):
    """(MCMC sample with the highest likelihood (1 = first), the event order of each subtype in it, Subtype 1
    first). pySuStaIn starts the MCMC from the EM order, so this is often the EM order itself."""
    names = event_names(Z_vals, regions)
    best = int(np.argmax(np.asarray(result["samples_likelihood"]).ravel()))
    orders = [[names[e] for e in result["samples_sequence"][s][:, best].astype(int)]
              for s in pvd_order(result, order)]
    return best + 1, orders


def stage_sd(prob_stage):
    """How sure SuStaIn is about each person's stage: the SD (in stages) of their stage probabilities.
    SD = sqrt(sum(stage^2 * p) - mean^2), mean = sum(stage * p). 0 = sure; larger = could be several stages."""
    prob = prob_stage / prob_stage.sum(axis=1, keepdims=True)          # each person's row sums to 1
    stages = np.arange(prob.shape[1])
    mean = prob @ stages
    return np.sqrt(np.clip(prob @ stages ** 2 - mean ** 2, 0, None))


def subject_table(result, subject_ids, poor_fit_threshold, order=None):
    """One row per person: subtype, stage and their probabilities.

    Stage 0 = no abnormal tau -> "S0" (no subtype), as in the paper.
    poor_fit = tau+ but subtype probability below the threshold (paper: 0.5).
    order (from size_order): renumber subtypes so Subtype 1 = largest, as in the PVD.
    Without order, pySuStaIn's own numbers are kept.
    """
    raw = result["ml_subtype"].ravel().astype(int)
    if order is not None:
        raw = np.argsort(order)[raw]                    # position of each subtype in the size order
    table = pd.DataFrame({
        "ml_subtype": raw + 1,                                         # 1..k, like the figure titles
        "prob_subtype": result["prob_ml_subtype"].ravel(),
        "ml_stage": result["ml_stage"].ravel().astype(int),
        "prob_stage": result["prob_ml_stage"].ravel(),
    }, index=subject_ids)
    table["stage_sd"] = stage_sd(result["prob_stage"]).round(3)
    table["subtype"] = np.where(table["ml_stage"] == 0, "S0", "Subtype " + table["ml_subtype"].astype(str))
    table["poor_fit"] = (table["ml_stage"] > 0) & (table["prob_subtype"] < poor_fit_threshold)
    return table


def model_likelihoods(output_folder, dataset_name):
    """Log-likelihood of every saved model (1, 2, ... subtypes), from pySuStaIn's MCMC samples.

    Higher = fits this data better. It (almost) always rises with more subtypes, so it cannot
    choose the number of subtypes on its own - cross-validation (CVIC) does that.
    """
    rows, n = {}, 1
    while os.path.exists(os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n - 1}.pickle")):
        likelihood = load_sustain_result(output_folder, dataset_name, n)["samples_likelihood"].ravel()
        rows[n] = {"mean log-likelihood": likelihood.mean(), "best log-likelihood": likelihood.max()}
        n += 1
    return pd.DataFrame(rows).T.rename_axis("subtypes")


def mcmc_traces(output_folder, dataset_name):
    """{number of subtypes: log-likelihood of every MCMC sample} for every saved model."""
    traces, n = {}, 1
    while os.path.exists(os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n - 1}.pickle")):
        traces[n] = load_sustain_result(output_folder, dataset_name, n)["samples_likelihood"].ravel()
        n += 1
    return traces


def stage_quartiles(stages):
    """Q1, median and Q3 of the stages, always a real stage (no 12.5): the stage of the person there."""
    return [int(np.quantile(stages, q, method="inverted_cdf")) for q in (0.25, 0.5, 0.75)]


def subtype_summary(table, orders=None):
    """People per group (S0, Subtype 1, ...): number, %, how many fit poorly, and the stage
    [Q1, median, Q3]. With orders {"Subtype 1": [event names, first to last], ...}: also the last event
    reached at each of these stages (stage k = the first k events of the subtype have happened)."""
    rows = {}
    for group, people in table.groupby("subtype"):
        quartiles = stage_quartiles(people["ml_stage"])
        row = {"people": len(people), "%": round(100 * len(people) / len(table), 1),
               "poor fit": int(people["poor_fit"].sum()),
               "stage [Q1, median, Q3]": "[" + ", ".join(str(q) for q in quartiles) + "]"}
        if orders is not None:
            events = orders.get(group)                  # S0 has no events
            row["last event at Q1, median, Q3"] = ("" if events is None else
                                                    "[" + ", ".join(events[q - 1] for q in quartiles) + "]")
        rows[group] = row
    return pd.DataFrame(rows).T.rename_axis("subtype")
