"""Turn the new people's lobe values (SUVR) into z-scores on the trained model's scale.

z = (value - normal mean) / normal SD. The methods differ only in where "normal" comes from:
  training  - the TRAINING cohort's GMM-normal people (the reference pipeline)
  own_gmm   - the same GMM recipe, but on the uploaded people themselves
  reference - the uploaded amyloid-negative, cognitively normal people
  zscored   - the uploaded values already are z-scores (nothing is changed)
"""
import pandas as pd

from src.trained_model import gmm_normal, lobe_values

METHODS = {
    "training": "A. Training cohort's normal",
    "own_gmm": "B. GMM on your own data",
    "reference": "C. Your amyloid-negative CN people",
    "zscored": "D. Already z-scores",
}
METHOD_HELP = {
    "training": "For each lobe a Gaussian mixture is fitted on the model's TRAINING "
                "cohort; its lower (normal) bell curve's people give the normal mean and SD. Your data are "
                "then on exactly the scale the model was trained on - only right if your PET data were made "
                "like the training data (same tracer, reference region and processing).",
    "own_gmm": "The same recipe, on YOUR people: per lobe a Gaussian mixture; the lower bell curve's people are "
               "normal. Corrects for scanner or processing differences, but needs enough people with and "
               "without tau (at least ~50).",
    "reference": "Normal = your amyloid-negative, cognitively normal people (diagnosis and amyloid columns "
                 "needed, at least ~10 such people). The classic choice when a healthy group is available.",
    "zscored": "Your uploaded values already are z-scores on the model's scale: lobes are the mean of the "
               "regional values and nothing else is changed.",
}


def amyloid_negative(values):
    """True where amyloid is negative: 0 / 0.0 / 'negative' / 'neg' / 'no' / 'false' / 'A-' (any case)."""
    text = values.astype(str).str.strip().str.lower()
    return text.isin(["0", "0.0", "negative", "neg", "no", "false", "a-", "-"])


def amyloid_positive(values):
    """True where amyloid is positive: 1 / 'positive' / 'pos' / 'yes' / 'true' / 'A+' (any case)."""
    text = values.astype(str).str.strip().str.lower()
    return text.isin(["1", "1.0", "positive", "pos", "yes", "true", "a+", "+"])


def healthy_reference(data, diagnosis_col, amyloid_col, cn_labels):
    """The uploaded people who are cognitively normal (label in cn_labels, any case) and amyloid-negative."""
    diagnosis = data[diagnosis_col].astype(str).str.strip().str.upper()
    return diagnosis.isin([label.upper() for label in cn_labels]) & amyloid_negative(data[amyloid_col])


def normal_table(rows):
    """{lobe: (mean, sd, n, note)} -> a table with one row per lobe."""
    return pd.DataFrame(rows, index=["normal mean", "normal SD", "normal people", "note"]).T.rename_axis("lobe")


def normalize(method, lobes_new, package, gmm_settings, data=None, columns=None, cn_labels=()):
    """z-scores of the new people's lobes with one method. Returns (z table, normal table, fitted GMMs
    {lobe: (values the GMM saw, model)} for the GMM methods, notes)."""
    lobes = package["lobes"]
    rows, fits, notes = {}, {}, []
    if method == "zscored":
        return lobes_new[lobes].copy(), None, {}, ["The uploaded values were used as z-scores."]
    if method == "training":
        source = lobe_values(package["reference"], package["regions"])
    elif method == "own_gmm":
        source = lobes_new
        if len(source) < 50:
            notes.append(f"Only {len(source)} people: a Gaussian mixture is unreliable below ~50.")
    elif method == "reference":
        if not columns or not columns.get("diagnosis") or not columns.get("amyloid"):
            raise ValueError("Method C needs the diagnosis and amyloid columns (tab 1).")
        healthy = healthy_reference(data.loc[lobes_new.index], columns["diagnosis"], columns["amyloid"], cn_labels)
        if healthy.sum() < 10:
            raise ValueError(f"Method C needs at least 10 amyloid-negative CN people; found {int(healthy.sum())}.")
        source = lobes_new[healthy]
        for lobe in lobes:
            rows[lobe] = (source[lobe].mean(), source[lobe].std(), len(source), "")
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    if method in ("training", "own_gmm"):
        for lobe in lobes:
            values = source[lobe].dropna()
            mean, sd, n, model, note = gmm_normal(values, gmm_settings)
            rows[lobe] = (mean, sd, n, note)
            fits[lobe] = (values.to_numpy(), model)
            if note:
                notes.append(f"{lobe}: {note}.")
    table = normal_table(rows)
    means, sds = table["normal mean"].astype(float), table["normal SD"].astype(float)
    z = (lobes_new[lobes] - means[lobes]) / sds[lobes]
    return z, table, fits, notes


def available_methods(columns, data, cn_labels):
    """The methods that can run with the chosen columns (C needs diagnosis + amyloid and >= 10 healthy)."""
    methods = ["training", "own_gmm"]
    if columns.get("diagnosis") and columns.get("amyloid") and data is not None:
        if healthy_reference(data, columns["diagnosis"], columns["amyloid"], cn_labels).sum() >= 10:
            methods.append("reference")
    return methods
