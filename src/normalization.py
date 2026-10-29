"""Turn the new people's lobe values (SUVR) into z-scores on the trained model's scale.

z = (value - normal mean) / normal SD. The methods differ only in where "normal" comes from (the baseline);
lobe averaging, the trained model and its cut-points stay the same:
  training  - the TRAINING cohort's GMM-normal people (the reference pipeline, the benchmark)
  own_gmm   - the same GMM recipe, but on the uploaded people themselves
  reference - the uploaded amyloid-negative, cognitively normal people
  zscored   - the uploaded values already are z-scores (nothing is changed)
"""
import pandas as pd

from src.trained_model import gmm_normal, lobe_values

METHODS = {     # in this order in the app
    "training": "1. Reference pipeline: training cohort's GMM normal",
    "own_gmm": "2. GMM normal of your own data",
    "reference": "3. z-score: your amyloid-negative CN people",
    "zscored": "4. Already z-scores",
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
            raise ValueError("Method 3 needs the diagnosis and amyloid columns (step 1).")
        healthy = healthy_reference(data.loc[lobes_new.index], columns["diagnosis"], columns["amyloid"], cn_labels)
        if healthy.sum() < 10:
            raise ValueError(f"Method 3 needs at least 10 amyloid-negative CN people; found {int(healthy.sum())}.")
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
    """The methods that can run with the chosen columns (3 needs diagnosis + amyloid and >= 10 healthy)."""
    methods = ["training", "own_gmm"]
    if columns.get("diagnosis") and columns.get("amyloid") and data is not None:
        if healthy_reference(data, columns["diagnosis"], columns["amyloid"], cn_labels).sum() >= 10:
            methods.append("reference")
    return methods
