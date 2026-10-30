"""The trained model applied to its own training cohort, with the same settings as the user's run - so the app can
show the same group tables (people per group, studies, clinical summary) for both. Only group numbers leave this
module: the training cohort's individual values stay private."""
import pandas as pd

from src.normalization import normalize
from src.trained_model import apply_model, check_z, clean_up, label_table, lobe_values, tau_positive


def training_table(package, model_cfg, names, cleanup, threshold, n_samples, poor_fit):
    """One row per training person: subtype, stage, ... (as the user's result table). Normalization: method 1 (the
    training cohort's own GMM normal) - the scale the model was trained on."""
    reference = package["reference"]
    z = normalize("training", lobe_values(reference, package["regions"]), package, model_cfg["gmm"])[0]
    z = z[~check_z(z, package, model_cfg["max_z_above_z_max"])]
    table, _, _ = apply_model(z, package, n_samples)
    if cleanup:
        _, tau_pos, _ = tau_positive(reference.loc[z.index], reference, model_cfg["cleanup"]["regions"], threshold,
                                     model_cfg["gmm"])
        table = clean_up(table, tau_pos)
    return label_table(table, names, poor_fit)


def study_counts(table, study, names):
    """People per group (rows: S0, then the subtypes) and study (columns)."""
    counts = pd.crosstab(table["subtype"], study.loc[table.index].astype(str))
    return counts.reindex([g for g in ["S0"] + list(names) if g in counts.index]).rename_axis("group")


def clinical_table(table, data, columns, names):
    """Per group: people, mean and SD age, % female, % of each diagnosis. columns: {"age": ..., "sex": ...,
    "diagnosis": ...} - the column names in data (empty = not shown)."""
    data = data.loc[table.index]
    group = table["subtype"]
    out = pd.DataFrame({"people": group.value_counts()})
    if columns.get("age"):
        age = pd.to_numeric(data[columns["age"]], errors="coerce")
        out["age, mean"] = age.groupby(group).mean().round(1)
        out["age, SD"] = age.groupby(group).std().round(1)
    if columns.get("sex"):
        female = data[columns["sex"]].astype(str).str.strip().str.upper().str[0] == "F"
        out["female, %"] = (100 * female.groupby(group).mean()).round(1)
    if columns.get("diagnosis"):
        share = pd.crosstab(group, data[columns["diagnosis"]].astype(str), normalize="index") * 100
        out = out.join(share.round(1).add_prefix("% "))
    order = [g for g in ["S0"] + list(names) if g in out.index]
    return out.loc[order].rename_axis("group")
