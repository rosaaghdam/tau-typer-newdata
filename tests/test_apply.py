"""The new pipeline's calculations (src/trained_model.py, normalization.py, roi_names.py, compare.py) on the
simulated data. Run from the project folder:   ./env/bin/python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ROOT
from src.input_format import check_format, input_format, template_table
from src.compare import agreement, crosstab, group_counts, labels_from_numbers
from src.normalization import amyloid_negative, normalize
from src.roi_names import match_regions, region_key
from src.trained_model import (apply_model, check_z, clean_up, load_model_package, lobe_values, required_regions,
                               tau_positive)

SIM = ROOT / "data" / "simulated"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))["model"]


@pytest.fixture(scope="module")
def package():
    return load_model_package(CFG, ROOT)


@pytest.fixture(scope="module")
def new():
    return pd.read_csv(SIM / "df_sustain_true.csv", index_col="filename")


def test_region_names_in_other_styles():
    assert region_key("CTX_LH_ENTORHINAL") == region_key("ctx-lh-entorhinal") == region_key("entorhinal_L")
    assert region_key("Left-Hippocampus") == ("L", "hippocampus")
    assert region_key("ctx-lh-entorhinal_SIZE") != region_key("ctx-lh-entorhinal")     # a volume is not tau
    table = match_regions(["ctx_lh_cuneus", "CTX-LH-CUNEUS", "x"], ["ctx-lh-cuneus", "ctx-rh-cuneus"])
    assert list(table["status"]) == ["more than one", "missing"]


def test_reference_pipeline_is_reproduced_exactly(package, new):
    """Lobes -> training-normal z-scores -> model: equal to the answer key made with the original code."""
    z, _, _, _ = normalize("training", lobe_values(new, package["regions"]), package, CFG["gmm"])
    key_z = pd.read_csv(SIM / "reference_pipeline_lobar_z.csv", index_col="filename").loc[z.index, package["lobes"]]
    assert np.allclose(z.to_numpy(), key_z.to_numpy(), atol=1e-12)
    table, _ = apply_model(z, package, CFG["n_samples"])
    key = pd.read_csv(SIM / "expected_reference_pipeline.csv", index_col="filename").loc[table.index]
    assert (table["subtype_number"] == key["subtype"]).all() and (table["ml_stage"] == key["stage"]).all()
    assert np.allclose(table[[f"p_S{n}" for n in range(1, 5)]], key[[f"p_S{n}" for n in range(1, 5)]], atol=1e-12)


def test_cleanup_sets_tau_negative_people_to_s0(package, new):
    z, _, _, _ = normalize("training", lobe_values(new, package["regions"]), package, CFG["gmm"])
    table, _ = apply_model(z, package, CFG["n_samples"])
    _, tau_pos, notes = tau_positive(new, package["reference"], CFG["cleanup"]["regions"], CFG["cleanup"]["threshold"],
                                     CFG["gmm"])
    cleaned = clean_up(table, tau_pos)
    assert notes == []
    assert cleaned["subtype_number"].value_counts().sort_index().tolist() == [319, 90, 71, 31, 19]   # the reference notebook
    assert (cleaned.loc[cleaned["cleaned"], ["subtype_number", "ml_stage"]] == 0).all().all()


def test_every_normalization_method_runs(package, new):
    lobes = lobe_values(new, package["regions"])
    for method in ("own_gmm", "reference", "choroid", "zscored"):
        z, normals, _, _ = normalize(method, lobes, package, CFG["gmm"], data=new,
                                     columns={"diagnosis": "diagnosis", "amyloid": "amyloid"}, cn_labels=["CN"])
        assert z.shape == (len(new), 10) and z.notna().all().all()
        assert normals is None if method == "zscored" else len(normals) == 10


def test_amyloid_text_and_numbers():
    assert amyloid_negative(pd.Series(["Negative", "0", "neg", "Positive", "1", ""])).tolist() == \
        [True, True, True, False, False, False]


def test_people_pysustain_cannot_use_are_found(package):
    z = pd.DataFrame(0.0, index=["a", "b", "c"], columns=package["lobes"])
    z.loc["b", "L_MTL"] = package["Z_max"][0] + 20
    z.loc["c", "R_Frontal"] = np.nan
    assert check_z(z, package, 10).tolist() == [False, True, True]


def test_compare_two_results():
    names = ["A", "B"]
    a = labels_from_numbers(pd.Series([0, 1, 2, 2], index=list("wxyz")), names)
    b = labels_from_numbers(pd.Series([0, 1, 1, 2], index=list("wxyz")), names)
    assert crosstab(a, b, names, names).loc["B", "A"] == 1
    assert group_counts({"a": a}, names)["a"].tolist() == [1, 1, 2]
    row = agreement({"a": a, "b": b}, {"a": pd.Series([0, 3, 5, 6], index=list("wxyz")),
                                        "b": pd.Series([0, 3, 5, 7], index=list("wxyz"))}).iloc[0]
    assert row["same group, %"] == 75.0 and row["same stage, %"] == 75.0


def test_required_regions_include_the_cleanup_ones(package):
    needed = required_regions(package, CFG["cleanup"]["regions"])
    assert set(CFG["cleanup"]["regions"]) <= set(needed) and len(needed) == len(set(needed)) == 64


def test_inclusion_rule_reads_amyloid_as_text():
    from src.qc import apply_inclusion_rule
    data = pd.DataFrame({"dx": ["CN", "CN", "MCI", "AD", "AD", "SCD", "other"], "age": [70, 35, 70, 70, 70, 60, 70],
                         "amy": ["Negative", "Negative", "Positive", "Negative", "1", "Negative", "Positive"]})
    kept, reasons = apply_inclusion_rule(data, {"diagnosis": "dx", "age": "age", "amyloid": "amy"}, 40,
                                         ["CN", "SCD"], ["MCI", "AD"])
    assert kept.index.tolist() == [0, 2, 4, 5]
    assert reasons["N"].tolist() == [2, 2, 1, 1, 1]


def test_input_template_and_format_check(package):
    """The template lists the information columns + every region the model needs; the format check passes the
    example data and reports the regions missing from the renamed test file."""
    fmt = input_format(required_regions(package, CFG["cleanup"]["regions"]))
    template = template_table(fmt)
    assert template.shape == (1, 72) and template.columns[0] == "subject_id"
    regions = fmt.loc[8:, "column"].tolist()
    example = pd.read_csv(SIM / "df_sustain_true.csv")
    assert set(check_format(example, match_regions(list(example.columns), regions), "filename")["Level"]) == {"OK"}
    renamed = pd.read_csv(SIM / "test_newuser.csv")
    checks = check_format(renamed, match_regions(list(renamed.columns), regions), "Subject")
    assert checks["Level"].tolist() == ["error", "warning", "OK"]
    assert "Left-Hippocampus" in checks.loc[0, "Details"]


def test_choroid_w_score_is_the_notebooks_regression(package, new):
    """Method 4 = (lobe - (a + b x choroid)) / residual SD, with a and b fitted on the whole training cohort
    (the original notebook's W_Transform with Choroid_tot) - checked against statsmodels."""
    import statsmodels.formula.api as smf
    w = normalize("choroid", lobe_values(new, package["regions"]), package, CFG["gmm"], data=new)[0]
    train = lobe_values(package["reference"], package["regions"]).assign(
        Choroid_tot=package["reference"][["Left-Choroid-Plexus", "Right-Choroid-Plexus"]].mean(axis=1))
    fit = smf.ols("L_MTL ~ Choroid_tot", data=train).fit()
    choroid = new[["Left-Choroid-Plexus", "Right-Choroid-Plexus"]].mean(axis=1)
    expected = (lobe_values(new, package["regions"])["L_MTL"] - fit.predict(pd.DataFrame({"Choroid_tot": choroid}))) \
        / fit.resid.std()
    assert np.allclose(w["L_MTL"], expected)
