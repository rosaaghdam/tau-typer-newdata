"""Small checks of the calculations in src/ (no app, no pySuStaIn run)."""
import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.app_runs import unzip_run, zip_run
from src.results import NumbersOnlyUnpickler, event_names, stage_quartiles, stage_sd
from src.sustain_model import make_z_vals


def test_stage_sd_sure_and_unsure():
    sure = np.array([[0, 0, 1.0, 0]])                     # always stage 2
    split = np.array([[0, 0.5, 0.5, 0]])                  # stage 1 or 2
    assert stage_sd(sure)[0] == 0
    assert stage_sd(split)[0] == pytest.approx(0.5)


def test_stage_quartiles_are_real_stages():
    assert stage_quartiles(pd.Series([12, 13])) == [12, 12, 13]   # no 12.5


def test_event_names_follow_pysustain_order():
    Z_vals = make_z_vals(["A", "B"], {"A": [2, 5], "B": [2]})       # B has only one event
    assert event_names(Z_vals, ["A", "B"]) == ["A z2", "B z2", "A z5"]


def test_result_files_with_code_are_refused():
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo hacked",))
    with pytest.raises(pickle.UnpicklingError):
        NumbersOnlyUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()
    assert NumbersOnlyUnpickler(io.BytesIO(pickle.dumps({"x": np.arange(3)}))).load()["x"].sum() == 3


def test_run_zip_round_trip(tmp_path):
    run = tmp_path / "runs" / "my_run"
    run.mkdir(parents=True)
    (run / "settings.yaml").write_text("n_people: 1\n")
    (run / "pid.txt").write_text("123\n")                # never in the ZIP
    (tmp_path / "run.zip").write_bytes(zip_run(run))
    copy = unzip_run(tmp_path / "run.zip", tmp_path / "runs", "my_run")
    assert copy.name == "my_run_2"                        # never overwrites the original
    assert (copy / "settings.yaml").exists() and not (copy / "pid.txt").exists()


def test_zip_with_paths_outside_is_refused(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("settings.yaml", "x: 1")
        z.writestr("../escape.txt", "x")
    (tmp_path / "bad.zip").write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        unzip_run(tmp_path / "bad.zip", tmp_path / "runs", "bad")


def test_region_trend_tests_find_an_interaction():
    """A region that rises with age only in one group: the interaction p-value is small."""
    from src.qc import region_trend_tests
    rng = np.random.default_rng(0)
    age = rng.uniform(50, 90, 200)
    sex = np.where(np.arange(200) % 2 == 0, "F", "M")
    region = 1 + np.where(sex == "F", 0.02 * age, 0) + rng.normal(0, 0.05, 200)
    data = pd.DataFrame({"age": age, "sex": sex, "R1": region})
    table = region_trend_tests(data, ["R1"], "age", data["sex"], "sex")
    assert table.loc["R1", "p: age x sex (interaction)"] < 0.001


def test_covariate_interaction_is_tested_and_kept_with_its_own_terms():
    """age x sex: an age effect only in men is found, and sex stays in the model with the interaction."""
    from src.zscore import covariate_adjustment, covariate_matrix
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"age": rng.uniform(50, 85, 300), "sex": rng.choice(["F", "M"], 300)})
    data["R1"] = 1 + 0.02 * data["age"] * (data["sex"] == "M") + rng.normal(0, 0.05, 300)
    covariates = covariate_matrix(data, ["age", "sex"], [("age", "sex")])
    assert list(covariates.columns) == ["age", "sex=M", "age x sex=M"]
    _, p_values, _, slopes = covariate_adjustment(data, ["R1"], covariates, pd.Series(True, index=data.index))
    assert p_values.loc["R1", "age x sex=M"] < 0.001
    assert slopes.loc["R1", "age x sex=M"] == pytest.approx(0.02, abs=0.005)
    assert slopes.loc["R1", "sex=M"] != 0                  # kept because the interaction is kept


def test_typed_colours():
    """Hex codes (with or without #) are used, wrong ones are ignored; maps by name or by colours."""
    from src.figures import colour_list, colour_map, subtype_colors
    assert colour_list("#00897B, 112233, nonsense") == ["#00897B", "#112233"]
    assert colour_map("Blues", "Reds") == "Blues"
    assert colour_map("oops", "Reds") == "Reds"
    assert colour_map("#FFFFFF, #000000", "Reds").N == 256
    assert tuple(subtype_colors(2, ["#000000"])[0]) == (0, 0, 0, 1)   # typed first, default after


def test_report_holds_only_the_chosen_items():
    """The ZIP has the chosen figures and tables (numbered in order), the results per person and texts."""
    import matplotlib.pyplot as plt
    from src.report import build_report
    fig, ax = plt.subplots()
    d = {"per_person": pd.DataFrame({"subtype": [1]}), "texts": {"README.txt": "hi"}}
    data = build_report(d, {"Tau per subtype": fig}, {"Table 1: the subtypes compared": pd.DataFrame({"a": [1]})})
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert set(names) == {"figures/01_tau_per_subtype.png", "figures/01_tau_per_subtype.pdf",
                          "tables/01_table_1_the_subtypes_compared.csv", "tables/per_person.csv", "README.txt"}


def test_first_scan_is_the_baseline():
    """The earliest scan of each person is used (dates in any row order); the later ones are follow-ups."""
    from src.qc import split_baseline, visit_times
    data = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "date": ["2021-01-01", "2019-01-01", "2020-05-01",
                                                                    "2018-01-01", "2020-01-01"], "x": range(5)})
    first, later = split_baseline(data, "id", "date")
    assert list(first["x"]) == [1, 2, 3] and list(later["x"]) == [0, 4]
    assert visit_times(pd.Series(["2019-01-01", "2021-01-01"]))[1] == "years"
    first, later = split_baseline(data, "id", None)       # no visit column: first row, no follow-ups
    assert list(first["x"]) == [0, 2, 3] and len(later) == 0


def test_each_cohort_gets_its_own_normal():
    """Cohort B reads 20% higher: z-scored per cohort, the healthy people of both are around 0."""
    from src.zscore import reference_group_normal, zscore_by_cohort, zscore_new_scans
    rng = np.random.default_rng(0)
    cohorts = pd.Series(["A"] * 100 + ["B"] * 100)
    values = pd.DataFrame({"R1": rng.normal(1.2, 0.1, 200)})
    values.loc[cohorts == "B", "R1"] *= 1.2
    healthy = pd.Series(True, index=values.index)
    z, references = zscore_by_cohort(values, ["R1"], cohorts, lambda rows: reference_group_normal(
        values[rows], ["R1"], healthy[rows]))
    assert abs(z.groupby(cohorts)["R1"].mean()).max() < 1e-9
    new = zscore_new_scans(values.iloc[[0, 150]], ["R1"], cohorts.iloc[[0, 150]], references)
    assert np.allclose(new["R1"], z.iloc[[0, 150]]["R1"])  # the same normal for later scans


def test_same_covariate_adjustment_for_new_scans():
    """Applying the learned slopes to the same people gives the same adjusted values."""
    from src.zscore import apply_covariate_adjustment, covariate_adjustment, covariate_matrix
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"age": rng.uniform(50, 85, 200), "sex": rng.choice(["F", "M"], 200)})
    data["R1"] = 1 + 0.01 * data["age"] + rng.normal(0, 0.05, 200)
    covariates = covariate_matrix(data, ["age", "sex"])
    healthy = pd.Series(True, index=data.index)
    adjusted, _, _, slopes = covariate_adjustment(data, ["R1"], covariates, healthy, only_significant=False)
    again = apply_covariate_adjustment(data, ["R1"], covariates, slopes, covariates[slopes.columns].mean())
    assert np.allclose(again["R1"], adjusted["R1"])


def test_staging_the_model_people_again_gives_their_stage(one_run):
    """pySuStaIn's staging of new scans, given the model's own people, mostly gives back their stage."""
    import yaml
    from conftest import ROOT
    from src.results import load_sustain_result
    from src.sustain_model import stage_new_scans
    folder = ROOT / "results" / "app_runs" / one_run
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    result = load_sustain_result(folder, settings["dataset_name"], 2)
    z = pd.read_csv(folder / "zscores.csv", index_col="subject").iloc[:80]
    staged = stage_new_scans(z, settings["regions"], settings, result, 200)
    same = (staged["ml_stage"].ravel() == result["ml_stage"].ravel()[:80]).mean()
    assert same > 0.8


def test_simulation_check_pairs_subtypes_by_people_not_by_name():
    """SuStaIn's 'Subtype 1' can be the simulation's 'limbic': subtypes are paired by shared people."""
    from src.simulation_check import compare_with_truth, read_truth, summary_numbers
    found = pd.DataFrame({"subtype": ["S0", "Subtype 1", "Subtype 1", "Subtype 2", "Subtype 2", "Subtype 2"],
                          "ml_stage": [0, 5, 6, 10, 12, 3], "prob_subtype": 0.9},
                         index=["p1", "p2", "p3", "p4", "p5", "p6"])
    truth = read_truth(pd.DataFrame({"id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                                     "true": ["none", "limbic", "limbic", "posterior", "posterior", "limbic", "S0"],
                                     "stage": [0, 5, 7, 10, 12, 3, 0]}), "id", "true", "stage")
    rows, pairing = compare_with_truth(found, truth)
    assert pairing == {"Subtype 1": "limbic", "Subtype 2": "posterior"}
    assert len(rows) == 6                                  # p7 is not in the run
    numbers = summary_numbers(rows)
    assert numbers["subtype right (tau+ in both)"] == pytest.approx(4 / 5)
    assert numbers["S0 found (specificity)"] == 1
    assert numbers["stage error (mean absolute)"] == pytest.approx(1 / 5)


def test_three_groups_even_with_many_ties():
    """A number column always gives 3 groups (if it has 3+ different values), named by their values."""
    from src.qc import three_groups
    cdr = pd.Series([0] * 70 + [0.5] * 10 + [1, 2, 3, 4.5, 6, 8.5] * 3 + [np.nan] * 2)
    groups, names = three_groups(cdr)
    assert len(names) == 3 and groups.isna().sum() == 2
    age = pd.Series(np.arange(60.0, 90.0))
    groups, names = three_groups(age)
    assert names == ["60 to 69", "70 to 79", "80 to 89"]


def test_numbers_with_empty_cells_are_still_numbers():
    """memory_composite has empty cells: it is still a number column (not one category per value)."""
    from src.qc import number_and_category_columns
    data = pd.DataFrame({"memory": [0.5, -1.2, np.nan, np.nan, 0.3, 1.1], "sex": ["F", "M", "F", None, "M", "F"]})
    assert number_and_category_columns(data, ["memory", "sex"]) == (["memory"], ["sex"])


def test_run_time_is_shown_readably(tmp_path):
    """run_time.txt (seconds) -> a short text; no file -> "-"."""
    from src.app_runs import run_time
    assert run_time(tmp_path) == "-"
    for seconds, text in [(42, "42 s"), (750, "12 min 30 s"), (3900, "1 h 05 min")]:
        (tmp_path / "run_time.txt").write_text(f"{seconds}\n")
        assert run_time(tmp_path) == text


def test_pvd_title_tau_plus_count_matches_the_tables(one_run):
    """The PVD's 'tau+' count of each subtype is the number of tau+ people of that subtype in the tables."""
    import yaml
    from conftest import ROOT
    from src.figures import plot_pvd
    from src.results import load_sustain_result, pvd_order, size_order, subject_table
    folder = ROOT / "results" / "app_runs" / one_run
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    result = load_sustain_result(folder, settings["dataset_name"], 2)
    order = size_order(result)
    z = pd.read_csv(folder / "zscores.csv", index_col="subject")
    table = subject_table(result, z.index, 0.5, order)
    fig = plot_pvd(result, make_z_vals(settings["regions"], settings["cut_points"]), settings["regions"],
                   ["Subtype 1", "Subtype 2"], None, subtype_order=pvd_order(result, order))
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    for n, title in enumerate(titles, start=1):
        tau_plus = ((table["ml_subtype"] == n) & (table["ml_stage"] > 0)).sum()
        assert title.endswith(f"tau+ {tau_plus})"), title


def test_sample_data_has_age_and_sex_effects_that_the_adjustment_removes():
    """The sample data: in the healthy group SUVR rises with age, more in men (age x sex); after the app's
    covariate adjustment the z-scores no longer follow age, in women or men."""
    from scipy import stats
    from conftest import ROOT
    from src.zscore import (amyloid_negative_cn, covariate_adjustment, covariate_matrix, zscore_by_cohort,
                            zscore_regions)
    data = pd.read_csv(ROOT / "data" / "toy" / "sim_tau_400.csv")
    regions = [c for c in data.columns if c[:2] in ("L_", "R_")]
    healthy = amyloid_negative_cn(data, "diagnosis", "amyloid_positive", ["CN", "SCD"])
    covariates = covariate_matrix(data, ["age", "sex"], [("age", "sex")])
    adjusted, p_values, _, _ = covariate_adjustment(data, regions, covariates, healthy, True, 0.05)
    assert (p_values["age"] < 0.05).all() and (p_values["age x sex=M"] < 0.05).sum() >= 8
    for values, should_follow_age in [(data, True), (adjusted, False)]:
        z = zscore_by_cohort(values, regions, data["site"],
                             lambda rows: zscore_regions(values[rows], regions, 2, 42)[1])[0]
        men = healthy & (data["sex"] == "M")
        rho = stats.spearmanr(data.loc[men, "age"], z.loc[men, "L_temporal"])[0]
        assert (rho > 0.4) if should_follow_age else (abs(rho) < 0.2)


def test_brain_figures_colour_the_right_atlas_regions():
    """Lobe names colour all atlas regions of the lobe, atlas names just that region, sides kept apart."""
    from src.brain import atlas_regions_of, atlas_values
    assert "entorhinal" in atlas_regions_of("L_MTL") and "superiortemporal" not in atlas_regions_of("L_MTL")
    assert "superiortemporal" in atlas_regions_of("R_temporal")
    assert atlas_regions_of("lh_precuneus") == ["precuneus"]
    assert atlas_regions_of("subject_id") == []
    values = atlas_values({"L_frontal": 3.0, "R_frontal": 1.0})
    assert values[("L", "superiorfrontal")] == 3.0 and values[("R", "superiorfrontal")] == 1.0


def test_best_mcmc_sample_is_the_highest_likelihood_one(one_run):
    """The best MCMC sample is the one with the highest likelihood; its orders are real event orders."""
    import yaml
    from conftest import ROOT
    from src.results import best_sample_orders, event_orders, load_sustain_result, size_order
    folder = ROOT / "results" / "app_runs" / one_run
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    result = load_sustain_result(folder, settings["dataset_name"], 2)
    Z_vals = make_z_vals(settings["regions"], settings["cut_points"])
    best, orders = best_sample_orders(result, Z_vals, settings["regions"], size_order(result))
    assert best == int(np.argmax(result["samples_likelihood"].ravel())) + 1
    em = [e for e, _, _ in event_orders(result, Z_vals, settings["regions"], size_order(result))]
    assert all(sorted(o) == sorted(e) for o, e in zip(orders, em))      # the same events, maybe another order


def test_pairs_together():
    """Pairs of people of one subtype still in one subtype of the next model."""
    from src.figures import pairs_together
    flows = pd.DataFrame([[3, 1], [0, 2]], index=[1, 2], columns=[1, 2])
    per_subtype, everyone = pairs_together(flows)
    assert per_subtype[1] == 50 and per_subtype[2] == 100      # 3 of 6 pairs; 1 of 1 pair
    assert round(everyone, 1) == round(100 * 4 / 7, 1)


def test_subtype_fractions(one_run):
    """EM, best-sample and mean MCMC fractions: one per subtype, each set summing to about 1."""
    import yaml
    from conftest import ROOT
    from src.results import load_sustain_result, size_order, subtype_fractions
    folder = ROOT / "results" / "app_runs" / one_run
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    result = load_sustain_result(folder, settings["dataset_name"], 2)
    for f in subtype_fractions(result, size_order(result)):
        assert len(f) == 2 and abs(f.sum() - 1) < 1e-6
