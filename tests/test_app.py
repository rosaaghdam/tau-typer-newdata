"""The app's server, run in memory (no browser): no output may fail in the main steps.
Run from the project folder:   ./env/bin/python -m pytest tests -q
"""
from shiny.testserver import test_server

from conftest import (REGIONS, APP, DATA_INPUTS, FIGURE_INPUTS, NORM_INPUTS, errors, finished_runs, toy_upload,
                      two_cohort_file)


def load(ts, run, k="2"):
    """Load a saved run in tab 3 and choose the model in tabs 3-5. The model boxes are set twice: the
    app also sets them when a run is loaded, and the test server applies that one step later."""
    models = {"result_model": k, "fig_model": k, "an_model": k}
    ts.set_inputs(saved_run=run, load_run=1, **models)
    ts.set_inputs(**models)


def test_upload_and_normalize():
    """Tabs 1-2 with the toy data: upload, match columns, normalize (GMM)."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(**DATA_INPUTS)                      # the test server shows some outputs one step later
        assert errors(ts) == {}
        assert ts.get_output("qc_table").status == "ok"
        ts.set_inputs(normalize=1)
        ts.set_inputs(normalize=2)                        # a few outputs follow only on the next update
        assert errors(ts) == {}
        assert ts.get_output("z_head").status == "ok"            # z-scores were made
        assert ts.get_output("events_table").status == "ok"


def test_normalize_with_reference_group_and_covariates():
    """Reference-group normalization with age adjusted first."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(norm_method="reference", adjust_cov=True, adjust_candidates=["age", "sex"],
                      adjust_interactions=["age x sex"], normalize=1)
        ts.set_inputs(normalize=2)                        # a few outputs follow only on the next update
        assert errors(ts) == {}
        assert ts.get_output("slopes_table").status == "ok"
        assert "age x sex" in str(ts.get_output("slopes_table").value)   # the interaction was tested


def test_show_saved_run(one_run):
    """Tabs 3-5: load a finished run; every figure and table renders without an error."""
    others = [run for run in finished_runs() if run != one_run]
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, an_other_run=others[0] if others else "", an_other_k="2",
                      **FIGURE_INPUTS)
        load(ts, one_run)
        assert errors(ts) == {}
        for name in ["subtype_summary_table", "event_order_view", "fig_pvd", "fig_probs", "fig_stages",
                     "fig_tree", "an_table1", "an_regions_plot"]:
            assert ts.get_output(name).status == "ok", name
        ts.set_inputs(pvd_more=["1"], subtype_colors="#000000, 112233", brain_colors="#FFFFFF, #0F5C6E")
        assert errors(ts) == {}
        assert "Model with 1 subtype" in str(ts.get_output("fig_pvd_more").value)
        assert "1 figure(s)" not in str(ts.get_output("report_contents").value)   # many figures ticked


def test_s0_people_have_no_subtype(one_run):
    """Stage 0 = S0 in the summary table, as in the paper (pySuStaIn itself still gives them a subtype)."""
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        rows = ts.get_output("subtype_summary_table").value["payload"]["data"]
        assert rows[0][0] == "S0"


def test_report_is_made_in_the_background(one_run):
    """Tab 6: 'Make report (ZIP)' and 'Make PDF report' build the files in a separate process; then the
    download buttons appear. (Both in one test: background jobs of two test sessions clash.)"""
    import time
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **{**FIGURE_INPUTS, "tree_report": False, "person_report": True})
        load(ts, one_run)
        ts.set_inputs(make_report=1)
        for _ in range(120):                              # wait up to 2 minutes
            if ts.get_output("report_ready").status == "ok" and ts.get_output("report_ready").value:
                break
            time.sleep(1)
            ts.set_inputs(tick=_)                         # lets the app check whether the job is done
        assert errors(ts) == {}
        assert "Download" in str(ts.get_output("report_ready").value)
        ts.set_inputs(make_pdf=1)
        wait_for(ts, "pdf_ready")
        assert errors(ts) == {}
        assert "Download" in str(ts.get_output("pdf_ready").value)


def test_inclusion_rule_note_and_region_exploration():
    """Tab 1: the note under the rule checkbox, and regions plotted against age, split by sex, with tests."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS, eda_x="", eda_split="")
        ts.set_inputs(apply_rule=True)
        ts.set_inputs(apply_rule=True)                    # the note sits in a box the app adds itself
        assert "people kept" in str(ts.get_output("rule_note").value)
        ts.set_inputs(eda_x="age", eda_split="sex")
        assert errors(ts) == {}
        assert ts.get_output("region_hist").status == "ok"
        assert "interaction" in str(ts.get_output("eda_tests").value)


def wait_for(ts, output, seconds=120):
    """Wait until a background job has filled 'output' (lets the app check every second)."""
    import time
    for _ in range(seconds):
        if ts.get_output(output).status == "ok" and ts.get_output(output).value:
            return
        time.sleep(1)
        ts.set_inputs(tick=_)


def test_report_summary(one_run):
    """Tab 6: the written summary of the run."""
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        assert "have abnormal tau" in str(ts.get_output("report_summary").value)


def test_loading_a_run_fills_tabs_1_and_2(one_run, tmp_path):
    """A run saved with its input data: loading it shows the data in tab 1 and its z-scores in tab 2."""
    import shutil
    import pandas as pd
    from conftest import ROOT, TOY
    runs = ROOT / "results" / "app_runs"
    copy = runs / "_test_restore"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(runs / one_run, copy)
    try:
        z = pd.read_csv(copy / "zscores.csv", index_col="subject")
        sample = ROOT / "data" / "toy" / "sim_tau_400.csv"           # the file the run was made from
        data = pd.read_csv(sample if str(z.index[0]).startswith("SIM") else TOY)
        data[data["subject_id"].isin(z.index)].to_csv(copy / "input_data.csv", index=False)
        data.set_index("subject_id").loc[z.index, list(z.columns)].to_csv(copy / "input_values.csv",
                                                                        index_label="subject")
        with test_server(APP, timeout_secs=300) as ts:
            ts.set_inputs(saved_run="_test_restore", **FIGURE_INPUTS)
            load(ts, "_test_restore")
            assert "_test_restore" in str(ts.get_output("data_status").value)
            assert ts.get_output("data_head").status == "ok"
            assert "as saved with the run" in str(ts.get_output("norm_status").value)
            assert ts.get_output("z_head").status == "ok"
            assert errors(ts) == {}
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def test_split_by_a_number_with_many_ties():
    """MMSE: many people have 30, so equal thirds are impossible - the split must still work."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS, eda_x="", eda_split="")
        ts.set_inputs(eda_x="age", eda_split="MMSE")
        assert errors(ts) == {}
        assert ts.get_output("region_hist").status == "ok"
        ts.set_inputs(eda_x="")
        assert errors(ts) == {}


def test_explore_the_z_scores():
    """Tab 2: z-scores against plasma p-tau217, split by sex, with the per-region tests."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(normalize=1)
        ts.set_inputs(normalize=2)                        # a few outputs follow only on the next update
        ts.set_inputs(zeda_x="plasma_ptau217", zeda_split="sex")
        ts.set_inputs(zeda_x="plasma_ptau217", zeda_split="sex")   # the app also refills these boxes
        assert errors(ts) == {}
        assert ts.get_output("z_hist").status == "ok"
        assert "interaction" in str(ts.get_output("zeda_tests").value)


def test_two_cohorts_and_follow_up_scans(tmp_path):
    """Tabs 1-2: the first scan of each person is used, later scans are kept; each cohort gets its own normal
    (with either method), and the follow-up scans are z-scored with it."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(two_cohort_file(tmp_path)), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(col_cohort="cohort", col_visit="scan_date")
        ts.set_inputs(col_cohort="cohort", col_visit="scan_date")   # the test server shows it one step later
        status = str(ts.get_output("data_status").value)
        assert "650 scans of 500 people" in status and "2 cohorts" in status
        for clicks, method in [(1, "reference"), (3, "gmm")]:
            ts.set_inputs(norm_method=method, adjust_cov=True, adjust_candidates=["age"], normalize=clicks)
            ts.set_inputs(normalize=clicks + 1)               # a few outputs follow only on the next update
            assert errors(ts) == {}
            norm = str(ts.get_output("norm_status").value)
            assert "each of the 2 cohorts" in norm and "150 follow-up scans" in norm
            assert "cohort" in str(ts.get_output("norm_reference").value)


def test_follow_up_scans_and_cohorts_of_a_run(one_run):
    """Tab 5: a run with follow-up scans and a cohort column - the scans are staged by the model, and the
    subtypes are counted per cohort."""
    import shutil
    import pandas as pd
    import yaml
    from conftest import ROOT
    runs = ROOT / "results" / "app_runs"
    copy = runs / "_test_followup"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(runs / one_run, copy)
    try:
        z = pd.read_csv(copy / "zscores.csv", index_col="subject", dtype={"subject": str})
        later = z.iloc[:60] + 0.5                        # 60 people, a little more tau two visits later
        later["visit"] = 3
        later.to_csv(copy / "followup_zscores.csv", index_label="subject")
        covariates = pd.read_csv(copy / "covariates.csv", index_col="subject", dtype={"subject": str})
        covariates["site"] = ["A" if i % 2 else "B" for i in range(len(covariates))]
        covariates["visit"] = 1
        covariates.to_csv(copy / "covariates.csv", index_label="subject")
        settings = yaml.safe_load(open(copy / "settings.yaml"))
        settings.setdefault("columns", {}).update(cohort="site", visit="visit")
        yaml.safe_dump(settings, open(copy / "settings.yaml", "w"), sort_keys=False)
        with test_server(APP, timeout_secs=300) as ts:
            ts.set_inputs(saved_run="_test_followup", **FIGURE_INPUTS)
            load(ts, "_test_followup")
            assert errors(ts) == {}
            assert "60 people have 60 follow-up scans" in str(ts.get_output("an_followup_status").value)
            assert ts.get_output("an_followup_plot").status == "ok"
            assert "2 cohorts" in str(ts.get_output("an_cohort_status").value)
            assert ts.get_output("an_cohort_plot").status == "ok"
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def test_simulation_check_with_a_perfect_truth(one_run, tmp_path):
    """Simulation check: a truth file that is exactly the run's own result (renamed) gives 100%."""
    import pandas as pd
    import yaml
    from conftest import ROOT
    from src.results import load_sustain_result, size_order, subject_table
    folder = ROOT / "results" / "app_runs" / one_run
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    result = load_sustain_result(folder, settings["dataset_name"], 2)
    z = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str})
    table = subject_table(result, z.index, 0.5, size_order(result))
    names = {"S0": "S0", "Subtype 1": "posterior", "Subtype 2": "limbic"}
    truth = pd.DataFrame({"subject_id": table.index, "true_subtype": table["subtype"].map(names),
                          "true_stage": table["ml_stage"]})
    path = tmp_path / "truth.csv"
    truth.to_csv(path, index=False)
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        for _ in range(2):
            ts.set_inputs(truth_file=toy_upload(path), truth_id="subject_id", truth_subtype="true_subtype",
                          truth_stage="true_stage", sim_model="2")
        assert errors(ts) == {}
        assert "Subtype 1 = posterior" in str(ts.get_output("sim_numbers").value)
        assert "100.0" in str(ts.get_output("sim_subtype_table").value)     # every subtype found right
        for name in ["sim_confusion_plot", "sim_stage_plot", "sim_subtype_table", "sim_people"]:
            assert ts.get_output(name).status == "ok", name


def test_disease_columns_are_chosen_in_tab_1():
    """The disease-related columns are guessed by name, can be changed in tab 1, are shown apart in
    Demographics, and tab 2 warns only for the ones marked in tab 1."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(adjust_cov=True, adjust_candidates=["age", "MMSE"])
        assert "MMSE is marked as disease-related" in str(ts.get_output("adjust_warning").value)   # by name
        assert "Disease-related columns by diagnosis" in str(ts.get_output("covariates_view").value)
        assert ts.get_output("disease_boxplots").status == "ok"
        for _ in range(2):
            ts.set_inputs(col_disease=["CDR_SB"])           # the user's own choice: MMSE not in it
        assert ts.get_output("adjust_warning").value is None or "MMSE" not in str(ts.get_output("adjust_warning").value)
        assert errors(ts) == {}


def test_sample_data_inclusion_rule_and_one_scan_note():
    """The sample data: the inclusion rule leaves out 10 people, and a visit column with one scan per person
    says that nothing changes."""
    from conftest import ROOT
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(ROOT / "data" / "toy" / "sim_tau_400.csv"), **DATA_INPUTS, **NORM_INPUTS)
        for _ in range(2):
            ts.set_inputs(apply_rule=True, col_visit="age")
        assert "390 of 400 people kept" in str(ts.get_output("rule_note").value)
        assert ts.get_output("missing_plot").status == "ok"
        assert "only one row, so nothing changes" in str(ts.get_output("data_status").value)


def test_every_column_of_the_downloadable_data_can_split_and_be_plotted_against():
    """Region distributions (tab 1): every other column of the sample data (and the old toy data), as 'Plot
    against' and as 'Split by', draws without an error; number columns give 3 groups."""
    import pandas as pd
    from conftest import ROOT, TOY
    for path in [ROOT / "data" / "toy" / "sim_tau_400.csv", TOY]:
        data = pd.read_csv(path)
        others = [c for c in data.columns if c not in REGIONS + ["subject_id"]]
        with test_server(APP, timeout_secs=300) as ts:
            ts.set_inputs(data_file=toy_upload(path), **DATA_INPUTS, **NORM_INPUTS)
            for column in others:
                for x, split in [("", column), (column, ""), ("age", column)]:
                    ts.set_inputs(eda_x=x, eda_split=split)
                    ts.set_inputs(eda_x=x, eda_split=split)
                    assert errors(ts) == {}, (path.name, x, split)
                    assert ts.get_output("region_hist").status == "ok", (path.name, x, split)


def test_z_max_too_low_for_the_data_is_flagged():
    """Sample data with Z_max 15: in most regions more than 5% of the tau+ people go past it - tab 2 says so."""
    from conftest import ROOT
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(ROOT / "data" / "toy" / "sim_tau_400.csv"), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(normalize=1)
        ts.set_inputs(normalize=2)
        assert "below the 95th percentile" in str(ts.get_output("zmax_status").value)
        for _ in range(2):
            ts.set_inputs(z_max_method="percentile", z_max_percentile=95)
        assert "covers the data" in str(ts.get_output("zmax_status").value)


def test_upload_summary_appears_after_the_upload():
    """Nothing before an upload; then the number of samples (rows) and variables (columns) of the file."""
    from conftest import ROOT
    with test_server(APP, timeout_secs=120) as ts:
        assert "Data uploaded" not in str(ts.get_output("upload_summary").value)
        ts.set_inputs(data_file=toy_upload(ROOT / "data" / "toy" / "sim_tau_400.csv"), **DATA_INPUTS, **NORM_INPUTS)
        for _ in range(2):
            ts.set_inputs(col_cohort="site")
        columns = len(open(ROOT / "data" / "toy" / "sim_tau_400.csv").readline().split(","))
        assert f"400 samples, {columns} variables" in str(ts.get_output("upload_summary").value)


def test_eda_report_and_its_pdf():
    """EDA report: every part is drawn for the sample data (3 sites) and the toy data, and the PDF is made."""
    from conftest import ROOT
    from src.eda import build_eda_pdf, eda_sections
    import pandas as pd
    for path in [ROOT / "data" / "toy" / "sim_tau_400.csv", None]:
        with test_server(APP, timeout_secs=300) as ts:
            ts.set_inputs(data_file=toy_upload(path) if path else toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
            for _ in range(2):
                ts.set_inputs(col_cohort="site" if path else "")
            assert errors(ts) == {}
            report = str(ts.get_output("eda_report").value)
            assert "Missing values" in report and "Correlations" in report
            assert ("Differences between site groups" in report) == bool(path)
    data = pd.read_csv(ROOT / "data" / "toy" / "sim_tau_400.csv")
    roles = {"id": "subject_id", "diagnosis": "diagnosis", "cohort": "site"}
    sections = eda_sections(data, roles, REGIONS)
    assert build_eda_pdf(sections, "EDA").startswith(b"%PDF")


def test_no_two_elements_share_an_id():
    """Every id in the page is used once (a repeated id, e.g. two downloads with the same name, breaks one)."""
    import re
    from collections import Counter
    import importlib.util
    spec = importlib.util.spec_from_file_location("tau_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html = str(module.app_ui)
    repeated = [name for name, n in Counter(re.findall(r'\sid="([^"]+)"', html)).items() if n > 1]
    assert repeated == []
