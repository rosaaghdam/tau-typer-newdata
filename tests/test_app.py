"""The app's server, run in memory (no browser): no output may fail in any step.
Run from the project folder:   ./env/bin/python -m pytest tests -q
"""
from shiny.testserver import test_server

from conftest import APP, errors

# test_server has no browser, so boxes the browser would fill in are set by hand (as the app starts)
START = {"up_col_id": "filename", "up_col_cohort": "study", "up_col_age": "age", "up_col_sex": "sex",
         "up_col_diagnosis": "diagnosis", "up_col_amyloid": "amyloid",
         "nm_method": "training", "nm_lobe": "L_MTL", "ap_cleanup": True, "ap_poor": 0.5, "ap_samples": 200,
         "rs_sd_limit": 1.0, "ap_run": 0, "cp_run": 0,
         "up_region_pick": ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]}


def ok(ts, *names):
    for name in names:
        assert ts.get_output(name).status == "ok", (name, ts.get_output(name).status)


def test_whole_app_with_the_example_data():
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(up_example=1, **START)
        ts.set_inputs(nm_lobe="R_MTL")                           # the test server shows some outputs one step later
        assert errors(ts) == {}
        ok(ts, "up_head", "up_match_table", "up_qc", "up_missing", "up_lobes_plot", "up_demographics",
           "nm_normals", "nm_gmm", "nm_hist", "nm_heat", "nm_z_table",
           "up_format_check", "up_format_table", "up_qc_note",
           "up_region_heat", "up_region_compare", "up_region_table",
           "up_region_hist", "up_region_box")
        assert "530 of 530 rows kept" in str(ts.get_output("up_data_note").value)
        assert "All 64 regions match the input template" in str(ts.get_output("up_qc_note").value)
        qc = ts.get_output("up_qc").value["payload"]
        assert list(qc["columns"]) == ["Check", "Result", "Status"]
        assert qc["data"][0] == ["Region columns match the input template", "64 of 64 same name", "OK"]
        assert ts.get_output("up_filter_note").value is None          # no filter used yet: no sentence

        ts.set_inputs(ap_run=1)                                  # step 3: apply the model
        ts.set_inputs(rs_sd_limit=1.5)                           # some outputs follow on the next update
        assert errors(ts) == {}
        ok(ts, "ap_counts", "ap_pvd", "ap_cards", "ap_onsets", "ap_cleanup_plot", "ap_people", "rs_summary", "rs_patterns", "rs_boxes",
           "rs_brain", "rs_stages", "rs_stage_hist", "rs_probs", "rs_studies", "rs_study_table", "rs_clinical",
           "rp_note")

        ts.set_inputs(cp_run=1)                                  # step 5: every method
        ts.set_inputs(cp_a="1. Reference pipeline: training cohort's GMM normal", cp_b="truth")
        assert errors(ts) == {}
        ok(ts, "cp_counts", "cp_counts_table", "cp_agreement", "cp_pair", "cp_pair_stages", "cp_normals")


def test_upload_with_other_column_names_and_a_region_chosen_by_hand(tmp_path):
    """CAPITALS_WITH_UNDERSCORES are found ('renamed'); a region under an odd name is chosen in the sidebar."""
    import pandas as pd
    from conftest import ROOT
    data = pd.read_csv(ROOT / "data" / "simulated" / "df_sustain_true.csv")
    data.columns = [c.upper().replace("-", "_") for c in data.columns]
    data = data.rename(columns={"CTX_LH_CUNEUS": "cuneus_left_side_value"})        # not recognisable
    path = tmp_path / "other_names.csv"
    data.to_csv(path, index=False)
    upload = [{"name": path.name, "size": path.stat().st_size, "type": "text/csv", "datapath": str(path)}]
    start = {**START, "up_col_id": "FILENAME", "up_col_cohort": "STUDY", "up_col_age": "AGE", "up_col_sex": "SEX",
             "up_col_diagnosis": "DIAGNOSIS", "up_col_amyloid": "AMYLOID"}
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(up_file=upload, **start)
        ts.set_inputs(nm_lobe="L_MTL")
        assert "missing" in str(ts.get_output("up_match_note").value)
        assert "regions missing" in str(ts.get_output("up_status").value)
        import re
        fix = re.findall(r'id="(up_fix_\d+)"', ts.get_output("up_fixes").value["html"])
        assert len(fix) == 1                                     # only ctx-lh-cuneus is not found
        ts.set_inputs(**{fix[0]: "cuneus_left_side_value"})
        ts.set_inputs(nm_lobe="L_MTL")
        assert errors(ts) == {}
        assert "530 of 530 rows kept" in str(ts.get_output("up_data_note").value)
        assert "chosen by you" in str(ts.get_output("up_match_note").value)


def test_inclusion_rule(tmp_path):
    """Ticking the paper's inclusion rule removes young CN and amyloid-negative MCI people, with the reasons."""
    import pandas as pd
    from conftest import ROOT
    data = pd.read_csv(ROOT / "data" / "simulated" / "df_sustain_true.csv")
    data.loc[data.index[data["diagnosis"] == "CN"][:5], "age"] = 35            # 5 CN people too young
    data.loc[data.index[data["diagnosis"] == "MCI"][:7], "amyloid"] = "Negative"  # 7 MCI people amyloid-negative
    path = tmp_path / "rule.csv"
    data.to_csv(path, index=False)
    upload = [{"name": path.name, "size": path.stat().st_size, "type": "text/csv", "datapath": str(path)}]
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(up_file=upload, **START)
        ts.set_inputs(up_rule=False)
        assert "530 of 530 rows kept" in str(ts.get_output("up_data_note").value)
        ts.set_inputs(up_rule=True)
        ts.set_inputs(nm_lobe="R_MTL")                           # the test server shows some outputs one step later
        assert errors(ts) == {}
        assert "518 of 530 rows kept (inclusion rule on)" in str(ts.get_output("up_data_note").value)
        note = str(ts.get_output("up_filter_note").value)
        assert "Kept: 518 of 530 people (98%)." in note and "12 by the inclusion rule" in note
        ok(ts, "up_rule_table", "nm_normals")
        reasons = ts.get_output("up_rule_table").value["payload"]["data"]
        assert [row[1] for row in reasons] == [286, 232, 5, 7, 0]


def test_other_normalization_methods_and_a_filter():
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(up_example=1, **START)
        ts.set_inputs(up_keep_cohort=["bf2", "a4"])
        ts.set_inputs(nm_method="reference")                     # the test server shows some outputs one step later
        assert errors(ts) == {}
        assert "374 of 530 rows kept" in str(ts.get_output("up_data_note").value)
        note = str(ts.get_output("up_filter_note").value)
        assert "Kept: 374 of 530 people (71%)." in note and "156 by study" in note
        ok(ts, "up_studies", "nm_normals")
        ts.set_inputs(nm_method="own_gmm", ap_run=1)
        ts.set_inputs(nm_lobe="R_Frontal")
        assert errors(ts) == {}
        ok(ts, "nm_gmm", "ap_counts", "rs_patterns")
