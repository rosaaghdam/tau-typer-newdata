"""Shared test settings: the project folder on the path, and the inputs a browser would fill in."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = ROOT / "app" / "app.py"
TOY = ROOT / "data" / "toy" / "toy_tau_clinical.csv"
REGIONS = ["L_MTL", "R_MTL", "L_temporal", "R_temporal", "L_parietal", "R_parietal",
           "L_occipital", "R_occipital", "L_frontal", "R_frontal"]

# test_server has no browser, so boxes that the browser fills in (defaults, or choices the app sends)
# are set by hand - the same values the app starts with
BUTTONS = {name: 0 for name in ["normalize", "pr_reset", "run_button", "cancel_run", "load_run", "delete_run",
                                     "make_report", "make_pdf"]}
# under each figure: its colours (where it has them), as the app starts
TOOL_INPUTS = {"brain_colors": "#C8211B, #F26B1D, #FBB735, #F9E833", "regions_colors": "RdBu_r",
               "correlations_colors": "RdBu", "eda_colors": "", "boxplots_colors": "", "bars_colors": "",
               "disease_boxplots_colors": "", "disease_bars_colors": "",
               "zeda_colors": "", "sim_confusion_colors": "Blues", "subtype_colors": "#00897B, #E87BA4, #8A5A00, #9E9D24, #AD1457, #008300",
               "pvd_more": [], "adjust_interactions": []}
DATA_INPUTS = {**BUTTONS, **TOOL_INPUTS, "col_id": "subject_id", "col_age": "age", "col_sex": "sex", "col_diagnosis": "diagnosis",
               "col_amyloid": "amyloid_positive", "col_regions": REGIONS, "apply_rule": False, "col_cohort": "",
               "col_visit": ""}
NORM_INPUTS = {"norm_method": "gmm", "adjust_cov": False, "adjust_candidates": [], "adjust_only_sig": True,
               "adjust_alpha": 0.05, "gmm_components": 2, "per_region": False, "cut_points": "2, 5, 10",
               "z_max_method": "manual", "z_max_manual": "15", "z_max_percentile": 95, "gmm_region": "L_MTL",
               "zeda_x": "", "zeda_split": ""}
FIGURE_INPUTS = {**BUTTONS, **TOOL_INPUTS, "subtype_names": "", "sd_limit": 1.0, "stages_subtype": "all",
                 "stages_person": "", "fig_person": "TOY_0001", "eda_x": "", "eda_split": "", "an_variables": ["age", "sex", "MMSE"],
                 "an_stage_variable": "age", "an_adjust": ["age", "sex"]}


def toy_upload(path=TOY):
    """The value input_file gives the server after a CSV upload."""
    return [{"name": path.name, "size": path.stat().st_size, "type": "text/csv", "datapath": str(path)}]


def two_cohort_file(folder):
    """The toy data as two cohorts with a scan date: cohort B's scanner reads 15% higher, and 150 people
    have a second scan two years later (a little more tau). Saved as a CSV in folder."""
    import pandas as pd
    data = pd.read_csv(TOY)
    data["cohort"] = ["A" if i % 2 == 0 else "B" for i in range(len(data))]
    data.loc[data["cohort"] == "B", REGIONS] *= 1.15
    data["scan_date"] = "2019-03-01"
    later = data.sample(150, random_state=0).copy()
    later[REGIONS] = later[REGIONS] * 1.03 + 0.02
    later["scan_date"] = "2021-03-01"
    path = folder / "two_cohorts.csv"
    pd.concat([data, later]).sample(frac=1, random_state=1).to_csv(path, index=False)   # rows in any order
    return path


def finished_runs():
    """Finished runs in results/app_runs (not in git - the run tests are skipped without one)."""
    from src.app_runs import list_runs
    runs = list_runs(ROOT / "results" / "app_runs")
    finished = list(runs.loc[runs["status"] == "finished", "run"]) if len(runs) else []
    return [run for run in finished if not run.startswith("_")]   # "_..." = runs the tests make themselves


def errors(ts):
    """Outputs that failed (an empty output is fine: it waits for something)."""
    return {name: value.error for name, value in ts.to_values().outputs.items() if value.status == "error"}


@pytest.fixture
def one_run():
    runs = finished_runs()
    if not runs:
        pytest.skip("no finished run in results/app_runs")
    return runs[0]
