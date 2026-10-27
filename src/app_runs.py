"""App tab 3 - one folder per SuStaIn run, and starting pySuStaIn in its own process.

A run folder holds: settings.yaml, run_time.txt (seconds the run took), zscores.csv, zscore_reference.csv (if any; one row per cohort and
region), covariates.csv (the other columns: age, sex, diagnosis, ... - used by tab 5),
followup_zscores.csv (later scans of the same people, if any - staged in tab 5), log.txt (all
that pySuStaIn prints), pid.txt (the process number while it runs), status.txt (finished / failed /
cancelled) and pySuStaIn's own files. A run folder can be downloaded as a ZIP and uploaded again.
"""
import io
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.zscore import reference_table

ROOT = Path(__file__).resolve().parent.parent          # project folder


def prepare_run_folder(folder, settings, z, reference=None, covariates=None, data=None, values=None,
                       followups=None):
    """Make a NEW run folder and save what the run needs. Fails if the folder already exists.
    reference: {cohort: normal reference}. data (the table used in tab 1) and values (the regional values
    that were z-scored in tab 2) are saved too, so loading the run later can show tabs 1 and 2 again.
    followups: z-scores of the later scans (made with the same normal), staged later with the model."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)          # never reuse: pySuStaIn would load old results
    with open(folder / "settings.yaml", "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    z.to_csv(folder / "zscores.csv", index_label="subject")
    if reference is not None:                           # needed later to z-score new people the same way
        reference_table(reference).to_csv(folder / "zscore_reference.csv", index=False)
    if covariates is not None:                          # same people and IDs as zscores.csv
        covariates.to_csv(folder / "covariates.csv", index_label="subject")
    if data is not None:
        data.to_csv(folder / "input_data.csv", index=False)
    if values is not None:
        values.to_csv(folder / "input_values.csv", index_label="subject")
    if followups is not None and len(followups):
        followups.to_csv(folder / "followup_zscores.csv", index_label="subject")


def read_run_inputs(folder):
    """What a saved run used in tabs 1-2, to show those tabs again when the run is loaded:
    {"data", "columns", "regions", "normalized", "complete"}. Runs made before the input data were
    saved give only the other columns (covariates.csv) and the z-scores: "complete" is then False."""
    folder = Path(folder)
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    columns = settings.get("columns", {})
    z = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str})
    reference = read_reference(folder / "zscore_reference.csv")
    complete = (folder / "input_data.csv").exists()
    if complete:
        data = pd.read_csv(folder / "input_data.csv")
        values = pd.read_csv(folder / "input_values.csv", index_col="subject", dtype={"subject": str})
    else:                                               # older run: other columns + IDs only
        data = pd.read_csv(folder / "covariates.csv", dtype={"subject": str})
        data = data.rename(columns={"subject": columns.get("id") or "subject"})
        values = None
    normalized = {"z": z, "reference": reference, "values": values, "method": settings["normalization"],
                  "n_components": settings.get("gmm_components"), "n_reference": None, "n_left_out": 0,
                  "inputs": None, "adjustment": None, "restored_from": folder.name, "cohorts": None,
                  "followup_z": None}
    return {"data": data, "columns": columns, "regions": settings["regions"], "normalized": normalized,
            "disease_columns": settings.get("disease_columns"),
            "complete": complete, "run": folder.name}

def read_reference(path):
    """The saved normal reference as {cohort: table with rows normal_mean, normal_sd}, or None.
    Runs made before cohorts were possible have no cohort column: one cohort, "all"."""
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if "cohort" not in table:
        table["cohort"] = "all"
    return {str(cohort): rows.set_index("region")[["normal_mean", "normal_sd"]].T
            for cohort, rows in table.groupby("cohort")}


def start_run(folder):
    """Start src/run_job.py for this folder as a separate process (so it can run for hours and be stopped)."""
    log = open(Path(folder) / "log.txt", "w")
    env = dict(os.environ, MPLBACKEND="Agg")             # figures go to files only, no windows
    process = subprocess.Popen([sys.executable, "-u", "-m", "src.run_job", str(folder)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    (Path(folder) / "pid.txt").write_text(f"{process.pid}\n")
    return process


def run_status(folder):
    """finished / failed / cancelled (from status.txt), else running - or stopped if its process is
    gone (e.g. the app was closed during the run)."""
    status, pid = folder / "status.txt", folder / "pid.txt"
    if status.exists():
        return status.read_text().strip()
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)            # signal 0 = only check that the process exists
            return "running"
        except (OSError, ValueError):
            pass
    return "stopped"


def run_time(folder):
    """How long a run took, e.g. "12 min 30 s" or "1 h 05 min" (from run_time.txt; "-" for runs made
    before the time was saved, or still running)."""
    path = Path(folder) / "run_time.txt"
    if not path.exists():
        return "-"
    seconds = int(float(path.read_text()))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s"


def list_runs(runs_folder):
    """One row per run folder, newest first: status, start time and the main settings."""
    rows = []
    for settings_file in Path(runs_folder).glob("*/settings.yaml"):
        folder, settings = settings_file.parent, yaml.safe_load(open(settings_file))
        rows.append({"run": folder.name, "status": run_status(folder),
                     "started": datetime.fromtimestamp(settings_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                     "people": settings["n_people"], "max subtypes": settings["n_subtypes_max"],
                     "start points": settings["n_startpoints"], "MCMC": settings["n_iterations_mcmc"],
                     "folds": settings["n_folds"] or "-", "run time": run_time(folder)})
    columns = ["run", "status", "started", "run time", "people", "max subtypes", "start points", "MCMC", "folds"]
    runs = pd.DataFrame(rows, columns=columns).sort_values("started", ascending=False)
    unfinished = ~runs["status"].isin(["finished", "running"])          # failed, cancelled, stopped: at the end
    return pd.concat([runs[~unfinished], runs[unfinished]])


def zip_run(folder):
    """The whole run folder as a ZIP file (bytes), to keep it outside the app."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and path.name != "pid.txt":
                zip_file.write(path, path.relative_to(folder))
    return buffer.getvalue()


def unzip_run(zip_path, runs_folder, name, max_mb=None):
    """Make a new run folder from a downloaded run ZIP. Returns the folder.
    Checks the ZIP first: it must have settings.yaml, only plain file names (nothing outside the folder)
    and, with max_mb, unpack to at most that many MB."""
    with zipfile.ZipFile(zip_path) as zip_file:
        names = zip_file.namelist()
        unpacked_mb = sum(info.file_size for info in zip_file.infolist()) / 1e6
        if max_mb is not None and unpacked_mb > max_mb:
            raise ValueError(f"The run is too big ({unpacked_mb:.0f} MB unpacked; the limit is {max_mb} MB).")
        if "settings.yaml" not in names:
            raise ValueError("This is not a Tau-typer run: settings.yaml is missing.")
        if any(n.startswith(("/", "\\")) or ".." in Path(n).parts for n in names):
            raise ValueError("The ZIP has file names outside its folder.")
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._") or "uploaded_run"
        folder, n = Path(runs_folder) / name, 2
        while folder.exists():                         # never overwrite a run
            folder, n = Path(runs_folder) / f"{name}_{n}", n + 1
        zip_file.extractall(folder)
    return folder
