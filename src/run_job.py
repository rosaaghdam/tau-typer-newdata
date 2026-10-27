"""Run the original pySuStaIn for one app run folder, in its own process.

Started by the app (src/app_runs.py, start_run), or by hand from the project folder:
    python -m src.run_job results/app_runs/<run name>
Reads settings.yaml + zscores.csv from the folder; pySuStaIn writes its files there.
At the end writes status.txt: "finished", or "failed" (the error is then at the end of log.txt), and
run_time.txt: how many seconds the run took.
"""
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from src.sustain_model import cross_validate, run_sustain, sustain_inputs


def main(folder):
    settings = yaml.safe_load(open(folder / "settings.yaml"))
    z = pd.read_csv(folder / "zscores.csv", index_col="subject")
    data, Z_vals, Z_max = sustain_inputs(z, settings["regions"], settings["cut_points"], settings["z_max"])
    print(f"Run {folder.name}: {data.shape[0]} people x {data.shape[1]} regions", flush=True)
    model, _ = run_sustain(data, Z_vals, Z_max, settings["regions"], settings, str(folder), settings["seed"])
    if settings["n_folds"]:                             # None = no cross-validation
        cvic, loglike = cross_validate(model, data.shape[0], settings["n_folds"], settings["seed"])
        models = pd.RangeIndex(1, len(cvic) + 1, name="subtypes")
        pd.DataFrame({"CVIC": cvic, "mean test log-likelihood": loglike.mean(axis=0)},
                     index=models).to_csv(folder / "cross_validation.csv")
    (folder / "status.txt").write_text("finished\n")


if __name__ == "__main__":
    run_folder = Path(sys.argv[1])
    start = time.time()
    try:
        main(run_folder)
    except Exception:
        traceback.print_exc()                           # the error goes to log.txt
        (run_folder / "status.txt").write_text("failed\n")
        sys.exit(1)
    finally:
        (run_folder / "run_time.txt").write_text(f"{time.time() - start:.0f}\n")   # seconds, shown in tab 3
