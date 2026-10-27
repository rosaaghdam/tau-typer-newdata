"""Step 3 - how fast is the app? Runs the app's server in memory (shiny.testserver, no browser)
and times what a user does most: show a saved run (tabs 3-5) and switch the model.

Run from the project folder:   ./env/bin/python code/03_benchmark_app.py [run name]
It needs a finished run in results/app_runs/ (default: the newest finished one).
Prints the median time of each action over 3 repeats, and which outputs failed or stayed empty.
"""
import statistics
import sys
import time
from pathlib import Path

from shiny.testserver import test_server

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.app_runs import list_runs

runs = list_runs(ROOT / "results" / "app_runs")
run = sys.argv[1] if len(sys.argv) > 1 else runs.loc[runs["status"] == "finished", "run"].iloc[0]
MODEL_INPUTS = ("result_model", "fig_model", "an_model")
# boxes a browser fills in itself (defaults, or choices sent by the app) - set by hand here
BOXES = {"normalize": 0, "pr_reset": 0, "run_button": 0, "cancel_run": 0, "load_run": 0, "delete_run": 0,
         "subtype_names": "", "fig_vmax": 12, "sd_limit": 1.0, "stages_subtype": "all", "stages_person": "",
         "fig_person": "TOY_0001", "an_variables": ["age", "sex", "MMSE"], "an_stage_variable": "age",
         "an_adjust": ["age", "sex"], "an_other_k": "2"}


def timed(action):
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


def one_session():
    """One visit: load the run, then switch every model box to 3 subtypes and back to 4."""
    times = {}
    with test_server(ROOT / "app" / "app.py", timeout_secs=300) as ts:
        other = runs.loc[(runs["status"] == "finished") & (runs["run"] != run), "run"].iloc[0]
        ts.set_inputs(saved_run=run, an_other_run=other, **BOXES)
        models = {i: "4" for i in MODEL_INPUTS}
        # the app also sets the model boxes when a run is loaded; the test server applies that one step later
        times["show a saved run (all tabs)"] = timed(lambda: (ts.set_inputs(saved_run=run, load_run=1, **models),
                                                               ts.set_inputs(**models)))
        times["switch to 3 subtypes"] = timed(lambda: ts.set_inputs(**{i: "3" for i in MODEL_INPUTS}))
        times["switch back to 4 subtypes"] = timed(lambda: ts.set_inputs(**{i: "4" for i in MODEL_INPUTS}))
        outputs = ts.to_values().outputs
    problems = {name: value.status for name, value in outputs.items() if value.status in ("error", "silent")}
    return times, problems, len(outputs)


results, problems = {}, {}
for repeat in range(3):
    times, problems, n_outputs = one_session()
    for action, seconds in times.items():
        results.setdefault(action, []).append(seconds)
print(f"Run {run}: {n_outputs} outputs; median of 3 repeats")
for action, seconds in results.items():
    print(f"  {action:32s} {statistics.median(seconds):6.2f} s   (all: {', '.join(f'{s:.2f}' for s in seconds)})")
print("Outputs with an error or empty:", problems or "none")
