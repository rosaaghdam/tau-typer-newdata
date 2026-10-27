# Tau-typer

Automated tau-PET subtyping with **SuStaIn**, following Vogel et al. 2021 (*Nature Medicine*).
A Shiny for Python app takes a table of regional tau-PET values, normalizes it, runs the
**original pySuStaIn** (never changed or copied) and shows the subtypes, stages, figures and
analyses.

## Install (once)

The project uses one conda environment inside the project folder (`./env`), with pySuStaIn
pinned to commit `708fa22`:

```bash
bash code/01a_create_environment.sh
```

## Run the app

From the project folder (see `steps/S10_run_app.txt`):

```bash
conda activate ./env
```

```bash
shiny run app/app.py
```

Then open the address it prints (usually http://127.0.0.1:8000).

| Tab | What you do there |
|---|---|
| 1. Data & QC | Upload a CSV (one row per person), match the columns, check the data, optional inclusion rule; explore the regions against age, sex, ... (with trend and interaction tests) |
| 2. Normalization | z-scores (GMM, amyloid-negative CN reference, or already normalized), cut-points, Z_max |
| 3. Run SuStaIn | Run pySuStaIn in the background (fast test or full run), cross-validation, saved runs |
| 4. Figures | PVD, tau patterns, subtype probabilities, stages, choosing k, one person |
| 5. Subtype analysis | Table 1, regional differences, stage vs age/cognition, asymmetry, compare runs |
| 6. Report | Automatic written summary, PDF report, ZIP of every figure and table, results per person |
| About | pySuStaIn version + commit, papers to cite, privacy |

Try it with the simulated toy data: tab 1 → "Sample Data" → upload that file.

## Folders

| Folder | Contents |
|---|---|
| `app/` | the Shiny app (user interface only): `app.py` joins the tabs, one file per tab (`tab1_data.py` ... `tab_about.py`), `common.py` = shared settings and texts |
| `tests/` | automatic checks of `src/` and of the app (run in memory, no browser) |
| `src/` | all calculations (they only *call* pySuStaIn) |
| `code/`, `steps/` | the step-by-step scripts used to build and check the pipeline |
| `config.yaml` | every setting in one place |
| `notes/` | plans, progress log, explanations (`SuStaIn.md`), review checklist |
| `data/toy/` | simulated toy data (safe to share) |
| `results/` | outputs and app runs - **not in git** |

## Tests and speed

From the project folder:

```bash
./env/bin/python -m pytest tests -q
```

```bash
./env/bin/python code/03_benchmark_app.py
```

The app tests that show a run need a finished run in `results/app_runs/` (they are skipped otherwise).
The benchmark times "show a saved run" and "switch model" with all tabs drawn at once.

## Data privacy

Real patient data never goes into git (`data/` and `results/` are ignored). The app keeps uploads and
results on the computer that runs it. On a shared server set `runs.private_per_visit: true` in
`config.yaml`: every visit then gets its own runs folder, which is deleted when the visit ends.

## Cite

- Vogel JW et al. Four distinct trajectories of tau deposition identified in Alzheimer's disease.
  *Nature Medicine* 27, 871-881 (2021).
- Young AL et al. Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases
  with Subtype and Stage Inference. *Nature Communications* 9, 4273 (2018).
- Aksman LM et al. pySuStaIn: a Python implementation of the Subtype and Stage Inference algorithm.
  *SoftwareX* 16, 100811 (2021).

License: to be decided with the lab.
