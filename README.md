# Tau-typer

Applies an **already-trained** tau-PET **SuStaIn** model (Vogel et al. 2021, *Nature Medicine*) to new data and
returns each person's subtype and stage. Nothing is trained: the app only calls the **original pySuStaIn**
(`subtype_and_stage_individuals_newData`), never changed or copied. It follows the original
`Apply_MT_to_NewData` notebook and reproduces its results exactly on the simulated test data.

> **Stand-in model.** Until the lab's trained model files arrive, the app uses the invented model and
> simulated data in `data/simulated/`. Swapping in the real files only changes `config.yaml` (section `model`).

## Install (once)

The project uses one conda environment, with pySuStaIn pinned to commit `708fa22`:

```bash
bash code/01a_create_environment.sh
```

## Run the app

From the project folder (see `steps/S11_run_new_app.txt`):

```bash
conda activate ../env
```

```bash
shiny run app/app.py
```

Then open the address it prints (usually http://127.0.0.1:8000).

| Step | What happens |
|---|---|
| 1. Upload & QC | Upload regional tau-PET data (FreeSurfer regions, one row per scan). Column names are matched to the model's regions whatever their spelling (`ctx-lh-entorhinal`, `CTX_LH_ENTORHINAL`, `entorhinal_L`); regions not found can be chosen by hand. Choose ID / study / clinical columns, filter studies or diagnoses, QC, the 10 lobes vs the training data, explore, demographics. |
| 2. Normalize | z-scores on the model's scale. A: the training cohort's GMM normal, B: a GMM on your own data, C: your amyloid-negative CN people, D: values already z-scores. Normal table, GMM fits, z-score figures. |
| 3. Apply model | The saved model gives each person a subtype, stage and probabilities (stage 0 = S0). Optional false-positive cleanup (entorhinal / precuneus). The model's PVD, per-person table. |
| 4. Results | Tau pattern per subtype, lobes per subtype, the brain (Vogel Fig. 1c/1e), stages and their certainty, subtype probabilities, subtypes per study, clinical summary. |
| 5. Compare | The model applied after every normalization method, side by side, and against known labels (the simulated truth, or your own labels file): counts, agreement, adjusted Rand index, two-result cross-tables. |
| 6. Report | Results per person (CSV), every figure and table (ZIP), a PDF report. |

Try it with the simulated example data: step 1, "Use the simulated example data".

## Folders

| Folder | Contents |
|---|---|
| `app/` | the Shiny app: `app.py` joins the steps, one file per step (`tab1_upload.py` ... `tab6_report.py`, `tab_about.py`), `common.py` = shared settings and helpers |
| `src/` | all calculations: `trained_model.py` (load the model, lobes, apply, cleanup), `normalization.py`, `roi_names.py` (column matching), `compare.py`, `figures_apply.py` (+ older shared figure code) |
| `data/simulated/` | invented test data in the original input format + stand-in model + answer keys (safe to share) |
| `tests/` | automatic checks (`test_apply.py` = calculations, `test_app.py` = the whole app in memory) |
| `code/`, `steps/` | the step-by-step scripts used to build and check the pipeline |
| `config.yaml` | every setting in one place (section `model` = the trained model's files) |
| `reference/` | original notebooks and model files - **local only, not in git** |

## Tests

```bash
../env/bin/python -m pytest tests -q
```

`code/07a_check_pipeline_vs_reference.py` checks that the pipeline gives exactly the answer key made with the original
analysis code (`data/simulated/expected_reference_pipeline.csv`).

## Data privacy

Real participant data never goes into git (`data/` except `data/simulated/` and `data/toy/`, `reference/` and
`results/` are ignored). The app keeps uploads in memory on the computer that runs it.

## Cite

- Vogel JW et al. Four distinct trajectories of tau deposition identified in Alzheimer's disease.
  *Nature Medicine* 27, 871-881 (2021).
- Young AL et al. Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases
  with Subtype and Stage Inference. *Nature Communications* 9, 4273 (2018).
- Aksman LM et al. pySuStaIn: a Python implementation of the Subtype and Stage Inference algorithm.
  *SoftwareX* 16, 100811 (2021).

License: to be decided with the lab.
